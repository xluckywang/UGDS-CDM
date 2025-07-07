import torch
import torchvision
import torch.nn as nn
from vmk_30 import VMK
import torch.optim as optim
import torchvision.transforms as transforms
import sys
import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time
from ptflops import get_model_complexity_info
import argparse
import warnings
warnings.filterwarnings("ignore")

sys.path.append(os.path.abspath('./../'))
from Image_Preprocessing import preprocess_data


def main(args):
    # 设置设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 当前时间
    start_time = time.time()

    # 数据处理
    train_dataset = preprocess_data(args.traindata)
    val_dataset = preprocess_data(args.valdata)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                               shuffle=True, num_workers=args.num_workers)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size,
                                             shuffle=False, num_workers=args.num_workers)
    # 假设 trainLoader 是您的数据加载器
    for images, labels in train_loader:
        print("Image batch shape:", images.shape)
        # 获取每张图片的大小
        print(f"Image size (Height x Width): {images.shape[2]} x {images.shape[3]}")
        break  # 只查看第一个批次的图片大小

    net = VMK().to(device)
    net.to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(net.parameters(), lr=args.lr, betas=(args.beta1, 0.999),weight_decay=0.0001)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.8)
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=6, verbose=True, min_lr=3.78e-7, eps=1e-10)
    # 引入余弦退火学习率调度器
    #scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0)

    # 获取模型参数和复杂度
    flops, params = get_model_complexity_info(net, (1, 224, 224), as_strings=False, print_per_layer_stat=False)
    if params is None:
        p=0
    total_params = params/1e6  # 将参数数量转换为百万单位
    flops = flops  # 模型复杂度（FLOPs）

    # 初始化度量指标的字典
    metrics = {
        'epoch': [],
        'train_loss': [],
        'train_accuracy': [],
        'train_precision': [],
        'train_recall': [],
        'train_f1': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_precision': [],
        'val_recall': [],
        'val_f1': [],
        'epoch_duration': [],
        'inference_time_ms': [],
        'FPS': [],
        'total_params(M)': [],
        'flops': []
    }

    best_val_accuracy = 0.0
    best_model_state = None
    best_train_metrics = None
    best_val_metrics = None
    patience = args.patience
    no_improve_epochs = 0
    last_val_accuracy = None

    for epoch in range(args.epochs):  # loop over the dataset multiple times
        net.train()
        running_loss = 0.0
        all_train_predictions = []
        all_train_labels = []

        for step, data in enumerate(train_loader, start=0):
            inputs, labels, *_ = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            # 累加损失
            running_loss += loss.item()

            # Store predictions and labels for train metrics
            _, predicted = torch.max(outputs, dim=1)
            all_train_predictions.extend(predicted.cpu().numpy())
            all_train_labels.extend(labels.cpu().numpy())



        # Calculate train metrics
        train_loss = running_loss / len(train_loader)
        train_accuracy = accuracy_score(all_train_labels, all_train_predictions)
        train_precision = precision_score(all_train_labels, all_train_predictions, average='weighted')
        train_recall = recall_score(all_train_labels, all_train_predictions, average='weighted')
        train_f1 = f1_score(all_train_labels, all_train_predictions, average='weighted')
        # 每个 epoch 结束时更新学习率
        scheduler.step(train_loss)

        net.eval()
        running_val_loss = 0.0
        all_val_predictions = []
        all_val_labels = []
        # Inference time and FPS calculation
        torch.cuda.synchronize()
        start_inference = time.time()
        with torch.no_grad():
            for step, data in enumerate(val_loader, start=0):
                inputs, labels, *_ = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = net(inputs)
                outputs = outputs.to(device)
                loss = loss_function(outputs, labels)
                running_val_loss += loss.item()

                # Store predictions and labels for val metrics
                _, predicted = torch.max(outputs, dim=1)
                all_val_predictions.extend(predicted.cpu().numpy())
                all_val_labels.extend(labels.cpu().numpy())
        torch.cuda.synchronize()
        end_inference = time.time()
        inference_time = (end_inference - start_inference) / len(val_loader.dataset) * 1000  # 单张图片推理时间，以毫秒为单位
        FPS = 1 / (inference_time / 1000)  # Frames per second


        # Calculate validation metrics
        val_loss = running_val_loss / len(val_loader)
        val_accuracy = accuracy_score(all_val_labels, all_val_predictions)
        val_precision = precision_score(all_val_labels, all_val_predictions, average='weighted')
        val_recall = recall_score(all_val_labels, all_val_predictions, average='weighted')
        val_f1 = f1_score(all_val_labels, all_val_predictions, average='weighted')
        epoch_duration = time.time() - start_time



        # with torch.no_grad():
        #     for step, data in enumerate(val_loader, start=0):
        #         inputs, *_ = data
        #         inputs = inputs.to(device)
        #         outputs = net(inputs)



        # Save metrics
        metrics['epoch'].append(epoch + 1)
        metrics['train_loss'].append(train_loss)
        metrics['train_accuracy'].append(train_accuracy)
        metrics['train_precision'].append(train_precision)
        metrics['train_recall'].append(train_recall)
        metrics['train_f1'].append(train_f1)
        metrics['val_loss'].append(val_loss)
        metrics['val_accuracy'].append(val_accuracy)
        metrics['val_precision'].append(val_precision)
        metrics['val_recall'].append(val_recall)
        metrics['val_f1'].append(val_f1)
        metrics['epoch_duration'].append(epoch_duration)
        metrics['inference_time_ms'].append(inference_time)
        metrics['FPS'].append(FPS)
        metrics['total_params(M)'].append(total_params)
        metrics['flops'].append(flops)

        # Print metrics for this epoch
        print(f"[{epoch + 1}] train_loss: {train_loss:.3f}  train_accuracy: {train_accuracy:.3f}  "
              f"train_precision: {train_precision:.3f}  train_recall: {train_recall:.3f}  train_f1: {train_f1:.3f}  "
              f"val_loss: {val_loss:.3f}  val_accuracy: {val_accuracy:.3f}  "
              f"val_precision: {val_precision:.3f}  val_recall: {val_recall:.3f}  val_f1: {val_f1:.3f}  "
              f"duration: {epoch_duration:.3f}ms  inference_time: {inference_time:.3f}ms "
              f"FPS: {FPS:.3f} total_params(M): {total_params}, flops: {flops}")

        # Check for improvement and update best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model_state = net.state_dict()
            best_train_metrics = (train_loss, train_accuracy, train_precision, train_recall, train_f1)
            best_val_metrics = (val_loss, val_accuracy, val_precision, val_recall, val_f1)
            no_improve_epochs = 0
            torch.save(best_model_state, args.best_model_path)
        elif val_accuracy == last_val_accuracy:
            no_improve_epochs += 1
        else:
            no_improve_epochs = 0

        last_val_accuracy = val_accuracy

        # Early stopping
        if no_improve_epochs >= patience:
            print("Early stopping triggered")
            break

    print('Finished Training')

    # Save best model
    if best_model_state is not None:
        # 输出最优模型时的训练集和验证集指标
        print(f"Best model train metrics: loss: {best_train_metrics[0]:.3f}, accuracy: {best_train_metrics[1]:.3f}, "
              f"precision: {best_train_metrics[2]:.3f}, recall: {best_train_metrics[3]:.3f}, f1: {best_train_metrics[4]:.3f}")
        print(f"Best model val metrics: loss: {best_val_metrics[0]:.3f}, accuracy: {best_val_metrics[1]:.3f}, "
              f"precision: {best_val_metrics[2]:.3f}, recall: {best_val_metrics[3]:.3f}, f1: {best_val_metrics[4]:.3f}")

    # Convert metrics dictionary to DataFrame and save to
    df = pd.DataFrame(metrics)
    df.to_excel(args.output_path, index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="LeNet Training Script")

    # 可选参数
    parser.add_argument('--traindata', type=str, default='./../../../Data/Data/train_30',
                        help='Path to the training data')

    parser.add_argument('--valdata', type=str, default='./../../../Data/Data/val_270', help='Path to the validation data')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training and validation')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of workers for data loading')
    parser.add_argument('--lr', type=float, default=0.0005, help='Learning rate')#0.0007  0.00088
    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs to train')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--best_model_path', type=str, default='./bestzt_vmknet.pth',
                        help='Path to save the best model')
    parser.add_argument('--output_path', type=str, default='bestzt_vmknet.xlsx',
                        help='Path to save the metrics Excel file')
    parser.add_argument('--beta1', type=float, default=0.90, help='beta1')

    args = parser.parse_args()
    main(args)
