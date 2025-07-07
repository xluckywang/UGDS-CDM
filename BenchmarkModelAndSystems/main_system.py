import os
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import datasets
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import time
import numpy as np
from datetime import datetime
from io import BytesIO
import base64
from LeNet import LeNet
from AlexNet import AlexNet
from VGG16 import vgg as VGG16
from GoogLeNet import GoogLeNet
from ResNet34 import resnet34 as ResNet34
from MobileNetV2 import MobileNetV2 as MobileNet
from ShuffleNetV2 import shufflenet_v2_x1_0 as ShuffleNet
from DenseNet import  densenet121 as DenseNet
from EfficientNet import efficientnet_b0 as  EfficientNet
from RegNet import  create_regnet as RegNet
from SwinViT import  swin_tiny_patch4_window7_224  as SwinViT
# from vmk import VMK as vmkan
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from torchvision.models import resnet18
import torch.optim as optim
import seaborn as sns
from torchvision import models
from utils_rlt import GradCAM, show_cam_on_image, center_crop_img
import cv2
defect_labels = {
    0: 'bubble',
    1: 'dust',
    2: 'fouling',
    3: 'no defects',
    4: 'pinhole',
    5: 'sagging',
    6: 'scratch',
    7: 'shrink'
}
#显示图片
def plt_to_base64(img):
    buf = BytesIO()
    plt.figure(figsize=(2, 1))
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    plt.close()
    return image_base64

def load_images_from_folder(folder):  # 下载数据集
    images = []
    for subfolder in os.listdir(folder):
        subfolder_path = os.path.join(folder, subfolder)
        if os.path.isdir(subfolder_path):
            for filename in os.listdir(subfolder_path):
                file_path = os.path.join(subfolder_path, filename)
                if os.path.isfile(file_path):  # 确保是文件而不是文件夹
                    images.append((subfolder, filename, file_path))
    return images

# 加载模型
def load_model(model, weight_path, device):
    model = model.to(device)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()
    return model

# 显示图像
def display_images(images, titles, cols=5):
    rows = (len(images) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    for i, (img, title) in enumerate(zip(images, titles)):
        ax = axes[i // cols, i % cols]
        ax.imshow(img, cmap='gray')
        ax.set_title(title)
        ax.axis('off')
    for j in range(i + 1, rows * cols):
        axes[j // cols, j % cols].axis('off')
    st.pyplot(fig)

def plot_combined_summary(train_data, val_data, test_data):
    # 统计每个数据集中的缺陷数量
    train_summary = train_data['Defect'].value_counts().reset_index()
    train_summary.columns = ['Defect', 'Train']
    val_summary = val_data['Defect'].value_counts().reset_index()
    val_summary.columns = ['Defect', 'Validation']
    test_summary = test_data['Defect'].value_counts().reset_index()
    test_summary.columns = ['Defect', 'Test']

    # 合并数据集统计信息
    summary = pd.merge(train_summary, val_summary, on='Defect', how='outer').merge(test_summary, on='Defect', how='outer')
    summary = summary.fillna(0)  # 填充空值为0
    summary['Total'] = summary['Train'] + summary['Validation'] + summary['Test']

    # 绘制图表
    fig, ax = plt.subplots(figsize=(12, 8))
    width = 0.25

    train_summary = train_summary.set_index('Defect')
    val_summary = val_summary.set_index('Defect')
    test_summary = test_summary.set_index('Defect')

    defect_types = summary['Defect']

    x = range(len(defect_types))

    ax.bar(x, summary['Train'], width=width, label='Train')
    ax.bar([p + width for p in x], summary['Validation'], width=width, label='Validation')
    ax.bar([p + width * 2 for p in x], summary['Test'], width=width, label='Test')

    ax.set_xlabel('Defect Type')
    ax.set_ylabel('Count')
    ax.set_title('Dataset Summary')
    ax.set_xticks([p + 1.5 * width for p in x])
    ax.set_xticklabels(defect_types, rotation=45, ha='right')
    ax.legend()

    for rect in ax.patches:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2., height + 0.1, int(height), ha='center', va='bottom')

    st.pyplot(fig)

    # 显示表格并美化
    summary = summary.set_index('Defect')
    summary_styled = summary.style.set_properties(**{
        'text-align': 'center',
        'font-size': '12pt'  # 调整字体大小
    }).set_table_styles([
        {'selector': 'thead th', 'props': [('background-color', '#f4f4f4'), ('font-weight', 'bold'), ('text-align', 'center'), ('font-size', '14pt')]},
        {'selector': 'tbody td', 'props': [('text-align', 'center'), ('font-size', '14pt')]},
        {'selector': 'tbody tr:nth-child(even)', 'props': [('background-color', '#f9f9f9')]},
        {'selector': 'tbody tr:nth-child(odd)', 'props': [('background-color', '#ffffff')]}
    ])
    st.table(summary_styled)

#数据预处理
def data_preprocessing(hsize, wsize):
    data_transform = transforms.Compose([
        transforms.Grayscale(),  # 灰度处理
        transforms.Resize((hsize, wsize)),  # 修改为用户输入的图像大小
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # 归一化
    ])
    return data_transform
#数据预处理
def preprocess_data(data_root, hsize, wsize):
    data_transform = data_preprocessing(hsize, wsize)
    dataset = datasets.ImageFolder(root=data_root, transform=data_transform)
    return dataset


#分析训练集特征
def analyze_training_features(train_folder, save_path='defect_types_tsne.png'):
    if 'feature_plot' not in st.session_state:
        categories = ['bubble', 'dust', 'fouling', 'no defects','pinhole', 'sagging', 'scratch', 'shrink']

        # 图像预处理
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        # 使用预训练的ResNet18提取特征
        model = resnet18(pretrained=True)
        model.eval()
        feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])  # 去掉最后的分类层

        # 加载图像并提取特征
        features = []
        labels = []

        for label, category in enumerate(categories):
            category_dir = os.path.join(train_folder, category)
            for filename in os.listdir(category_dir):
                filepath = os.path.join(category_dir, filename)
                image = Image.open(filepath).convert('RGB')
                image = transform(image).unsqueeze(0)

                with torch.no_grad():
                    feature = feature_extractor(image).squeeze().numpy()
                features.append(feature)
                labels.append(label)

        features = np.array(features)
        labels = np.array(labels)

        # 对特征进行标准化
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        # 使用 T-SNE 进行降维
        tsne = TSNE(n_components=2, random_state=42, perplexity=30, learning_rate=200)
        X_embedded = tsne.fit_transform(features_scaled)

        # 可视化
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=labels, cmap='tab10', s=20, alpha=0.7)
        plt.colorbar(scatter, ticks=range(len(categories)), label='Defect Type')
        plt.clim(-0.5, len(categories) - 0.5)
        plt.title('T-SNE of Defect Types')
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')

        # 保存图像
        plt.savefig(save_path)
        st.session_state['feature_plot'] = save_path

    st.image(st.session_state['feature_plot'], caption='T-SNE of Defect Types')

class ResizeTransform:
    def __init__(self, im_h: int, im_w: int):
        self.height = self.feature_size(im_h)
        self.width = self.feature_size(im_w)

    @staticmethod
    def feature_size(s):
        s = math.ceil(s / 4)  # PatchEmbed
        s = math.ceil(s / 2)  # PatchMerging1
        s = math.ceil(s / 2)  # PatchMerging2
        s = math.ceil(s / 2)  # PatchMerging3
        return s

    def __call__(self, x):
        result = x.reshape(x.size(0),
                           self.height,
                           self.width,
                           x.size(2))
        result = result.permute(0, 3, 1, 2)
        return result


def main():

    st.set_page_config(layout="wide", page_title="Human–machine interaction surface defect detection system")#网页名字

    st.markdown("""
        <style>
            .centered-title1 {
                position: absolute;
                top: -60px;   /* 调整标题的垂直位置 */
                left: 960px;  /* 调整标题的水平位置 */
                font-size: 48px;  /* 调整字体大小 */
                text-align: center;
                z-index: 1;  /* 确保标题位于框的上方 */
            }
        </style>
        <style>
            .centered-title2 {
                position: absolute;
                top: 1510px;   /* 调整标题的垂直位置 */
                left: 1600px;  /* 调整标题的水平位置 */
                font-size: 28px;  /* 调整字体大小 */
                text-align: center;
                z-index: 1;  /* 确保标题位于框的上方 */
            }
        </style>
        <style>
            .centered-title3 {
                position: absolute;
                top: 1560px;   /* 调整标题的垂直位置 */
                left: 1510px;  /* 调整标题的水平位置 */
                font-size: 24px;  /* 调整字体大小 */
              
                z-index: 1;  /* 确保标题位于框的上方 */
            }
        </style>
        <style>
            .centered-title4 {
                position: absolute;
                top: 1560px;   /* 调整标题的垂直位置 */
                left: 1960px;  /* 调整标题的水平位置 */
                font-size: 24px;  /* 调整字体大小 */
               
                z-index: 1;  /* 确保标题位于框的上方 */
            }
        </style>
        <style>
            .centered-title5 {
                position: absolute;
                top: 1560px;   /* 调整标题的垂直位置 */
                left: 2410px;  /* 调整标题的水平位置 */
                font-size: 24px;  /* 调整字体大小 */
               
                z-index: 1;  /* 确保标题位于框的上方 */
            }
        </style>
        <style>
            .centered-title6 {
                position: absolute;
                top: 1560px;   /* 调整标题的垂直位置 */
                left: 1060px;  /* 调整标题的水平位置 */
                font-size: 24px;  /* 调整字体大小 */
             
                z-index: 1;  /* 确保标题位于框的上方 */
            }
        </style>
        <style>
            .centered-title7 {
                position: absolute;
                top: 1560px;   /* 调整标题的垂直位置 */
                left: 610px;  /* 调整标题的水平位置 */
                font-size: 24px;  /* 调整字体大小 */
                
                z-index: 1;  /* 确保标题位于框的上方 */
            }
        </style>
        <div style="
            width: 2900px; 
            height: 2100px; 
            border: 2px solid black; 
            position: absolute; 
            top: 20px; 
            left: -20px;
            z-index: 0;">
        </div>
        <div style="
            width: 570px; 
            height: 150px; 
            border: 2px solid black; 
            position: absolute; 
            top: 80px; 
            left: -10px;
            z-index: 0;">
        </div>
        <div style="
            width: 570px; 
            height: 850px; 
            border: 2px solid black; 
            position: absolute; 
            top: 235px; 
            left: -10px;
            z-index: 0;">
        </div>
        <div style="
            width: 570px; 
            height: 460px; 
            border: 2px solid black; 
            position: absolute; 
            top: 1090px; 
            left: -10px;
            z-index: 0;">
        </div>
        <div style="
            width: 570px; 
            height: 390px; 
            border: 2px solid black; 
            position: absolute; 
            top: 1560px; 
            left: -10px;
            z-index: 0;">
        </div>
       <div style="
            width: 1125px; 
            height: 490px; 
            border: 2px solid black; 
            position: absolute; 
            top: 80px; 
            left: 565px;
            z-index: 0;">
        </div>
        <div style="
            width: 1125px; 
            height: 940px; 
            border: 2px solid black; 
            position: absolute; 
            top: 575px; 
            left: 565px;
            z-index: 0;">
        </div>
        <div style="
            width: 2300px; 
            height: 580px; 
            border: 2px solid black; 
            position: absolute; 
            top: 1520px; 
            left: 565px;
            z-index: 0;">
        </div>
        <div style="
            width: 1125px; 
            height: 200px; 
            border: 2px solid black; 
            position: absolute; 
            top: 80px; 
            left: 1695px;
            z-index: 0;">
        </div>
        <div style="
            width: 1125px; 
            height: 1230px; 
            border: 2px solid black; 
            position: absolute; 
            top: 285px; 
            left: 1695px;
            z-index: 0;">
        </div>
        <h1 class='centered-title1'>🔧 Human–machine interaction surface defect detection system</h1>
       
        """, unsafe_allow_html=True)
    config_col, train_col, test_col = st.columns([1, 2, 2])

    with config_col:
        st.header("⚙️ Preparation process")

        st.subheader("📂 Select dataset")
        dataset_folder = st.text_input('⚡️ Enter the path of the dataset folder⚡',
                                       help='Input the path of the dataset folder (e.g., C:\\Users\\Lenovo\\PycharmProjects\\jiance\\Data\\Data).')

        if dataset_folder:
            train_folder = os.path.join(dataset_folder, 'train')
            val_folder = os.path.join(dataset_folder, 'val')
            test_folder = os.path.join(dataset_folder, 'test')

            if os.path.exists(train_folder) and os.path.exists(val_folder) and os.path.exists(test_folder):
                train_images = load_images_from_folder(train_folder)
                val_images = load_images_from_folder(val_folder)
                test_images = load_images_from_folder(test_folder)

                train_data = pd.DataFrame(train_images, columns=['Defect', 'Filename', 'Filepath'])
                val_data = pd.DataFrame(val_images, columns=['Defect', 'Filename', 'Filepath'])
                test_data = pd.DataFrame(test_images, columns=['Defect', 'Filename', 'Filepath'])

                st.subheader("📄 Dataset situation")
                plot_combined_summary(train_data, val_data, test_data)
                st.subheader("🔖 Training dataset feature analysis")
                st.session_state['show_feature_plot'] = True
                # analyze_training_features(train_folder)



            else:
                st.warning("The selected folder path does not exist")
                return

            st.subheader("✅ Determine parameters")


            # 创建3行2列的布局
            col1, col2 = st.columns(2)
            with col1:
                model_name = st.selectbox('Select Model', [
                    'LeNet', 'AlexNet', 'VGG16', 'GoogLeNet', 'ResNet34', 'MobileNet',
                    "ShuffleNet", 'DenseNet', 'EfficientNet', 'RegNet', 'SwinViT',
                    # 'vmkan',
                ])
            with col2:
                batch_size = st.number_input('Batch size', value=32, min_value=16, max_value=128)
            col3, col4 = st.columns(2)
            with col3:
                lr = st.number_input('Initial LR', value=0.001, min_value=0.0, max_value=1.0)
            with col4:
                epochs = st.number_input('Training epoch', value=10, min_value=1, max_value=200)
            col5, col6 = st.columns(2)
            with col5:
                hsize = st.number_input('Crop size height', value=224, min_value=1, max_value=500)
            with col6:
                wsize = st.number_input('Crop size width', value=224, min_value=1, max_value=500)


            save_model_path = st.text_input('⬇️ Model weight storage address', value='./model.pth')

            st.markdown(
                """
                <style>
                .center-button {
                    position: absolute;
                    top: 300px;
                    left: 500px;
                }
                .stButton button {
                    background-color: #4CAF50; /* 绿色背景 */
                    color: white; /* 白色文字 */
                    padding: 6px 32px; /* 按钮内边距 */
                    font-size: 16px; /* 字体大小 */
                    border-radius: 8px; /* 圆角 */
                    border: none; /* 无边框 */
                    cursor: pointer; /* 鼠标变为手型 */
                    transition: background-color 0.3s ease; /* 动画过渡 */
                }
                .stButton button:hover {
                    background-color: #45a049; /* 鼠标悬停时变色 */
                }
                </style>
                """,
                unsafe_allow_html=True
            )

            # 创建一个容器，并将按钮居中
            with st.container():
                st.markdown('<div class="center-button">', unsafe_allow_html=True)
                if st.button("Start preprocessing"):
                    st.session_state['preprocessed'] = True
                    st.session_state['training'] = False
                    st.session_state['detecting'] = False
                    # 预处理数据集
                    preprocess_data(dataset_folder, hsize, wsize)
                    st.success("Preprocessing complete!")


    if 'preprocessed' in st.session_state:
        with train_col:
            st.header("💡 training process")

            # 定义样式
            style = """
            <style>
                .centered-text {
                    text-align: center;
                    font-family: 'Arial', sans-serif;
                    font-size: 14px;
                    line-height: 1.6;
                }
                .title {
                    font-size: 18px;
                    font-weight: bold;
                }
                .highlight {
                    font-size: 20px;
                    color: #FF6347;
                }
                .metrics-table {
                    margin: 0 auto;
                    width: 80%;
                    border-collapse: collapse;
                }
                .metrics-table td {
                    padding: 8px;
                    border: 1px solid #dddddd;
                }
            </style>
            """
            st.markdown(style, unsafe_allow_html=True)

            epoch_text = st.empty()
            time_text = st.empty()
            metrics_text = st.empty()


            if st.button("started training"):
                st.session_state['training'] = True
                st.session_state['detecting'] = False

                # 训练模型
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                start_time = time.time()

                train_loader = torch.utils.data.DataLoader(preprocess_data(train_folder, hsize, wsize),
                                                           batch_size=batch_size,
                                                           shuffle=True, num_workers=8)
                val_loader = torch.utils.data.DataLoader(preprocess_data(val_folder, hsize, wsize),
                                                         batch_size=batch_size,
                                                         shuffle=False, num_workers=8)

                # 定义模型字典
                model_dict = {
                    'LeNet': LeNet,
                    'AlexNet': AlexNet,
                    'VGG16': VGG16,
                    'GoogLeNet': GoogLeNet,
                    'ResNet34': ResNet34,
                    'MobileNet': MobileNet,
                    'ShuffleNet': ShuffleNet,
                    'DenseNet': DenseNet,
                    'EfficientNet': EfficientNet,
                    'RegNet': RegNet,
                    'SwinViT': SwinViT,
                    # 'vmkan': vmkan,
                }

                # 初始化模型
                if model_name in model_dict:
                    if model_name == 'VGG16':
                        net = VGG16(model_name="vgg16", num_classes=8).to(device)
                    elif model_name == 'GoogLeNet':
                        net = GoogLeNet(num_classes=8, aux_logits=False).to(device)
                    elif model_name == 'MobileNet':
                        net = MobileNet(num_classes=8).to(device)
                    elif model_name == 'ShuffleNet':
                        net = ShuffleNet(num_classes=8).to(device)
                    elif model_name == 'RegNet':
                        net = RegNet(model_name="RegNetY_400MF", num_classes=8).to(device)
                    else:
                        net = model_dict[model_name](num_classes=8).to(device)
                else:
                    raise ValueError(f"Unknown model name: {model_name}")

                # 定义损失函数和优化器
                loss_function = nn.CrossEntropyLoss()
                optimizer = optim.Adam(net.parameters(), lr=lr)

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
                    'epoch_duration': []
                }

                progress_bar = st.progress(0)

                for epoch in range(epochs):
                    net.train()
                    running_loss = 0.0
                    all_train_predictions = []
                    all_train_labels = []
                    if model_name==GoogLeNet:
                        for step, data in enumerate(train_loader, start=0):
                            inputs, labels, *_ = data
                            inputs, labels = inputs.to(device), labels.to(device)
                            optimizer.zero_grad()
                            logits, aux_logits2, aux_logits1 = net(inputs.to(device))
                            loss0 = loss_function(logits, labels.to(device))
                            loss1 = loss_function(aux_logits1, labels.to(device))
                            loss2 = loss_function(aux_logits2, labels.to(device))
                            loss = loss0 + loss1 * 0.3 + loss2 * 0.3
                            loss.backward()
                            optimizer.step()
                            # 累加损失
                            running_loss += loss.item()
                            outputs = logits

                            # Store predictions and labels for train metrics
                            _, predicted = torch.max(outputs, dim=1)
                            all_train_predictions.extend(predicted.cpu().numpy())
                            all_train_labels.extend(labels.cpu().numpy())
                    else:
                        for step, data in enumerate(train_loader, start=0):
                            inputs, labels = data
                            inputs, labels = inputs.to(device), labels.to(device)
                            optimizer.zero_grad()
                            outputs = net(inputs)
                            loss = loss_function(outputs, labels)
                            loss.backward()
                            optimizer.step()
                            running_loss += loss.item()

                            _, predicted = torch.max(outputs, dim=1)
                            all_train_predictions.extend(predicted.cpu().numpy())
                            all_train_labels.extend(labels.cpu().numpy())

                    train_loss = running_loss / len(train_loader)
                    train_accuracy = accuracy_score(all_train_labels, all_train_predictions)
                    train_precision = precision_score(all_train_labels, all_train_predictions, average='weighted')
                    train_recall = recall_score(all_train_labels, all_train_predictions, average='weighted')
                    train_f1 = f1_score(all_train_labels, all_train_predictions, average='weighted')

                    net.eval()
                    running_val_loss = 0.0
                    all_val_predictions = []
                    all_val_labels = []
                    with torch.no_grad():
                        for step, data in enumerate(val_loader, start=0):
                            inputs, labels = data
                            inputs, labels = inputs.to(device), labels.to(device)
                            outputs = net(inputs)
                            loss = loss_function(outputs, labels)
                            running_val_loss += loss.item()

                            _, predicted = torch.max(outputs, dim=1)
                            all_val_predictions.extend(predicted.cpu().numpy())
                            all_val_labels.extend(labels.cpu().numpy())

                    val_loss = running_val_loss / len(val_loader)
                    val_accuracy = accuracy_score(all_val_labels, all_val_predictions)
                    val_precision = precision_score(all_val_labels, all_val_predictions, average='weighted')
                    val_recall = recall_score(all_val_labels, all_val_predictions, average='weighted')
                    val_f1 = f1_score(all_val_labels, all_val_predictions, average='weighted')
                    epoch_duration = time.time() - start_time

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

                    progress_bar.progress((epoch + 1) / epochs)
                    epoch_text.markdown(f"<p class='centered-text highlight'>Epoch: {epoch + 1}/{epochs}</p>",
                                        unsafe_allow_html=True)
                    time_text.markdown(
                        f"<p class='centered-text highlight'>Beijing Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>",
                        unsafe_allow_html=True)

                    metrics_text.markdown(
                        f"""
                        <div class='centered-text'>
                            <p class='title'>Training set:</p>
                            <table class='metrics-table'>
                                <tr><td>loss</td><td>Acc</td><td>Pre</td><td>Rec</td><td>F1</td><td>Total training time (s)</td></tr>
                                <tr>
                                    <td>{train_loss:.3f}</td>
                                    <td>{train_accuracy:.3f}</td>
                                    <td>{train_precision:.3f}</td>
                                    <td>{train_recall:.3f}</td>
                                    <td>{train_f1:.3f}</td>
                                    <td>{epoch_duration:.2f}</td>
                                </tr>
                            </table>
                            <p class='title'>validation set:</p>
                            <table class='metrics-table'>
                                <tr><td>loss</td><td>Acc</td><td>Pre</td><td>Rec</td><td>F1</td></tr>
                                <tr>
                                    <td>{val_loss:.3f}</td>
                                    <td>{val_accuracy:.3f}</td>
                                    <td>{val_precision:.3f}</td>
                                    <td>{val_recall:.3f}</td>
                                    <td>{val_f1:.3f}</td>
                                </tr>
                            </table>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                st.success("Training complete!")

                save_path = save_model_path
                torch.save(net.state_dict(), save_path)

                df = pd.DataFrame(metrics)
                df.to_excel('training_metrics.xlsx', index=False)

                # 保存图表数据
                st.session_state['metrics'] = metrics
                st.session_state['train_cm'] = confusion_matrix(all_train_labels, all_train_predictions)
                st.session_state['val_cm'] = confusion_matrix(all_val_labels, all_val_predictions)

            # 重新加载训练信息和图表
            if 'metrics' in st.session_state:
                metrics = st.session_state['metrics']
                train_cm = st.session_state['train_cm']
                val_cm = st.session_state['val_cm']

                progress_bar = st.progress(1.0)

                # 中心化的显示
                epoch_text = st.empty()
                time_text = st.empty()
                metrics_text = st.empty()

                # 显示 Epoch 和时间
                epoch_text.markdown(f"<p class='centered-text highlight'>Epoch: {metrics['epoch'][-1]}/{epochs}</p>",
                                    unsafe_allow_html=True)
                time_text.markdown(
                    f"<p class='centered-text highlight'>Beijing Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>",
                    unsafe_allow_html=True)

                # 显示训练和验证集的指标
                metrics_text.markdown(
                    f"""
                    <div class='centered-text'>
                        <p class='title'>traindation set:</p>
                        <table class='metrics-table'>
                            <tr><td>loss</td><td>Acc</td><td>Pre</td><td>Rec</td><td>F1</td><td>Total training time (s)</td></tr>
                            <tr>
                                <td>{metrics['train_loss'][-1]:.3f}</td>
                                <td>{metrics['train_accuracy'][-1]:.3f}</td>
                                <td>{metrics['train_precision'][-1]:.3f}</td>
                                <td>{metrics['train_recall'][-1]:.3f}</td>
                                <td>{metrics['train_f1'][-1]:.3f}</td>
                                <td>{metrics['epoch_duration'][-1]:.2f}</td>
                            </tr>
                        </table>
                        <p class='title'>Validation set:</p>
                        <table class='metrics-table'>
                            <tr><td>loss</td><td>Acc</td><td>Pre</td><td>Rec</td><td>F1</td></tr>
                            <tr>
                                <td>{metrics['val_loss'][-1]:.3f}</td>
                                <td>{metrics['val_accuracy'][-1]:.3f}</td>
                                <td>{metrics['val_precision'][-1]:.3f}</td>
                                <td>{metrics['val_recall'][-1]:.3f}</td>
                                <td>{metrics['val_f1'][-1]:.3f}</td>
                            </tr>
                        </table>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                fig, ax = plt.subplots(2, 2, figsize=(12, 10))

                ax[0, 0].plot(metrics['epoch'], metrics['train_loss'], label='Train Loss')
                ax[0, 0].plot(metrics['epoch'], metrics['val_loss'], label='Validation Loss')
                ax[0, 0].set_title('Loss')
                ax[0, 0].legend()

                ax[0, 1].plot(metrics['epoch'], metrics['train_accuracy'], label='Train Accuracy')
                ax[0, 1].plot(metrics['epoch'], metrics['val_accuracy'], label='Validation Accuracy')
                ax[0, 1].set_title('Accuracy')
                ax[0, 1].legend()

                sns.heatmap(train_cm, annot=True, fmt="d", ax=ax[1, 0])
                ax[1, 0].set_title('Train Confusion Matrix')

                sns.heatmap(val_cm, annot=True, fmt="d", ax=ax[1, 1])
                ax[1, 1].set_title('Validation Confusion Matrix')

                st.pyplot(fig)

        st.markdown("""
            <style>
            .reportview-container .main .block-container{
                max-width: 95%;
                padding-top: 2rem;
                padding-right: 2rem;
                padding-left: 2rem;
                padding-bottom: 2rem;
            }
            </style>
            """, unsafe_allow_html=True)


        with test_col:
            st.header("💥 Testing process")
            st.subheader("📌 Select model weights")
            weight_path = st.text_input('⬆️ Enter the path of the model weight file', value='./model.pth')

            if st.button("Preprocess test set"):
                test_data = preprocess_data(test_folder, hsize, wsize)
                st.session_state['test_data'] = test_data
                st.session_state['test_index'] = 0
                st.session_state['predictions'] = []
                st.session_state['confidences'] = []
                st.session_state['detection_results'] = None


            if 'test_data' in st.session_state:
                test_data = st.session_state['test_data']
                test_index = st.session_state['test_index']

                image_placeholder = st.empty()
                info_placeholder = st.empty()
                table_placeholder = st.empty()

                def show_test_image(index):
                    img, _ = test_data[index]
                    img = img.numpy().transpose((1, 2, 0))
                    img = np.clip(img, 0, 1)  # 归一化处理

                    # 进行预测，获取类别
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    model_dict = {
                        'LeNet': LeNet,
                        'AlexNet': AlexNet,
                        'VGG16': VGG16,
                        'GoogLeNet': GoogLeNet,
                        'ResNet34': ResNet34,
                        'MobileNet': MobileNet,
                        'ShuffleNet': ShuffleNet,
                        'DenseNet': DenseNet,
                        'EfficientNet': EfficientNet,
                        'RegNet': RegNet,
                        'SwinViT': SwinViT,
                        # 'vmkan': vmkan,
                    }

                    model = load_model(model_dict[model_name](), weight_path, device)
                    model.eval()

                    img_tensor = transforms.ToTensor()(img).unsqueeze(0).to(device)
                    with torch.no_grad():
                        outputs = model(img_tensor)
                        _, predicted = torch.max(outputs, 1)
                        target_category = predicted.item()

                    # 假设模型和目标层的映射已经定义
                    model_target_layers = {
                        'LeNet': lambda model: [model.features[-1]],
                        'AlexNet': lambda model: [model.features[-1]],
                        'VGG16': lambda model: [model.features[-1]],
                        'GoogLeNet': lambda model: [model.inception5b],
                        'ResNet34': lambda model: [model.features[-1]],
                        'MobileNet': lambda model: [model.features[-1]],
                        'ShuffleNet': lambda model: [model.conv5],
                        'DenseNet': lambda model: [model.features[-2]],
                        'EfficientNet': lambda model: [model.features[-1]],
                        'RegNet': lambda model: [model.s4],
                        'SwinViT': lambda model: [model.norm],
                        # 'vmkan': lambda model: [model.swin],
                    }

                    # 生成热力图
                    if model_name in model_target_layers:
                        target_layers = model_target_layers[model_name](model)

                        # 处理 SwinViT 的特殊情况
                        if model_name == 'SwinViT':
                            cam = GradCAM(
                                model=model,
                                target_layers=target_layers,
                                use_cuda=torch.cuda.is_available(),
                                reshape_transform=ResizeTransform(im_h=224, im_w=224)
                            )
                        else:
                            cam = GradCAM(
                                model=model,
                                target_layers=target_layers,
                                use_cuda=torch.cuda.is_available()
                            )

                        # 调用 GradCAM
                        grayscale_cam = cam(input_tensor=img_tensor, target_category=target_category)
                        grayscale_cam = grayscale_cam[0, :]
                        visualization = show_cam_on_image(img.astype(dtype=np.float32), grayscale_cam, use_rgb=True)
                    else:
                        raise ValueError(f"Unknown model name: {model_name}")

                    # 转换图像为 base64
                    img_html = f"""
                        <div style="display: flex; gap: 80px;">
                            <div style="width: 400px; height: 400px;">
                                <img src="data:image/png;base64,{plt_to_base64(img)}" alt="image" style="width: 400px; height: 400px;">
                            </div>
                            <div style="width: 400px; height: 400px;">
                                <img src="data:image/png;base64,{plt_to_base64(visualization)}" alt="heatmap" style="width: 400px; height: 400px;">
                                <p style="text-align: center; margin-top: 5px;">Characteristic heatmap</p>
                            </div>
                        </div>
                    """
                    if 'predictions' in st.session_state and len(st.session_state['predictions']) > index:
                        prediction = st.session_state['predictions'][index]
                        confidence = st.session_state['confidences'][index]
                        info_html = f"""
                            <div style="clear: both;">
                                <p style="font-size: 20px; margin-left: 30px;">
                                    P{index + 1} Predict: <span style="font-weight: bold;">{defect_labels[prediction]}</span> 
                                    Confidence: <span style="font-weight: bold;">{confidence:.2f}</span>
                                </p>
                            </div>
                        """
                    else:
                        info_html = f"""
                            <div style="clear: both;">
                                <p style="font-size: 20px; margin-left: 30px;">Image Name: P{index + 1}</p>
                            </div>
                        """
                    image_placeholder.markdown(img_html, unsafe_allow_html=True)
                    info_placeholder.markdown(info_html, unsafe_allow_html=True)





                if st.button("Start detection"):
                    st.session_state['detecting'] = True
                    with st.spinner("Detecting in progress..."):
                        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                        model_dict = {
                            'LeNet': LeNet,
                            'AlexNet': AlexNet,
                            'VGG16': VGG16,
                            'GoogLeNet': GoogLeNet,
                            'ResNet34': ResNet34,
                            'MobileNet': MobileNet,
                            'ShuffleNet': ShuffleNet,
                            'DenseNet': DenseNet,
                            'EfficientNet': EfficientNet,
                            'RegNet': RegNet,
                            'SwinViT': SwinViT,
                            # 'vmkan': vmkan,
                        }

                        model = load_model(model_dict[model_name](), weight_path, device)

                        predictions = []
                        confidences = []

                        for i in range(len(test_data)):
                            img, _ = test_data[i]
                            img = img.to(device).unsqueeze(0)
                            outputs = model(img)
                            _, predicted = torch.max(outputs, 1)
                            confidence = torch.softmax(outputs, dim=1)[0][predicted].item()

                            predictions.append(predicted.item())
                            confidences.append(confidence)

                        st.session_state['predictions'] = predictions
                        st.session_state['confidences'] = confidences



                        detection_results = pd.DataFrame({
                            'Filename': [f"P{i + 1}" for i in range(len(test_data))],
                            'Predict': [defect_labels[pred] for pred in predictions],
                            'Confidence': confidences
                        })

                        st.session_state['detection_results'] = detection_results

                        show_test_image(test_index)

                if 'detection_results' in st.session_state and st.session_state['detection_results'] is not None:
                    table_placeholder.dataframe(st.session_state['detection_results'], width=880, height=600)
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button("The previous one"):
                        if test_index - 1 >= 0:
                            st.session_state['test_index'] -= 1
                            show_test_image(st.session_state['test_index'])
                with col2:
                    if st.button("Display current image"):
                        show_test_image(test_index)

                with col3:
                    if st.button("The next one"):
                        if test_index + 1 < len(test_data):
                            st.session_state['test_index'] += 1
                            show_test_image(st.session_state['test_index'])



if __name__ == "__main__":
    main()