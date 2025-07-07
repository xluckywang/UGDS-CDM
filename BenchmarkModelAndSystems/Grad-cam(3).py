import os
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from utils import GradCAM, show_cam_on_image, center_crop_img
from model import swin_tiny_patch4_window7_224 as create_model
import os
import math
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

def process_images(model, device, data_transform, target_layers, img_size, val_dir, output_dir):
    for root, dirs, files in os.walk(val_dir):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg', '.JPG', 'PNG')):  # 检查文件是否为图像
                img_path = os.path.join(root, file)
                img = Image.open(img_path).convert('RGB')

                # 打印图像原始尺寸，方便调试
                print(f"Original size of {file}: {img.size}")

                # 应用数据转换
                img_tensor = data_transform(img)

                # 打印图像尺寸，方便调试
                print(f"Processing {file}: Image size = {img_tensor.shape}")

                # 扩展批次维度并移动到相应设备
                input_tensor = torch.unsqueeze(img_tensor, dim=0).to(device)

                try:
                    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=torch.cuda.is_available(),
                                  reshape_transform=ResizeTransform(im_h=img_size, im_w=img_size))
                    with torch.no_grad():
                        output = model(input_tensor)
                    target_category = output.argmax(dim=1).item()  # 使用得分最高的类别

                    grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)
                    grayscale_cam = grayscale_cam[0, :]

                    # 确保用于叠加的图像与heatmap的尺寸匹配
                    img_resized = img.resize((224, 224))
                    visualization = show_cam_on_image(np.array(img_resized).astype(dtype=np.float32) / 255.,
                                                      grayscale_cam,
                                                      use_rgb=True)

                    # 创建类别输出文件夹
                    category = os.path.basename(root)
                    category_output_dir = os.path.join(output_dir, category)
                    if not os.path.exists(category_output_dir):
                        os.makedirs(category_output_dir)

                    # 保存可视化结果
                    output_path = os.path.join(category_output_dir, f"heatmap_{os.path.splitext(file)[0]}.jpg")
                    plt.imsave(output_path, visualization)
                    print(f"Saved heatmap for {file} at {output_path}")

                except RuntimeError as e:
                    print(f"Error processing {file}: {e}")
                    continue

def main():
    # 注意输入的图片必须是32的整数倍
    # 否则由于padding的原因会出现注意力飘逸的问题
    img_size = 224
    assert img_size % 32 == 0

    model = create_model(num_classes=8)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载模型参数并将模型转移到相应设备
    model.load_state_dict(torch.load('bestzt_270_swinvitnet_3_3_bqd_cddpm1.pth', map_location=device))
    model.to(device)
    model.eval()  # 设置模型为评估模式

    target_layers = [model.norm]

    data_transform = transforms.Compose([
        transforms.Grayscale(),  # 灰度处理
        transforms.Resize((224, 224)),  # 修改为适当的图像大小
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, ], std=[0.229, ])
    ])

    val_dir = './../../../Data/Data/val_270'
    output_dir = './Grad-cam/bestzt_270_swinvitnet_3_3_bqd_cddpm1.pth'

    process_images(model, device, data_transform, target_layers, img_size, val_dir, output_dir)

if __name__ == '__main__':
    main()