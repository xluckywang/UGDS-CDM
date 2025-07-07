import torch
from torchvision import datasets, transforms
def data_preprocessing():
    data_transform = transforms.Compose([
    transforms.Grayscale(),#灰度处理
    transforms.Resize((224, 224)),  # 修改为适当的图像大小
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))#归一化
])
    return data_transform
# 将图像预处理封装成函数
def preprocess_data(data_root):
    data_transform = data_preprocessing()
    dataset = datasets.ImageFolder(root=data_root, transform=data_transform)
    return dataset

