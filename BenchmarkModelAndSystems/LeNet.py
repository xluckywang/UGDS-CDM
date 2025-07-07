import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
    def __init__(self,num_classes=8):
        super(LeNet, self).__init__()
        # 定义特征层
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 5),  # input(1, 32, 32) output(16, 28, 28)
            nn.ReLU(),
            nn.MaxPool2d(2, 2),   # output(16, 14, 14)
            nn.Conv2d(16, 32, 5), # output(32, 10, 10)
            nn.ReLU(),
            nn.MaxPool2d(2, 2)    # output(32, 5, 5)
        )
        # 定义分类层
        self.classifier = nn.Sequential(
            nn.Linear(89888, 120), # input(32*5*5) output(120)
            nn.ReLU(),
            nn.Linear(120, 84),         # output(84)
            nn.ReLU(),
            nn.Linear(84, 8)            # output(7)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # 展平特征图
        x = self.classifier(x)
        return x

