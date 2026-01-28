import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.channels = channels
        # 改进1: 加入 Batch Normalization
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels) 
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return F.relu(x + y)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5)
        self.bn1 = nn.BatchNorm2d(16) # BN 配套
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5)
        self.bn2 = nn.BatchNorm2d(32) # BN 配套
        self.mp = nn.MaxPool2d(2)
        
        self.rblock1 = ResidualBlock(16)
        self.rblock2 = ResidualBlock(32)
        
        # 改进2: 加入 Dropout
        self.dropout = nn.Dropout(0.5) 
        self.fc = nn.Linear(512, 10)

    def forward(self, x):
        in_size = x.size(0)
        x = self.mp(F.relu(self.bn1(self.conv1(x)))) # 加入 BN
        x = self.rblock1(x)
        x = self.mp(F.relu(self.bn2(self.conv2(x)))) # 加入 BN
        x = self.rblock2(x)
        x = x.view(in_size, -1)
        x = self.dropout(x) # 加入 Dropout
        x = self.fc(x)
        return x