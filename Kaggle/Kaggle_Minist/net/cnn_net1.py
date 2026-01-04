import torch
import torch.nn.functional as F
class CNN_Net1(torch.nn.Module):
    def __init__(self):
        super(CNN_Net1, self).__init__()
        # 【关键修改 2】：输入通道改成 1 (in_channels=1)
        #  batch_size=64 是由 DataLoader 控制的，这里只管单张图片的结构
        # 结构：1 -> 32 -> 64 (通常不需要一开始就 128，32 对 MNIST 足够了且更快)
        self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(32) # <--- 新增 BN 32是通道数 按照通道数来归一化
        
    # 为什么要加 BN (Batch Normalization)？
        #       痛点：没有 BN 之前（内卷的数据）
        # 想象一个公司接力传话游戏：
        # 第 1 个人（第一层）说话声音很小（数据值很小）。
        # 第 2 个人（第二层）为了听清，必须把耳朵贴很近（权重变大）。
        # 第 3 个人突然大吼一声（数据值突然变大）。
        # 第 4 个人（第四层）被吓傻了，不知道该怎么调整。
        # 在神经网络里，这叫 Internal Covariate Shift。
        # 因为每一层的参数都在变，导致输出的数据分布忽大忽小
        # 下一层网络为了适应这种变化，就得小心翼翼地调整，导致训练非常慢，甚至训练不动。

        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(64) # <--- 新增 BN 
        self.conv3 = torch.nn.Conv2d(64, 128, kernel_size=2, stride=1, padding=1)
        self.bn3 = torch.nn.BatchNorm2d(128) # <--- 新增 BN
        self.pool = torch.nn.MaxPool2d(2, 2)
        
        # 计算全连接层输入：
        # Input: (1, 28, 28)
        # Conv1 -> (32, 28, 28) -> Pool -> (32, 14, 14)
        #\ Conv2 -> (64, 14, 14) -> Pool -> (64, 7, 7)
        #\ Conv3 -> (128, 8, 8) -> Pool -> (128, 4, 4)
        # Flatten -> 128 * 4 * 4 = 2048
        self.fc1 = torch.nn.Linear(128 * 4 * 4, 128) # 加一个中间层，过渡一下
        self.dropout = torch.nn.Dropout(0.5)         # <--- 新增 Dropout (丢弃50%)
        self.fc2 = torch.nn.Linear(128, 10)          # 输出层

    def forward(self, x):
        # Conv -> BN -> ReLU -> Pool
        x = self.pool(F.relu(self.bn1(self.conv1(x))))# 加入 BN
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))

        # 【关键修改 3】：使用 x.size(0) 动态获取 batch 大小
        # 之前的 view(batch_size, -1) 会在测试集最后一个 batch (不足64个时) 报错
        x = x.view(x.size(0), -1) 
        # 假设形状是 [64, 32, 14, 14]
        # x.shape = [ 64,   32,   14,   14 ]
        #             ↑      ↑      ↑      ↑
        # 索引(Index): 0      1      2      3
        #            (batch) (通道) (高)   (宽)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)       # 训练时随机丢弃，测试时自动全连
        x = self.fc2(x)
        #为什么最后一层不激活？以及为什么测试时自动全连？
        #交叉熵中已经包含了Softmax函数
        #softmax的作用是把输出转换为概率分布 用指数归一化
        return x
