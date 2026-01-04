import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
import torch.optim as optim 
from utils.Minist_loader import MnistDataloader
from os.path import join

# --- 1. 数据加载 ---
input_path = r'C:\Users\22684\.cache\kagglehub\datasets\hojjatk\mnist-dataset\versions\1'
training_images_filepath = join(input_path, 'train-images-idx3-ubyte/train-images-idx3-ubyte')
training_labels_filepath = join(input_path, 'train-labels-idx1-ubyte/train-labels-idx1-ubyte')
test_images_filepath = join(input_path, 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
test_labels_filepath = join(input_path, 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')

loader = MnistDataloader(test_images_filepath=test_images_filepath, test_labels_filepath=test_labels_filepath,
                          training_images_filepath=training_images_filepath, training_labels_filepath=training_labels_filepath)

(x_train_raw, y_train_raw), (x_test_raw, y_test_raw) = loader.load_data()

# --- 2. 数据预处理 ---
# 转为 Tensor 并归一化
x_train_tensor = torch.tensor(np.array(x_train_raw)).float() / 255.0
x_test_tensor  = torch.tensor(np.array(x_test_raw)).float() / 255.0

# 标准化
x_train_tensor = (x_train_tensor - 0.1307) / 0.3081
x_test_tensor  = (x_test_tensor - 0.1307) / 0.3081

# 【关键修改 1】：增加通道维度
# 目前形状是 [60000, 28, 28]，需要变成 [60000, 1, 28, 28] 才能喂给 CNN
x_train_tensor = x_train_tensor.unsqueeze(1) 
x_test_tensor = x_test_tensor.unsqueeze(1)
# unsqueeze 在索引 1 位置插入一个维度
# 形状变化示意:
# 数值:  [ 60000,     1,      28,     28 ]
# 含义:  (Batch)  (Channel)  (高)    (宽)
# 索引:     0        1         2       3
#                   ^
#                   |
#              新插入在这里


# 标签处理
y_train_tensor = torch.tensor(np.array(y_train_raw)).long()
y_test_tensor  = torch.tensor(np.array(y_test_raw)).long()

# --- 3. DataLoader ---
batch_size = 64
train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
test_dataset  = TensorDataset(x_test_tensor, y_test_tensor)#打包配对

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# --- 4. 网络定义 ---
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 【关键修改 2】：输入通道改成 1 (in_channels=1)
        #  batch_size=64 是由 DataLoader 控制的，这里只管单张图片的结构
        # 结构：1 -> 32 -> 64 (通常不需要一开始就 128，32 对 MNIST 足够了且更快)
        self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(32) # <--- 新增 BN
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
        return x

model = Net()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
model.to(device)

# --- 5. 训练配置 ---
criterion = torch.nn.CrossEntropyLoss()
# 把学习率改小一点，0.01 对 Adam 来说有点大，容易震荡，0.001 比较稳
optimizer = optim.Adam(model.parameters(), lr=0.001)
# 定义学习率衰减策略：每隔 3 个 Epoch，学习率 乘以 0.1
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

def train(epoch):
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader, 0):
        inputs, target = data
        inputs, target = inputs.to(device), target.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        if batch_idx % 300 == 299:
            print('[Epoch %d, Batch %5d] loss: %.3f' % (epoch + 1, batch_idx + 1, running_loss / 300))
            running_loss = 0.0

def test() -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, dim=1)# 获取每行最大值的索引 _代表值 predicted代表索引 最大值即为概率最大的地方
            total += labels.size(0) #每次加一个batch_size
            correct += (predicted == labels).sum().item()
    print('Accuracy on test set: %.2f %%' % (100 * correct / total))
    return 1 - correct / total

if __name__ == '__main__':
    # 打印一下输入的形状确认无误
    print("输入 Tensor 形状:", x_train_tensor.shape) # 应该是 [60000, 1, 28, 28]
    
    epoch_list = []
    miss_list = [] 
    for epoch in range(8): # 5 个 epoch 足够了
        train(epoch)
        miss_rate = test()
        scheduler.step() # <--- 记得每跑完一轮 update 一下学习率
        print("Current learning rate:", scheduler.get_last_lr())
        miss_list.append(miss_rate * 100)  # 转为百分比
        epoch_list.append(epoch)
        
# --- 6.画图
    import matplotlib.pyplot as plt

    plt.plot(epoch_list, miss_list, marker='o')
    plt.title('Miss Rate over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Miss Rate (%)')
    plt.grid()
    plt.show()