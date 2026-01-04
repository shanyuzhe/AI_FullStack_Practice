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

# 标签处理
y_train_tensor = torch.tensor(np.array(y_train_raw)).long()
y_test_tensor  = torch.tensor(np.array(y_test_raw)).long()

# --- 3. DataLoader ---
batch_size = 64
train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
test_dataset  = TensorDataset(x_test_tensor, y_test_tensor)

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
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = torch.nn.MaxPool2d(2, 2)
        
        # 计算全连接层输入：
        # Input: (1, 28, 28)
        # Conv1 -> (32, 28, 28) -> Pool -> (32, 14, 14)
        # Conv2 -> (64, 14, 14) -> Pool -> (64, 7, 7)
        # Flatten -> 64 * 7 * 7 = 3136
        self.fc = torch.nn.Linear(64 * 7 * 7, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        
        # 【关键修改 3】：使用 x.size(0) 动态获取 batch 大小
        # 之前的 view(batch_size, -1) 会在测试集最后一个 batch (不足64个时) 报错
        x = x.view(x.size(0), -1) 
        x = self.fc(x)
        return x

model = Net()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
model.to(device)

# --- 5. 训练配置 ---
criterion = torch.nn.CrossEntropyLoss()
# 把学习率改小一点，0.01 对 Adam 来说有点大，容易震荡，0.001 比较稳
optimizer = optim.Adam(model.parameters(), lr=0.001)

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
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy on test set: %.2f %%' % (100 * correct / total))
    return 1 - correct / total

if __name__ == '__main__':
    # 打印一下输入的形状确认无误
    print("输入 Tensor 形状:", x_train_tensor.shape) # 应该是 [60000, 1, 28, 28]
    
    epoch_list = []
    miss_list = [] 
    for epoch in range(4): # 5 个 epoch 足够了
        train(epoch)
        miss_rate = test()
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