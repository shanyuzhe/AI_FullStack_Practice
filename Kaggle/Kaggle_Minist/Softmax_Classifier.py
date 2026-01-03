import torch
import numpy as np
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
import torch.optim as optim 
from utils.Minist_loader import MnistDataloader
from os.path  import join

input_path = r'C:\Users\22684\.cache\kagglehub\datasets\hojjatk\mnist-dataset\versions\1'
training_images_filepath = join(input_path, 'train-images-idx3-ubyte/train-images-idx3-ubyte')
training_labels_filepath = join(input_path, 'train-labels-idx1-ubyte/train-labels-idx1-ubyte')
test_images_filepath = join(input_path, 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
test_labels_filepath = join(input_path, 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')

loader =  MnistDataloader(test_images_filepath=test_images_filepath, test_labels_filepath=test_labels_filepath,
                          training_images_filepath=training_images_filepath,training_labels_filepath=training_labels_filepath)

(x_train_raw, y_train_raw),(x_test_raw, y_test_raw) =  loader.load_data()
 
# --- 第二步：模拟 torchvision 的变换 (Transform) ---
# 1. 转为 Tensor 并归一化到 [0, 1] (相当于 ToTensor)
x_train_tensor = torch.tensor(np.array(x_train_raw)).float() / 255.0
x_test_tensor  = torch.tensor(np.array(x_test_raw)).float() / 255.0

# 2. 标准化 (相当于 Normalize)
x_train_tensor = (x_train_tensor - 0.1307) / 0.3081
x_test_tensor  = (x_test_tensor - 0.1307) / 0.3081

# 3. 标签转为 LongTensor
y_train_tensor = torch.tensor(np.array(y_train_raw)).long()
y_test_tensor  = torch.tensor(np.array(y_test_raw)).long()#分类标签默认使用long

# --- 第三步：封装为 DataLoader ---
train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
test_dataset  = TensorDataset(x_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=64, shuffle=False)


# 相当于手动完成了下述操作
# transform = transforms.Compose([
# transforms.ToTensor(),
# transforms.Normalize((0.1307, ), (0.3081, ))#均值和标准差
# ])#Normalize正态分布标准化 加上逗号才能识别出为元组


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.l1 = torch.nn.Linear(784, 512)
        self.l2 = torch.nn.Linear(512, 256)
        self.l3 = torch.nn.Linear(256, 128)
        self.l4 = torch.nn.Linear(128, 64)
        self.l5 = torch.nn.Linear(64, 10)
    def forward(self, x):
        x = x.view(-1, 784)#-1表示自动计算 拉直长一行的向量
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        return self.l5(x)

model = Net()

#softmax 利用指数函数进行映射到0~1 --> log --> -ylog(pre_y)     
#CrossEntropyLoss封装了上述流程
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr = 0.01)

def train(epoch):
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader, 0):
        inputs, target = data
        optimizer.zero_grad()
        # forward + backward + update
        outputs = model(inputs)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if batch_idx % 300 == 299:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_idx + 1, running_loss / 300))
            running_loss = 0.0
def test():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, dim=1)
            #dim = 1按照列的方向找最大值 第一个返回本身 第二个返回索引
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy on test set: %d %%' % (100 * correct / total))
    
if __name__ == '__main__':
    for epoch in range(10):
        train(epoch)
        test()