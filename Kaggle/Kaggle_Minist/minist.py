import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from net.cnn_net1 import CNN_Net1 as Net

# --- 配置 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
EPOCHS = 8

def main():
    # 1. 数据准备 (使用 torchvision)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # 自动下载到当前目录下的 data 文件夹
    train_ds = datasets.MNIST('data', train=True, download=True, transform=transform)
    test_ds = datasets.MNIST('data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    # 2. 初始化模型
    model = Net().to(DEVICE)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # 3. 训练与测试循环 (保持你原来的逻辑，封装一下更整洁)
    epoch_list = []
    acc_list = []
    
    for epoch in range(EPOCHS):
        train(model, train_loader, optimizer, criterion, epoch)
        acc = test(model, test_loader)
        scheduler.step()
        
        epoch_list.append(epoch)
        acc_list.append(acc)

    # 4. 保存与绘图
    torch.save(model.state_dict(), 'best_mnist.pth')
    plot_result(epoch_list, acc_list)

def train(model, loader, optimizer, criterion, epoch):
    model.train()
    running_loss = 0.0
    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        if batch_idx % 300 == 299:
             print(f'[Epoch {epoch+1}, Batch {batch_idx+1}] loss: {running_loss/300:.3f}')
             running_loss = 0.0

def test(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, dim=1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    
    acc = 100 * correct / total
    print(f'Accuracy on test set: {acc:.2f} %')
    return acc

def plot_result(epochs, accs):
    plt.plot(epochs, accs, marker='o')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.grid()
    plt.show()

if __name__ == '__main__':
    main()