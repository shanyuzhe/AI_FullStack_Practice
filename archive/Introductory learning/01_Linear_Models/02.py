import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F

# ===================== 1. 构建人工数据集 =====================
# 创建两个类别的数据集，使用正态分布生成两组数据

# 类别 0 的数据，均值为 (2, 2)
x0 = torch.randn(50, 2) + torch.tensor([2, 2])
# x0.shape = (50, 2)，正态分布加上均值偏移 均值为2

# 类别 1 的数据，均值为 (7, 7)
x1 = torch.randn(50, 2) + torch.tensor([7, 7])#均值为7

# 标签为 0、1
y0 = torch.zeros(50, dtype=torch.long)
y1 = torch.ones(50, dtype=torch.long)  
# y0.shape = (50,)

# 合并数据
X = torch.cat([x0, x1], dim=0)
# X.shape = (100, 2)

y = torch.cat([y0, y1], dim=0)
# y.shape = (100,)

# 可视化数据
plt.scatter(
    X[:, 0].numpy(), #x轴数据
    X[:, 1].numpy(), #y轴数据
    c=y.numpy(), #颜色根据类别标签变化
    cmap='coolwarm',  # 设置色卡，比如红蓝配色
    edgecolors='k',   # 给每个圆点加个黑边，看起来更立体
    alpha=0.8         # 设置透明度，防止点太多重叠看不清
)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Synthetic Data for Softmax Classification')
plt.show()
# ===================== 2. 定义Softmax分类模型 =====================
class SoftmaxClassifier(nn.Module):
    def __init__(self):
        super(SoftmaxClassifier, self).__init__()
        
        # 两个特征输入，两个类别输出
        self.linear = nn.Linear(2, 2)  # 输入2维，输出2维

    def forward(self, x):
        return self.linear(x)

# 实例化模型
model = SoftmaxClassifier()

# ===================== 3. 定义损失函数和优化器 =====================
# 交叉熵损失，内部包含了 softmax
criterion = nn.CrossEntropyLoss()

# 随机梯度下降 (SGD) 作为优化器
optimizer = optim.SGD(model.parameters(), lr=0.01)  # 学习率 0.01

# ===================== 4. 训练模型 =====================
epochs = 1000  # 训练回合数
for epoch in range(epochs):
    # 使用模型预测输出
    predictions = model(X)
    # predictions.shape = (100, 2)
    
    # 计算损失
    loss = criterion(predictions, y)
    
    # 梯度清零，防止累积
    optimizer.zero_grad()
    
    # 计算梯度
    loss.backward()
    
    # 更新参数
    optimizer.step()

    # 打印训练过程中的损失值
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

# ===================== 5. 可视化结果 =====================
# 使用训练好的模型在同一数据上进行预测
with torch.no_grad():
    predicted_labels = torch.argmax(model(X), dim=1)  # 取出每个样本的最高分对应的类别
    # predicted_labels.shape = (100,)

# 可视化真值与预测结果对比
# alpha参数控制透明度，edgecolors参数设置边缘颜色
plt.scatter(X[:, 0].numpy(), X[:, 1].numpy(), c=y.numpy(), marker='o', label='True Labels',
            cmap='coolwarm', edgecolors='k', alpha=0.5)#用圆圈表示真值
plt.scatter(X[:, 0].numpy(), X[:, 1].numpy(), c=predicted_labels.numpy(), marker='x', label='Predicted Labels',
            cmap='coolwarm', edgecolors='k', alpha=1)#用叉号表示预测值
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('True Labels vs Predicted Labels')
plt.legend(['True Labels (circles)', 'Predicted Labels (crosses)'])
plt.show()
