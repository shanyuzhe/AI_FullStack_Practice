import numpy as np
import matplotlib.pyplot as plt 
import torch
import torch.nn.functional as F

#1 prepare dataset

xy = np.loadtxt('diabetes.csv', delimiter=',',dtype=np.float32, skiprows=1)
#delimiter 标记分隔符
x_raw = xy[:, :-1]
#切片第一个':'表示所有行 第二个‘：-1’表示除了最后一列以外的所有列
y_raw = xy[:, [-1]]
#xy[:,-1]最后一列的所有数据，但是展开成一行，没有保持原来的维度
#xy[:,[-1]]保留了原有数据的（768,1）的形状


#归一化
#如果不归一化，由于各个特征的波动程度不一样
#z = w1 * x1 + w2 * x2 + b
#如果x1波动的特别大,w1变化一点点就会导致步子迈的太大
#从而导致越过极值点

#x_raw.min(axis = 0),axis = 0表示列方向找最小值
#利用 (x - min(x)) / (max(x) - min(x))做归一化运算
#把每一列都转化成相对距离
#这里会触发一个广播机制，当axis = 0时
#形如：
#【20,180】
#【20,100】---> [20, 100]按照列方向找最小值
#【60,140】
#然后[20,100]和原矩阵运算触发广播机制，变成768行的[20,100] 再进行差运算
#最后相当于每一列都变成了该列的(x - min(x)) / (max(x) - min(x))
#每一列的元素都变成了相对距离
x_data = torch.from_numpy((x_raw - x_raw.min(axis=0)) 
/ (x_raw.max(axis=0) - x_raw.min(axis=0)))

y_data = torch.from_numpy(y_raw)


#2 Define Model
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(8,16)
        self.linear2 = torch.nn.Linear(16,10)
        self.linear3 = torch.nn.Linear(10,4)
        self.linear4 = torch.nn.Linear(4, 1)
        self.sigmoid = torch.nn.Sigmoid()
    def forward(self,x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        x = self.sigmoid(self.linear4(x))
        return x
model = Model()

#3 Construct Loss and Optimizer
criterion = torch.nn.BCELoss(size_average=False)




#optimizer = torch.optim.SGD(model.parameters(),lr = 0.01)
optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)
#Adam相对于SGD而言可以自适应 
#举例而说， SGD的学习率固定，
#在谷底的时候可能会随着梯度改变反复震荡（前进一下，后退一下）
#而Adam改变方向的时候则会调低学习率，最后逼近谷底




#4 Training Cycle
epoch_list = []
loss_list = []
for epoch in range(3000):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    #print(epoch, loss.item())
    
    #loss 是BCE 是0维张量



    epoch_list.append(epoch)
    loss_list.append(loss.item())
#item() 做了什么“断舍离”？
#当你调用 loss.item() 时：

#数值拷贝：它从 Tensor 里取出那个 float 数值，
# 并创建了一个新的 Python float 对象。

#切断联系：这个新的 float 对象是一个独立的数字，
# 它不再包含 grad_fn 属性，也不再引用任何计算图。

#释放空间：

#当你把 float 数值存入列表后，
# 原来的 loss 张量在当前循环结束时，
# 它的引用计数（Reference Count）会减 1
# （如果你没有在别处引用它）。

#此时，PyTorch 的内存管理器发现这个 Tensor 没用了，
# 就会把这轮训练产生的巨大计算图从内存/显存中彻底抹除。

#相当于硬链接，不写item()就会增加引用计数，
# 导致backward不删除计算图

#python的变量都是引用赋值


    #Backward
    optimizer.zero_grad()
    loss.backward()
    
    #Update
    optimizer.step()
    


#5 可视化
plt.plot(epoch_list, loss_list)
plt.xlabel('epoch')
plt.ylabel('loss(BCE)')
plt.grid()
plt.show()

#阶段,现象,核心问题,你的解决方案
#最初版,Loss 26800.0 (死锁),学习率过大 + 数据未归一化,降低 lr + 归一化
#进阶版,Loss 496 (震荡),SGD 步长固定，在谷底反复横跳,尝试 Adam (自适应步长)
#优化版,Loss 309 (平稳),Sigmoid 导致模型在后期“疲软”,建议换 ReLU (增强梯度传导)