import numpy as np 
import matplotlib.pyplot as plt 
import torch
import torch.nn.functional as F

#1：Prepare dataset------------------------
x_data = torch.Tensor([[1.0], [2.0], [3.0]])
y_data = torch.Tensor([[0], [0],[1]])
#------------------------------------------


#2：Design model---------------------------------
class LogisticRegressionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)
    
    def forward(self, x):
        y_pred = F.sigmoid(self.linear(x))#添加sigmoid变换
        return y_pred
model = LogisticRegressionModel()
#-------------------------------------------------------

#3construct loss and optimizer-----------------------------------------------
#实例化 criterion 和 optimizer（it is callable)
#用BCE表示损失 
criterion = torch.nn.BCELoss(size_average = False)
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)
#print(y_pred.dtype, y_pred.shape) 调试技巧
#print(y_data.dtype, y_data.shape)
#----------------------------------------------------------


#4Training cycle------------------------
for epoch in range(1000):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    print(epoch, loss.item())
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()#更新权重

 #---------------------------------------
     

#5可视化
x = np.linspace(0, 10, 200)#生成1~10 200个点
#用linspace模拟测试集 
x_t = torch.Tensor(x).view((200, 1))
#转成200行1列 200个样本 每个样本一个特征
y_t = model(x_t)
#这里的model已经训练好了，所以y_t就是最终的y_t
y = y_t.data.numpy()#plot需要传numpy数组
plt.plot(x, y)#根据numpy数组画曲线
plt.plot([0, 10], [0.5,0.5], c = 'r')
#在P = 0.5处画直线,横坐标从0~10 纵坐标0.5 0.5
plt.xlabel('Hours')
plt.ylabel('Probability of pass')
plt.grid()#添加网格线
plt.show()

