import numpy as np 
import matplotlib.pyplot as plt 

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

def forward(x):#前馈函数
    return x * w

def loss(x, y):
    y_hat = forward(x)
    return (y_hat - y) * (y_hat - y)#计算误差损失

w_list = []
mse_list = []#均方误差

for w in np.arange(0.0, 4.1, 0.1):
    print('w=', w)
    l_sum = 0
    for x_val,y_val in zip(x_data, y_data):
        y_pred_val = forward(x_val)
        loss_val = loss(x_val,y_val)
        l_sum += loss_val
        print('\t', x_val, y_val, y_pred_val, loss_val)
    print('MSE = ', l_sum / 3)
    w_list.append(w)
    mse_list.append(l_sum / 3)

plt.plot(w_list, mse_list) #第一个参数是x轴 第二个y轴
plt.ylabel('Loss') #添加y轴标签
plt.xlabel('w')
plt.show()