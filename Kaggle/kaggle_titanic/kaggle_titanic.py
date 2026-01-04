import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt 
from torch.utils.data import Dataset, DataLoader
#数据集特别大的时候不可能所有一次性读入内存
#所以需要DataLoader 每次取一个Batch
import torch.nn.functional as F

def predict_test_data(model, test_filepath):
    # 1. 加载测试集
    df_test = pd.read_csv(test_filepath)
    passenger_ids = df_test['PassengerId'] # 记录 ID，最后写 CSV 要用
    
    # 2. 同样的数据清洗 (必须和训练集一模一样)
    # fillna空位填充 mean（）均值 median()中位数
    df_test['Sex'] = df_test['Sex'].map({'male':0, 'female' : 1})
    df_test['Age'] = df_test['Age'].fillna(df_test['Age'].mean())
    df_test['Fare'] = df_test['Fare'].fillna(df_test['Fare'].median())
    df_test['Embarked'] = df_test['Embarked'].map({'S':0, 'C':1, 'Q':2}).fillna(0)
    
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    x_raw = df_test[features].values.astype(np.float32)
    
    # 3. 同样的归一化 (使用测试集自己的 Min-Max)
    x_norm = (x_raw - x_raw.min(axis=0)) / (x_raw.max(axis=0) - x_raw.min(axis=0))
    x_tensor = torch.from_numpy(x_norm)
    
    # 4. 开启预测模式
    model.eval() # model切换到评估模式
    with torch.no_grad(): # 预测时不需要计算梯度，节省内存
        outputs = model(x_tensor)
        # 将概率转为 0 或 1 (阈值设为 0.5)
        #outputs经过了sigmoid算概率，如果概率大于0.5，就为TRUE
        #再把pytorch.tensor转成numpy数组，并展平 （tensor里面有梯度信息啥的）
        #flatten是转成一维序列，要不然可能会导致每一行数据带上多余的方括号
        predictions = (outputs > 0.5).int().numpy().flatten()
    
    # 5. 生成结果文件
    #pd.DataFrame：创建一个新的 Pandas 表格。
    submission = pd.DataFrame({
        "PassengerId": passenger_ids,
        "Survived": predictions
    })
    #不加行索引 kaggle要求的
    submission.to_csv('./dataset/submission.csv', index=False)
    
    print("预测完成，结果已保存至 ./dataset/submission.csv")

class DiabetesDataset(Dataset):
    def __init__(self, filepath):
        # 1. 加载数据，注意 skiprows
        #pandas做数据清洗
        df = pd.read_csv(filepath)
        #性别映射
        df['Sex']  = df['Sex'].map({'male':0, 'female' : 1})
        #填补年龄缺失
        df['Age'] = df['Age'].fillna(df['Age'].mean())
        df['Fare'] = df['Fare'].fillna(df['Fare'].median())
        features = ['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']
        
        df['Embarked'] = df['Embarked'].map({'S':0, 'C':1, 'Q':2}).fillna(0)
        
        x_raw = df[features].values.astype(np.float32)
        y_raw = df[['Survived']].values.astype(np.float32)
        
        
        
        self.len = df.shape[0]
        
        #self.len = xy.shape[0](np错误版本)
        #xy.shape是矩阵的形状
        #xy.shape[0]是行数
        #xy.shape[1]是列数
        #这里只需要知道行数，也就是多少条数据
        #DataLoader根据len和Batch_size来规划迭代次数
        
        #为什么要规划迭代次数？
        #一个Epoch定义模型看完了所有样本一次
        #而步数（Iterations） = 总样本数 / Batch_size
        #并且Dataloader只需要维护一个索引表
        #每次把一个Batch的数据读入内存即可



        # 2. 提取原始数据（错误版本）
        #cols = [i for i in range(xy.shape[1]) if i != 1]
        #x_raw = xy[:, cols]
        #y_raw = xy[:, [1]]
        
        # 3. 归一化 (Min-Max)
        x_norm = (x_raw - x_raw.min(axis=0)) / (x_raw.max(axis=0) - x_raw.min(axis=0))
        
        self.x_data = torch.from_numpy(x_norm)
        self.y_data = torch.from_numpy(y_raw)

    #__getitem__和__len__是魔术方法
    #必须实现，DataLoader需要他们

    #魔术方法（__function_name__)：
    #魔术方法本质上是 Python 预留的“后门”。
    #当你给一个类写了某个魔术方法，就相当于告诉 Python：
    #“当遇到某种特定操作（如加法、长度查询、索引取值）时，
    #请按照我写的这个逻辑来执行。”


    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    #返回某一个样本的特定指标，x_data[index]就是第index样本x的那8个指标
    def __len__(self):
        return self.len

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(7, 6)
        self.linear2 = torch.nn.Linear(6, 4)
        self.linear3 = torch.nn.Linear(4, 1)

    def forward(self, x):
        # 中间层换 ReLU 效果更好
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = torch.sigmoid(self.linear3(x))
        return x

# Windows 下多线程必须放在 if __name__ == '__main__': 下
if __name__ == '__main__':
    dataset = DiabetesDataset('./dataset/train.csv')
    train_loader = DataLoader(dataset=dataset, 
                              batch_size=32,
                              shuffle=True,
                              num_workers=0)
    #小数据使用多进程特别慢，搬用开销太大，spawn机制太重
    
    
    #shuffle 是否打乱顺序
    #num_worker线程数量
    """ if__name__ == '__main__'底层原理：
Linux/Mac：使用的是 fork 机制。创建子进程时，
会直接把主进程的整个“内存快照”复制一份。
子进程知道自己要做什么，不会重新执行一遍脚本。
Windows：使用的是 spawn 机制。
它创建子进程的方式是：启动一个新的 Python 解释器，
然后把你的 .py 文件从头到尾重新跑一遍！

灾难现场（如果不加保护）：
主进程运行到 DataLoader，发现要开 2 个子进程。

子进程 A 启动，重新跑你的脚本。
跑到 DataLoader 这一行，它又想开 2 个子进程……

子进程 B 启动，同上……

这就形成了递归炸弹，你的内存会瞬间被无数个
重复启动的 Python 进程撑爆，电脑直接卡死。 """

    """ if __name__ == '__main__': 的作用：
它像一道防火墙。
只有当你直接双击运行这个文件时，__name__ 才等于 '__main__'。
子进程是被“召唤”出来的，它们的 __name__ 不是 '__main__'。
结果：子进程执行脚本时，会跳过 if 块里的内容
（也就是跳过 DataLoader 的创建），
直接去执行分配给它们的搬运任务
 """

    """ 为什么循环也要放在if__name__ == __main__里面
实际情况是这样的：子进程确实不运行 for 循环，
但这正是我们想要的。
我们可以把整个过程想象成一个**“后勤保障系统”**：
1. 角色分工：大脑 vs 搬运工
主进程（大脑/指挥官）： 它进入了 if 块，执行 for 循环。
它的任务是：计算梯度、更新权重、控制节奏。
它不需要自己去搬数据，它只需要喊一声：“下一批数据！”
子进程（搬运工）： 
它们被 DataLoader “召唤”出来。
虽然它们跳过了 if 块里的 for 循环，但它们并没有闲着。
DataLoader 内部会给它们分配专门的任务：
去执行 Dataset 里的 __getitem__ 函数。 """

    """ 2. 子进程到底在跑什么？
当你运行到 enumerate(train_loader) 时，
DataLoader 会通过底层的 多进程管理机制（Multiprocessing），
直接指派子进程去运行你定义的 DiabetesDataset 对象。

子进程在后台默默地运行 __getitem__(index)。

它们把拿到的数据塞进一个共享内存队列（Shared Queue）。

主进程的 for 循环只需要从这个“队列”里直接拿现成的 Batch 就可以了

 """
    model = Model()
    epoch_list = []
    loss_list = []
    loss_sum = 0
    # size_average=True 已被废弃，建议用 reduction='mean'
    criterion = torch.nn.BCELoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01) # 换成 Adam 更快

    for epoch in range(100):
        loss_sum = 0
        for i, data in enumerate(train_loader, 0):#0表示计数器从0开始
            x, y = data
            #内循环每一个i对应一个batch 此时x是一个batch的数据
            #每一个batch调整一下，一共调整 N / batch_size次
            # Forward
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss_sum += loss.item()
            # Backward
            optimizer.zero_grad()
            loss.backward()

            # Update
            optimizer.step()
        # 这样写最准确，无论你的数据集多大都能自动适配
        avg_loss = loss_sum / len(train_loader)
        epoch_list.append(epoch)
        loss_list.append(avg_loss)
    
    
    
    
    
    plt.plot(epoch_list,loss_list)
    plt.grid()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()
    
    # 训练结束后调用预测函数
    predict_test_data(model, './dataset/test.csv')