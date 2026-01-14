import torch
import torch.nn.functional as F
#28 * 28的图片
class InceptionA(torch.nn.Module):
    def __init__(self, in_channels):
        super(InceptionA, self).__init__()
        self.branch1x1 = torch.nn.Conv2d(in_channels, 16, kernel_size=1)

        self.branch_pool = torch.nn.Conv2d(in_channels, 24, kernel_size=1)
        
        self.breanch5x5_1 = torch.nn.Conv2d(in_channels, 16, kernel_size=1)
        self.breanch5x5_2 = torch.nn.Conv2d(16, 24, kernel_size=5, padding=2)
        
        self.branch3x3_1 = torch.nn.Conv2d(in_channels, 16, kernel_size=1)
        self.branch3x3_2 = torch.nn.Conv2d(16, 24, kernel_size=3, padding=1)
        self.branch3x3_3 = torch.nn.Conv2d(24, 24, kernel_size=3, padding=1)
    
    def forward(self, x):
        branch1x1 = self.branch1x1(x) #--->16层
        
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)#--->24层
        
        breanch5x5 = self.breanch5x5_1(x)
        breanch5x5 = self.breanch5x5_2(breanch5x5)#--->24层
        
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)
        branch3x3 = self.branch3x3_3(branch3x3)#--->24层
        
        outputs = [branch1x1, branch_pool, breanch5x5, branch3x3]
        return torch.cat(outputs, dim=1)  # 在通道维度上拼接 16+24+24+24=88

#B C W H
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(88, 20, kernel_size=5)
        self.inception1 = InceptionA(in_channels=10)
        self.inception2 = InceptionA(in_channels=20)
        self.fc1 = torch.nn.Linear(1408, 50)
        self.fc2 = torch.nn.Linear(50, 10)
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = self.inception1(x)
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = self.inception2(x)#28->24->12->8 ->4
        #4 * 4 * 88 =1408
        x = x.view(x.size(0), 1408)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x