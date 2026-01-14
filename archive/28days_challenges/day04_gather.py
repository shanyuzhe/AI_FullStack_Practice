import torch

# 1. 模型输出的概率表 [Batch=3, Class=4]
# 每一行代表一个样本，每一列代表一个类别的概率
probs = torch.tensor([
    [0.1, 0.5, 0.2, 0.2],  # 样本1: 它是"狗"(index=1)的概率是 0.5
    [0.8, 0.1, 0.1, 0.0],  # 样本2: 它是"猫"(index=0)的概率是 0.8
    [0.1, 0.1, 0.1, 0.7]   # 样本3: 它是"鱼"(index=3)的概率是 0.7
])

# 2. 真实标签 (Index)
# Shape: [3, 1] -> 这是一个列向量，告诉我们需要取哪一列的数据
# 样本1取 index=1, 样本2取 index=0, 样本3取 index=3
target = torch.tensor([
    [1], 
    [0], 
    [3]
])

print(f"概率表形状: {probs.shape}")
print(f"标签形状:   {target.shape}")

# --- 核心挑战 ---
# 按照target把每一个样本的正确答案的估计值抓出来！
# 语法提示: torch.gather(input, dim, index)
# input: 源数据 (probs)
# dim:   我们要沿着哪个方向找？(想一想：是沿着行找列，还是沿着列找行？)
# index: 坐标数据 (target)

# TODO: 填空
result = torch.gather(probs, dim=1, index=target)
print("提取结果:\n", result)

# scatter--------------------------------------


# 1. 准备一张白纸 (Batch=3, Class=4)
one_hot = torch.zeros(3, 4)

# 2. 准备画笔 (value=1) 和 坐标 (target)
# target 还是刚才那个 [[1], [0], [3]]

# 3. 开始填色 (In-place 操作，注意有个下划线 scatter_)
# 含义：在 dim=1 的方向上，根据 target 的索引，把 value(1) 填进去
one_hot.scatter_(dim=1, index=target, value=1)

print("\n反向操作: One-Hot 编码结果:\n", one_hot)
# 预期结果：
# [0, 1, 0, 0]
# [1, 0, 0, 0]
# [0, 0, 0, 1]


print(torch.tensor([1,2,3]).shape)
print(torch.tensor([[1],[2],[3]]).shape)
print(torch.tensor([[1,2,3]]).shape)


data = torch.tensor([
    [10, 20, 30],  # Row 0
    [40, 50, 60],  # Row 1
    [70, 80, 90]   # Row 2
])

# index 的值代表"行号"
# [0, 2, 1] 对应 [第1列取Row0, 第2列取Row2, 第3列取Row1]
index = torch.tensor([[0, 2, 1]]) 

# dim=0：沿着"行"的方向（竖向）抓取
result = torch.gather(data, dim=0, index=index)

print("结果:\n", result)
# 结果应该是 [[10, 80, 60]]
# 形状也是 [1, 3]