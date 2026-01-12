### 核心口诀（一定要背下来）：

> **从右向左看，维度要么相等，要么其中一个是 1（不相等时只有维度是1才能广播）**
> 

如果两个数组的维度不满足这个条件，就会报错。

### 经典案例自测（不用跑代码，口算出形状）

假设：

- `A.shape = (32, 1, 256)`
- `B.shape = (1, 64, 256)`

请问：`C = A + B`，`C` 的形状是多少？

**答案：(32, 64, 256)**

**解析：**

1. **对齐**（从右往左）：
A: 32, 1, 256
B: 1, 64, 256
2. **比较**：
    - 最右边：256 vs 256（相等 ✅）
    - 中间：1 vs 64（一个是1 ✅，结果取 64）
    - 最左边：32 vs 1（一个是1 ✅，结果取 32）
3. **结果**：(32, 64, 256)

**TIPS:**

**A:32,2,256**

**B:1,64,256**

**这是错的！会报错！A有一维不是1**

- **是 1**：就像拥有了**“影分身术”**。它会自动把这一份数据“复制粘贴”，直到和对方的维度一样大。（注意：这是虚拟复制，不占内存，所以很快）。
- **不是 1 且不相等**：就是**“那咋办嘛”**。比如你有 2 行数据，对方有 3 行，计算机不知道该把你的第 1 行给谁，第 2 行给谁，于是直接报错。

```python
import torch
# TODO任务目标1：（求A[i]，B[i]之间的余弦相似度）
# 有两个矩阵 A 和 B，它们的形状都是 (batch_size, hidden_dim)。 
# 你需要计算每一行向量的余弦相似度（Cosine Similarity）。
batch_size = 4
hidden_dim = 128

torch.manual_seed(42)
A = torch.randn(batch_size, hidden_dim) # [4, 128]
B = torch.randn(batch_size, hidden_dim) # [4, 128]

numerator = (A * B).sum(dim = 1) # [4, 1]
norm_a = torch.norm(A,p=2,dim = 1,keepdim=True)
norm_b = torch.norm(B,p=2,dim = 1,keepdim=True)
#p=1绝对值之和（曼哈顿距离）
#p=2平方和开根号（直线距离）
denominator = norm_a * norm_b

cosine_sim = numerator / (denominator + 1e-8)
print(cosine_sim)
print(cosine_sim.size())

#TODO任务2：
# 输入：A 是 (Batch, Dim)，B 是 (Batch, Dim)。
# 目标：输出一个 (Batch, Batch) 的矩阵，其中 Output[i, j] 是 A[i] 和 B[j] 的相似度。
# 你能想出这行代码怎么写吗？ (提示：用到矩阵乘法 matmul 或者 @)

numerator = A @ B.T #[4,4]
#这里就不能用广播了 A[4,128] B.T[128, 4], 不相等的维度不是1
#其实矩阵乘法的本质就是逐行做内积 A[i] · b[j]
norm_a = torch.norm(A, dim = 1,keepdim=True)#[4,1]
norm_b = torch.norm(B, dim = 1, keepdim=True)
denominator = norm_a * norm_b.T#[4,4]
#这里用广播 在这个例子中 这里用广播和@是等价的
#因为刚好不相等的维度是1 所以符合广播规则
cosine_sim = numerator / (denominator + 1e-8)

print(cosine_sim)#[4,4]
print(cosine_sim.size())
```