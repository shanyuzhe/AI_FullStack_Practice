import torch
# 任务目标1：（求A[i]，B[i]之间的余弦相似度）
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

#任务2：
# 输入：A 是 (Batch, Dim)，B 是 (Batch, Dim)。
# 目标：输出一个 (Batch, Batch) 的矩阵，其中 Output[i, j] 是 A[i] 和 B[j] 的相似度。
# 你能想出这行代码怎么写吗？ (提示：用到矩阵乘法 matmul 或者 @)

numerator = A @ B.T #[4,4]
#这里就不能用广播了 A[4,128] B.T[128, 4], 不相等的维度不是1
#其实矩阵乘法的本质就是逐行做内积 A[i] · b[j]
norm_a = torch.norm(A, dim = 1,keepdim=True)
norm_b = torch.norm(B, dim = 1, keepdim=True)
denominator = norm_a * norm_b.T
#这里用广播 在这个例子中 这里用广播和@是等价的
#因为刚好不相等的维度是1 所以符合广播规则
cosine_sim = numerator / (denominator + 1e-8)

print(cosine_sim)#[4,4]
print(cosine_sim.size())