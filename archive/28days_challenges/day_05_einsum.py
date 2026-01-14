import torch
from utils import torch_playground

# --- 实验准备 ---
print("=== 任务 1: 理解 einsum 与 Batch Matmul ===")

# 设定维度：Batch=2, I=3(行), K=4(中间维), J=5(列)
B, I, K, J = 2, 3, 4, 5

# 创建随机张量 (模拟数据)
# lhs: Left Hand Side (左矩阵)
# rhs: Right Hand Side (右矩阵)
lhs = torch.randn(B, I, K)
rhs = torch.randn(B, K, J)

# --- 方式 A: 传统笨办法 (torch.bmm) ---
# bmm 专门用于 batch matrix multiplication
res_standard = torch.bmm(lhs, rhs)

# --- 方式 B: Einsum 大法 ---
# 你的思考过程：
# 1. lhs 的形状是 [B, I, K] -> 记作 'bik'
# 2. rhs 的形状是 [B, K, J] -> 记作 'bkj'
# 3. 这里的 'k' 是中间维度，是要被“吃掉”的（求和）
# 4. 这里的 'b' 是 Batch，要保留；'i' 和 'j' 也要保留
# 5. 所以结果应该是 'bij'

res_einsum = torch.einsum('bik, bkj -> bij', lhs ,rhs)

torch_playground.inspect(res_standard)
torch_playground.inspect(res_einsum)


# 创建一个 2x2 的特征图，数值全是 1
# 形状: [1, 1, 2, 2] -> b c h w
x = torch.ones(1, 1, 2, 2) 

print(f"原始数据:\n{x}")
# 输出:
# [[[[1., 1.],
#    [1., 1.]]]]

# --- 操作 1: Einsum 'bchw -> bc' ---
# h 和 w 消失了 -> 意味着把 4 个格子里的数加起来
res_sum = torch.einsum('bchw -> bc', x)
torch_playground.inspect(res_sum)
# 结果是 4.0 (1+1+1+1) -> 这证明它是求和！

# --- 操作 2: 如果是“直接消了维度”（切片/取值）---
# 比如我们要取左上角的那个点，不要其他的
res_slice = x[:, :, 0, 0]
torch_playground.inspect(res_slice)
# 结果是 1.0 -> 这才是“直接消了维度”



print("\n=== 任务 2: 用 einsum 实现 Attention Score ===")

# --- 模拟 Transformer 的维度 ---
# batch_size=2
# heads=8 (多头)
# seq_len=10 (序列长度)
# head_dim=64 (每个头的特征维度)
b, h, s, d = 2, 8, 10, 64

# 创建 Query 和 Key
Q = torch.randn(b, h, s, d)
K = torch.randn(b, h, s, d)

# --- 方式 A: 传统写法 (痛苦面具) ---
# 1. 先把 K 转置：[b, h, s, d] -> [b, h, d, s]
# 2. 再矩阵乘法
res_attn_standard = torch.matmul(Q, K.transpose(-2, -1))

# --- 方式 B: Einsum 写法 (优雅) ---
# 你的思考过程：
# Q: [b, h, s, d] -> 记作 'bhid' (用 i 代表 Q 的序列长度)
# K: [b, h, s, d] -> 记作 'bhjd' (用 j 代表 K 的序列长度)
# 注意：虽然 i 和 j 数值都是 10，但在逻辑上它们是两个方向（行和列），所以用不同字母。
#
# 我们要消去哪个维度？-> 'd' (特征维度)。
# 我们要保留哪些？-> 'b', 'h', 'i', 'j'。
#
# 结果形状应该是 [Batch, Heads, Seq_Q, Seq_K] -> 即 Attention Score Map
res_attn_einsum = torch.einsum('bhid,bhjd->bhij', Q, K)

# --- 验证环节 ---
print(f"Q shape: {Q.shape}")
print(f"K shape: {K.shape}")
print(f"Attention Map shape: {res_attn_einsum.shape}") # 应该是 [2, 8, 10, 10]

is_match_attn = torch.allclose(res_attn_standard, res_attn_einsum, atol=1e-6)
print(f"✅ Attention 核心计算一致: {is_match_attn}")
