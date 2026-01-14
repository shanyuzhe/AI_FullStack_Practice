import torch

# 1. 模拟网络输出的 logits (有正有负)
a = torch.tensor([-1.5, 2.0, -0.1, 0.8])

# 2. 准备备选方案
# 方案 X (True): 保持原样 -> 就是 a 自己
# 方案 Y (False): 变成 0  -> 一个全 0 的张量 (或者直接写 0 也可以，会自动广播)
zeros = torch.zeros_like(a)

# --- 核心挑战 ---
# TODO: 请用 torch.where 实现 ReLU 逻辑
# 语法: torch.where(条件, True时的值, False时的值)

result = torch.where(a > 0, a, zeros)

print("原始数据:", a)
print("ReLU结果:", result)