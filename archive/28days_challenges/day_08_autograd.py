import torch

def print_grad_info(tensor, name):
    """辅助函数：打印张量的梯度信息"""
    print(f"--- [{name}] Gradient Info ---")
    if tensor.grad is not None:
        print(f"Shape: {tensor.grad.shape}")
        print(f"Values: {tensor.grad}")
    else:
        print("None (无梯度)")
    print("-" * 30)

print("🚀 Day 8: Autograd 闭环训练开始\n")

# ==========================================
# 1. 准备数据 (requires_grad=True 是核心)
# ==========================================
# 这是一个形状为 (3,) 的向量
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
print(f"输入 x: {x}\n")

# ==========================================
# 场景 A: 正常反向传播 (The Happy Path)
# 计算图: x -> y (平方) -> z (乘法) -> loss (求和)
# ==========================================
print(">>> 场景 A: 正常反向传播")

y = x ** 2        # 算子1: PowerBackward
z = y * 4         # 算子2: MulBackward
loss = z.sum()    # 终点: SumBackward

# 反向传播前，清空梯度（虽然这里是第一次，但养成好习惯）
if x.grad is not None: x.grad.zero_()

loss.backward()   # 🚀 启动引擎！

print(f"计算过程: loss = sum((x^2) * 4)")
# 数学推导: 
# z = 4x^2 -> dz/dx = 8x
# x=[1,2,3] -> grad=[8, 16, 24]
print_grad_info(x, "x (Normal)")


# ==========================================
# 场景 B: 使用 .detach() (梯度截断/剪枝)
# ==========================================
print("\n>>> 场景 B: 使用 .detach() 截断梯度")

# 重置 x 的梯度
x.grad.zero_()

y = x ** 2
y_detached = y.detach()  # ✂️ 梯度在这里断开了
z = y_detached * 4
loss = z.sum()

# --- 修改开始 ---
if loss.requires_grad:
    loss.backward()
else:
    print("⚠️ 提示: loss.requires_grad 为 False，无法进行 backward()")
    print("这证明 detach() 成功切断了计算图！")
# --- 修改结束 ---

# 验证 x 的梯度（应该是 0，或者保持为被 zero_() 后的状态）
print_grad_info(x, "x (After detach)")

# ==========================================
# 场景 C: 使用 with torch.no_grad() (闭眼模式)
# 整个上下文都不追踪梯度，常用于推理/测试
# ==========================================
print("\n>>> 场景 C: 使用 torch.no_grad()")

with torch.no_grad():
    y = x ** 2
    z = y * 4
    loss = z.sum()
    
    print(f"Loss requires_grad状态: {loss.requires_grad}") # 应该是 False
    
    try:
        loss.backward()
    except RuntimeError as e:
        print(f"❌ 报错捕获: {e}")
        print("原因: 根本就没有构建计算图，无法 backward")

print("\n✅ Day 8 训练完成！")