太棒了，环境搭建完毕，我们正式开启 **Day 1：Tensor 基础与设备管理**。

请在项目根目录下新建一个文件 `day01_tensor_basics.py`。

今天的核心目标是建立**“物理直觉”**：知道 Tensor 到底是个什么东西，它住在哪里（内存还是显存），以及它穿什么衣服（数据类型）。

---

### 🟢 任务 1：创建张量与类型感知

在 `day01_tensor_basics.py` 中输入以下代码。我们要观察 PyTorch 默认的数据类型，以及如何手动指定类型。

```python
from torch_playground import *

print("--- 1. 张量创建与 Dtype ---")

# 1. 普通浮点数 (通常用于模型权重、输入特征)
# 默认是 float32
x_float = torch.randn(3, 3) 
inspect(x_float, "Float Tensor (Default)")

# 2. 整数 (通常用于分类任务的标签 Label、索引 Index)
# 必须显式指定 dtype=torch.long (即 int64)
x_int = torch.randint(0, 10, (3, 3), dtype=torch.long)
inspect(x_int, "Long Tensor (For Labels)")

# 3. 这里的坑：如果不指定 dtype，python 列表会自动推断
# [1, 2, 3] -> long
# [1.0, 2.0] -> float
x_list = torch.tensor([1, 2, 3])
inspect(x_list, "Tensor from List")

```

**🧠 知识点：**

* **Float32**: 神经网络的标准血液。
* **Long (Int64)**: 神经网络的指路牌（第几类、第几个词）。**切记：交叉熵损失（CrossEntropyLoss）的标签必须是 Long！**

---

### 🟢 任务 2：设备管理 (CPU vs GPU)

Tensor 是有“户口”的。在 CPU 上的 Tensor 无法和 GPU 上的 Tensor 直接运算，会报错 `Expected all tensors to be on the same device`。

在文件中继续添加：

```python
print("\n--- 2. 设备搬运 (.to) ---")

# 1. 创建在 CPU 上
x_cpu = torch.zeros(2, 2)
inspect(x_cpu, "Created on CPU")

# 2. 搬运到 GPU (如果有的话)
# 这里的 DEVICE 是我们 playground 里自动获取的
x_gpu = x_cpu.to(DEVICE)
inspect(x_gpu, f"Moved to {DEVICE}")

# 3. 验证：不同设备不能运算
try:
    z = x_cpu + x_gpu
except RuntimeError as e:
    print(f"\n[预期报错捕获] {e}")
    print("👉 解决：必须确保两个张量在同一个设备上 (都 .to(DEVICE))")

```

**🧠 知识点：**

* `.to(device)` 是一个**拷贝**操作（如果设备不同）。
* 数据加载通常发生在 CPU（硬盘 -> 内存），训练发生在 GPU。所以训练循环里总有一句 `inputs = inputs.to(device)`。

---

### 🟢 任务 3：小实验 —— 速度对比 (CPU vs CUDA)

这是今天的重头戏。我们要验证一下，到底为什么我们要买显卡？
我们做一个简单的**矩阵乘法**，分别在 CPU 和 GPU 上跑。

```python
print("\n--- 3. 性能实验: CPU vs GPU ---")

# 定义一个大矩阵 (4000x4000)
# 矩阵乘法复杂度是 O(N^3)，计算量巨大
N = 4000 

# 准备数据
a_cpu = torch.randn(N, N)
b_cpu = torch.randn(N, N)

a_gpu = a_cpu.to(DEVICE)
b_gpu = b_cpu.to(DEVICE)

# --- 实验 A: CPU ---
print(f"开始 CPU 计算 ({N}x{N})...")
with time_block("CPU MatMul"):
    c_cpu = torch.matmul(a_cpu, b_cpu)

# --- 实验 B: GPU ---
# 预热 (Warm-up): GPU 第一次运行会有初始化开销，先空跑一次
torch.matmul(torch.randn(10,10).to(DEVICE), torch.randn(10,10).to(DEVICE))

print(f"开始 GPU 计算 ({N}x{N})...")
with time_block("GPU MatMul"):
    c_gpu = torch.matmul(a_gpu, b_gpu)
    # ⚠️ 关键：GPU 是异步执行的，必须同步等待所有命令跑完才能测出真实时间
    torch.cuda.synchronize() 

```

### 📝 Day 1 复盘作业

运行上面的代码，观察终端输出。然后在你的**学习笔记**（可以是 Notion、Obsidian 或简单的 Markdown 文件）里写下今天的复盘。

**请按照模板填写这 5 行（发给我看看）：**

1. **现象**：GPU 矩阵乘法比 CPU 快了多少倍？（看 time block 的输出）
2. **原因**：为什么 GPU 适合做矩阵乘法？（简单理解：GPU 有成千上万个小核心并行计算，CPU 只有几个大核心）。
3. **易错点**：做 `x + y` 时报了 RuntimeError，通常是因为什么？
4. **解决**：如何统一张量的设备？
5. **核心代码**：（把你觉得今天最有用的那句代码记下来，比如 `.to(DEVICE)` 或 `inspect`）

**准备好了吗？运行代码，告诉我你的实验结果！**
---
