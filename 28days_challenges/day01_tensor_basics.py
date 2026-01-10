from utils.torch_playground import *

### 🟢 任务 1：创建张量与类型感知
print('----1.张量创建与Dtype----')

# 创建不同dtype的张量
# 创建默认张量
with time_block("Create Default Tensor"):
    x_default = torch.randn(3, 3)
    inspect(x_default, name = "x_default")

# 创建整数张量 必须显式指定dtype为torch.long
x_int = torch.randint(0, 10, (3, 3), dtype=torch.long)
inspect(x_int, name = "x_int") 

#这里有坑 如果指定 torch会自动推断为int
x_list = torch.tensor([1, 2, 3, 4])
inspect(x_list, name = "x_list")


# **🧠 知识点：**

# * **Float32**: 神经网络的标准血液。
# * **Long (Int64)**: 神经网络的指路牌（第几类、第几个词）。
# **切记：交叉熵损失（CrossEntropyLoss）的标签必须是 Long！**
# * **Int32/Int16/Int8**: 节省内存的利器，但要小心溢出。


## 🟢 任务 2：张量设备管理(cpu vs gpu)
# Tensor 是有“户口”的。在 CPU 上的 Tensor 无法和 GPU 上的 Tensor 直接运算，
# 会报错 `Expected all tensors to be on the same device`。


print("\n--- 2. 设备搬运 (.to) ---")

# 1. 创建在CPU上
x_cpu = torch.zeros((2, 2))
inspect(x_cpu, name="x_cpu")

# 2. 搬运到GPU（如果可用）
# 这里的DEVICE是我们从utils/torch_playground.py自动选择的设备
x_gpu = x_cpu.to(DEVICE)
inspect(x_gpu, name="x_gpu")

# 3. 验证：设备不同不能计算
try:
    _ = x_cpu + x_gpu
except RuntimeError as e:
    print(f"✅ 正确捕获设备不匹配错误: {e}")
    print("⚠️ 提示：确保所有张量在同一设备上再进行运算！（都.to(DEVICE)）")

# **🧠 知识点：**
# * `.to(device)` 是一个**拷贝**操作（如果设备不同）。
# * 数据加载通常发生在 CPU（硬盘 -> 内存），训练发生在 GPU。
# 所以训练循环里总有一句 `inputs = inputs.to(device)`。


### 🟢 任务 3：小实验 —— 速度对比 (CPU vs CUDA)
print("\n--- 3. 性能实验: CPU vs GPU ---")
N = 10000
a_cpu = torch.randn((N, N))
b_cpu = torch.randn((N, N))

a_gpu = a_cpu.to(DEVICE)
b_gpu = b_cpu.to(DEVICE)

print(f"开始 CPU 计算 ({N}x{N})...")
with time_block("CPU Matrix Multiplication"):
    c_cpu = a_cpu @ b_cpu
    
print(f"开始 GPU 计算 ({N}x{N})...")
with time_block("GPU Matrix Multiplication"):
    c_gpu = a_gpu @ b_gpu
     # ⚠️ 关键：GPU 是异步执行的，必须同步等待所有命令跑完才能测出真实时间
    torch.cuda.synchronize() 


