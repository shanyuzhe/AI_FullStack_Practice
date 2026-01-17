import torch
import numpy as np

print("=== 实验一：NumPy 数组与 PyTorch 张量共享内存 ===")

n = np.array([1, 2, 3, 4, 5])
t = torch.from_numpy(n)
print("初始 NumPy 数组:", n)
t[0] = 100
print("修改后的 PyTorch 张量:", t)

#numpy 数组和 PyTorch 张量共享内存

# 2. 实验二：PyTorch 张量与 NumPy 数组共享内存
print("\n=== 实验二：PyTorch 张量与 NumPy 数组共享内存 ===")
t2 = torch.tensor([10, 20, 30, 40, 50])
n2 = t2.numpy()
print("初始 PyTorch 张量:", t2)
n2[1] = 200
print("修改 NumPy 数组后 PyTorch 张量:", t2)

print("\n=== 实验三：什么时候不共享内存 ===")

# 情况 A: Tensor 在 GPU 上 (必须先 .cpu() 才能转 numpy，这时会发生拷贝)
t_cuda = torch.tensor([1, 2]).cuda()
n_cuda = t_cuda.cpu().numpy() 
# 改 n_cuda 不会影响 t_cuda

# 情况 B: 内存不连续（Contiguous）
a = np.array([1, 2, 3,4])
b = a[::2]# 取步长为2的切片，内存不连续
# t_b = torch.from_numpy(b) # 会报错，因为内存不连续
# 需要先变成连续的内存
t = torch.from_numpy(b.copy())
print("内存不连续时，必须先拷贝才能转换为张量:", t)