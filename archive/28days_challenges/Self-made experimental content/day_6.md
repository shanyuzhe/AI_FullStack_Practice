这个实验的目标是让你亲眼看到：**NumPy 和 PyTorch 在“共享内存”时有多危险（或多高效）。**

### 🧪 实验指导：验证共享内存

请直接在你的 `torch_[playground.py](http://playground.py)` 中运行以下代码。

#### 1. NumPy ➡ Tensor (共享内存验证)

```python
import torch
import numpy as np

print("=== 实验 1: NumPy -> Tensor ===")
# 1. 创建一个 numpy 数组
n = np.array([1, 2, 3])

# 2. 转换成 tensor (默认共享内存)
t = torch.from_numpy(n)

# 3. 修改 tensor 的第0个元素
print(f"修改前 NumPy: {n}")
t[0] = 100
print(f"修改 tensor 后 NumPy: {n}") 

# 观察：n[0] 变了吗？(应该变了，说明是同一个内存地址)
```

#### 2. Tensor ➡ NumPy (共享内存验证)

```python
print("\n=== 实验 2: Tensor -> NumPy ===")
# 1. 创建一个 tensor
t2 = torch.tensor([10, 20, 30])

# 2. 转换成 numpy (默认共享内存)
n2 = t2.numpy()

# 3. 修改 numpy
n2[0] = 999
print(f"修改 numpy 后 Tensor: {t2}")

# 观察：t2[0] 变了吗？
```

#### 3. 💣 踩坑：什么时候**不共享**？

试着把 Tensor 放到 GPU 上，或者做个切片拷贝，再转 numpy。

```python
print("\n=== 实验 3: 什么时候不共享？ ===")
# 情况 A: Tensor 在 GPU 上 (必须先 .cpu() 才能转 numpy，这时会发生拷贝)
# t_cuda = torch.tensor([1, 2]).cuda()
# n_cuda = t_cuda.cpu().numpy() 
# 改 n_cuda 不会影响 t_cuda

# 情况 B: 内存不连续 (Contiguous)
a = np.array([1, 2, 3, 4])
b = a[::2] # 步长为2，切片，此时 b 内存不连续
# t = torch.from_numpy(b) # 报错！ValueError
t = torch.from_numpy(b.copy()) # 必须 copy，此时就不共享了
```

### 📝 实验总结
- NumPy 数组和 PyTorch 张量在默认情况下是共享内存的，修改一个会影响另一个。
- 当 Tensor 在 GPU 上，或者内存不连续时，转换为 NumPy 会发生拷贝，不再共享内存。
- 内存共享可以提高效率，但也需要小心修改，避免意外改变数据。