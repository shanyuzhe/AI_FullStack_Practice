**🧠 核心概念预习**
在开始写代码前，我们需要解决两个问题：
1. **为什么要 Mask？**
    ◦ 在处理变长序列（比如一句话长，一句话短）时，我们通常会把短的补齐（Padding）。
    ◦ 计算 Attention 时，我们**绝对不能**让模型把注意力分配给这些无意义的 Padding 符。
    ◦ **做法**：在 Softmax 之前，把 Padding 位置的分数设为 **负无穷 (`-inf`)**。这样 $e^{-\infty} \approx 0$，概率就被抹平了。
2. **什么是数值稳定（Numerical Stability）？**
    ◦ Softmax 的公式是 $\frac{e^{x_i}}{\sum e^{x_j}}。$
    ◦ 如果 `$x_i$` 很大（比如 100），$e^{100}$会直接导致计算机浮点数溢出（Overflow），变成 `nan`。
    ◦ **技巧**：利用数学特性 $\frac{e^{x_i}}{\sum e^{x_j}} = \frac{e^{x_i - C}}{\sum e^{x_j - C}}$。通常我们要让 $C = \max(x)$。这样所有指数部分最大也就是 $e^0=1$，永远不会溢出。

**🧠 深度复盘：为什么要“数值稳定”？**
为什么要 `x - x_max`
• **如果不减 Max**：$e^{100}$ 会变成 `inf`，计算结果变成 `nan`（Not a Number）。
• **减去 Max 后**：最大的数变成了 $e^0 = 1$，其他的数都是 $e^{\text{负数}} \in (0, 1)$。这就保证了永远不会溢出。

**记忆要点**

```python

**torch.softmax(logits.masked_fill(mask, -float('inf')),dim = 1)**

#torch.max() -> (values,indices)
**x_max = x_masked.max(dim = 1,keepdim = True)[0]**
```

**官方API**

```python
logits = torch.tensor([
    [2.0, 1.0, 0.5, 3.0], # 样本1 (假设最后1位是pad，虽然分数很高，但应该被mask掉)
    [1.5, 2.5, 4.0, 1.0]  # 样本2 (假设最后2位是pad)
])

mask = torch.tensor([
    [False, False, False, True ], # 样本1：前3个有效，最后1个Mask
    [False, False, True,  True ]  # 样本2：前2个有效，最后2个Mask
])

**torch.softmax(logits.masked_fill(mask, -float('inf')),dim = 1)**

```

**手搓原理展示**

```python
import torch
import torch.nn.functional as F

# 1. 模拟 Attention Logits (分数)
# Shape: [Batch=2, Seq=4]
logits = torch.tensor([
    [2.0, 1.0, 0.5, 3.0], # 样本1 (假设最后1位是pad，虽然分数很高，但应该被mask掉)
    [1.5, 2.5, 4.0, 1.0]  # 样本2 (假设最后2位是pad)
])

# 2. 创建 Mask
# 通常 1/True 表示"有效"，0/False 表示"Mask/无效" (或者是反过来，看具体约定)
# 这里我们约定：True = 需要被 Mask 掉 (Padding 部分)
mask = torch.tensor([
    [False, False, False, True ], # 样本1：前3个有效，最后1个Mask
    [False, False, True,  True ]  # 样本2：前2个有效，最后2个Mask
])

print("原始分数:\n", logits)
print("Mask矩阵:\n", mask)

def masked_softmax_stable(x, mask):
    """
    x: 输入分数 [Batch, Seq]
    mask: 掩码 [Batch, Seq], True表示需要被mask掉的位置
    """
    # --- Step 1: Masking ---
    # 目标：把 mask 为 True 的位置，x 对应的值改成一个极小的数 (比如 -1e9)
    # 提示：使用 masked_fill
    x_masked = x.clone() # 克隆一份避免修改原数据
    # TODO: 在这里写 Mask 操作

    x_masked = x_masked.masked_fill(mask,-1e9)
    

    # --- Step 2: 数值稳定处理 (Subtract Max) ---
    # 目标：为了防止 exp 溢出，每个样本都要减去该样本当前的最大值
    # 注意：求 max 的时候，得用 mask 后的数据，否则可能减去的是 padding 位的大数
    # keepdim=True 很重要！
    # TODO: 计算 max 并减去
    
    #error: x_max = x_masked.max(dim = 1,keepdim = True)
    #torch.max() -> (values,indices)
    x_max = x_masked.max(dim = 1,keepdim = True)[0]
    
    x_stable = x_masked - x_max

    # --- Step 3: 计算 Softmax ---
    # 公式： exp(x) / sum(exp(x))
    # TODO: 完成计算
    exp_x = torch.exp(x_stable)
    
    sum_exp_x = exp_x.sum(dim=1,keepdim=True)
    probs = exp_x / sum_exp_x

    return probs

softmaxed = masked_softmax_stable(logits,mask=mask)

print(f"手搓———Softmax计算结果为:\n{softmaxed}")
ans = torch.softmax(logits.masked_fill(mask, -float('inf')),dim = 1)
print(f"官方API———Softmax计算结果为:\n{ans}")
if softmaxed.equal(ans):
    print("答案正确！")
else:
    print("计算错误！")
```



### 🧠 核心概念：什么是 Gather？

想象你在做一个 **文本分类任务**（比如判断一句话的情感）：

- 模型输出给你一张 **“概率表”**（Probabilities）。
- 你有真实的 **“正确答案索引”**（Target Labels）。
- **任务**：你需要从“概率表”里，把“正确答案”对应的那个概率值**抓（Gather）**出来，用来算 Loss。

**一句话解释 `gather`：**

> “在指定的维度上，根据提供的索引，查表取值。”
> 

**一句话解释 scatter：**

> “在指定的维度上，根据提供的索引，填入src“
> 

**记忆要点：**

```python
**torch.gather(input, dim, index)**
input: 源数据 (probs)
dim:   我们要沿着哪个方向找？
index: 坐标数据 (target),和输出的形状是一致的

**tensor.scatter_(dim, index, src)**
tensor (Self): 画布 / 底板。
通常是我们初始化好的全 0 矩阵（用来画 One-Hot），或者需要被修改的原始数据。
dim: 沿着哪个方向填？
index: 坐标 / 坑位。
src: 颜料 / 素材。你要填进去的具体数值。
可以是常数（比如 1），也可以是另一个张量（把别的矩阵里的数搬过来）。

print(torch.tensor([1,2,3]).shape)#[3]
print(torch.tensor([[1],[2],[3]]).shape)#[3,1] 列向量
print(torch.tensor([[1,2,3]]).shape)#[1,3] 行向量

```

```python

probs = torch.tensor([
    [0.1, 0.5, 0.2, 0.2],  # 样本1: 它是"狗"(index=1)的概率是 0.5
    [0.8, 0.1, 0.1, 0.0],  # 样本2: 它是"猫"(index=0)的概率是 0.8
    [0.1, 0.1, 0.1, 0.7]   # 样本3: 它是"鱼"(index=3)的概率是 0.7
])

#横向gather 
target = torch.tensor([
    [1], 
    [0], 
    [3]
])

result = torch.gather(probs, dim=1, index=target)
---------------------------------------------------
提取结果:
 tensor([[0.5000],
        [0.8000],
        [0.7000]])
----------------------------------------------------

#纵向gather
target = torch.tensor([[1,1,1,0]])
result = torch.gather(prbs,dim=0,index=target)
---------------------------------------------------
提取结果:
 tensor([[0.8，0.1,0.1,0.2]])
----------------------------------------------------

index的形状要适配输入，同时也是输出的形状
横向抓取就定义为列向量 行数和input相同 列数为1
纵向抓取就定义为行向量 列数和input相同 行数为1

        
 
```

---


### 🧠 核心概念：并行的 If-Else

你可以把它看作是 Excel 里的 `IF` 函数，或者是 Python 三元表达式 `x if condition else y` 的**张量版**。

**语法**：

```python
output = torch.where(condition, x, y)

```

* **condition**: 判决书（Bool Tensor）。
* **x**: 如果判决为 `True`，从这里取值。
* **y**: 如果判决为 `False`，从这里取值。

---

### 🧪 小实验：手写 ReLU 激活函数

我们最熟悉的激活函数 ReLU，逻辑是：**“如果大于0就保持原样，小于0就变成0”**。
这简直就是为 `torch.where` 量身定做的场景。

请在你的代码框里试试这个：

```python
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

# result = ...

# print("原始数据:", a)
# print("ReLU结果:", result)

```

<details><summary>点击查看参考答案</summary>

```python
result = torch.where(a > 0, a, zeros)
# 或者更简单的写法 (利用广播):
# result = torch.where(a > 0, a, 0.0)

print("ReLU结果:", result)
# 预期输出: [0.0000, 2.0000, 0.0000, 0.8000]

```

</details>

---

### 🆚 灵魂拷问：`where` 和 `masked_fill` 有啥区别？

你可能会问：“我刚才用 `masked_fill` 也能做 mask，为啥还要 `where`？”

这是一个非常好的面试题，区别在于**灵活度**：

1. **`masked_fill` (简单粗暴)**：
* 逻辑：`if condition: value` (False 的时候保持不动)。
* 限制：**只能填一个固定的数**（比如 `-inf` 或 `0`）。
* 场景：Softmax Mask, Padding Mask。


2. **`torch.where` (双路合并)**：
* 逻辑：`if condition: A else: B`。
* 优势：**数据来源是两个张量**。它可以把两张图片融合，或者把两个模型的预测结果融合。
* 场景：比如“当置信度 > 0.9 时取模型 A 的预测，否则取模型 B 的预测”。



---

### ✅ Day 4 完美收官

* **Boolean Mask**: 基础的布尔索引 `a[a>0]`。
* **Masked Fill**: 原地修改，填常数（Softmax 必备）。
* **Gather**: 查表，从表里抓数据（CrossEntropy 必备）。
* **Scatter**: 填表，往表里写数据（One-Hot 必备）。
* **Where**: 逻辑分支，融合两份数据（ReLU/数据增强 必备）。

这一套组合拳打下来，你对 PyTorch **“张量操作”** 这块的理解已经超过绝大多数入门者了！

**明天（Day 5）我们进军 `einsum`，去挑战矩阵运算的“大统一理论”！好好休息！**