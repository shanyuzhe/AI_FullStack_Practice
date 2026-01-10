
```markdown
# 🚀 PyTorch 深度学习实战

> **目标**：从零构建深度学习知识体系，彻底掌握张量操作、训练闭环、Debug 技巧，最终手写 Transformer/LLM。

这是我的个人 PyTorch 深度学习练习仓库。本项目不仅包含基础语法的刻意练习，还结合了 **《PyTorch深度学习实践》（刘二大人）** 的课程代码以及 **Kaggle** 经典赛事的实战代码。

---

## 📂 项目结构

```text
PYTORCH_PRACTICE/
├── 📂 28days_challenges/    # [NEW] 🔥 28天进阶挑战代码存放处
├── 📂 lecture/              # 刘二大人《PyTorch深度学习实践》课程代码
├── 📂 kaggle/               # Kaggle 实战演练 (Titanic, MNIST 等)
├── 📂 utils/                # 🛠️ 通用工具箱
│   ├── 📜 torch_playground.py   # 核心工具库：环境配置、计时、张量体检
│   └── 📜 debug_checklist.md    # 🚑 救命指南：Debug 检查清单
└── 📜 README.md             # 项目说明文档

```

---

## 📅 28天进阶路线图 (Roadmap)

所有挑战代码请存放于 `28days_challenges/` 文件夹中。

### Week 1: 维度直觉与张量操作 (NumPy ↔ PyTorch)

> **目标**：张量维度、广播、索引、reshape/permute 不再出错。

* [x] **Day 0**: 🛠️ 环境搭建与工具库准备 (已归档至 `utils/`)
* [ ] **Day 1**: Tensor 基础 + dtype/device 管理
* [ ] **Day 2**: 维度变换大通关 (reshape/view/permute)
* [ ] **Day 3**: 广播机制 (Broadcasting) 深度解析
* [ ] **Day 4**: 高级索引 (Advanced Indexing) & Mask
* [ ] **Day 5**: Einsum 爱因斯坦求和约定 (LLM 必备)
* [ ] **Day 6**: NumPy 与 PyTorch 的互转陷阱
* [ ] **Day 7**: 🔄 **复盘**：输出维度变换速查表

### Week 2: 自动化梯度与训练闭环

> **目标**：独立写出 Dataset → Model → Loss → Optimizer 的完整闭环。

* [ ] **Day 8**: Autograd 机制 (requires_grad, detach)
* [ ] **Day 9**: Loss 函数的输入细节 (Shape/Dtype 避坑)
* [ ] **Day 10**: Optimizer 与梯度管理 (zero_grad, clip_grad)
* [ ] **Day 11**: Train/Eval 模式的副作用 (Dropout/BN)
* [ ] **Day 12**: DataLoader 与 Dataset 自定义
* [ ] **Day 13**: 模型保存、加载与断点续训
* [ ] **Day 14**: 🔄 **复盘**：输出最小可运行训练模板

### Week 3: Debug 技巧与性能优化

> **目标**：遇到 NaN、Loss 不降、显存爆炸，知道如何定位。

* [ ] **Day 15**: NaN / Loss 爆炸的定位与修复
* [ ] **Day 16**: 显存管理与 Batch Size 调优
* [ ] **Day 17**: 混合精度训练 (AMP) 初探
* [ ] **Day 18**: DataLoader 加速 (num_workers, pin_memory)
* [ ] **Day 19**: 简单的性能分析 (Profiling)
* [ ] **Day 20**: 综合排错演练
* [ ] **Day 21**: 🔄 **复盘**：完善 `debug_checklist.md`

### Week 4: 对齐 LLM 项目 (Transformer 实现)

> **目标**：能讲清 Transformer 训练细节与维度变化。

* [ ] **Day 22**: Embedding + Positional Encoding
* [ ] **Day 23**: Self-Attention 实现 (Q, K, V)
* [ ] **Day 24**: Transformer Block (ResNet + LayerNorm)
* [ ] **Day 25**: 语言模型 (LM) 训练闭环
* [ ] **Day 26**: 理论面试题准备 (Scale dot-product, BN vs LN)
* [ ] **Day 27**: 🎓 **验收**：手写最小 LM 并解释 Tensor Shape
* [ ] **Day 28**: 🎉 **总复盘**

---

## 🚑 Debug 快速通道

遇到报错、Loss 不下降或维度对不上时，请优先查阅 `utils` 目录下的检查清单：

👉 **[点击查看 PyTorch Debug Checklist](https://www.google.com/search?q=./utils/debug_checklist.md)**

---

## 🛠️ 工具库使用说明 (utils/torch_playground.py)

`torch_playground.py` 是本项目的核心工具库，封装了**环境配置**、**随机种子固定**、**张量调试**和**性能计时**等常用功能，现已移动至 `utils` 文件夹。

建议在每个训练脚本的开头导入（确保 `utils` 包可被 Python 识别）：

```python
# 如果脚本在根目录下运行
from utils.torch_playground import *

```

导入时会自动执行：

1. **自动选择设备**：优先使用 CUDA，将结果存入全局变量 `DEVICE`。
2. **固定随机种子**：默认种子 `277527`，涵盖 CPU、GPU、NumPy 和 Python 原生随机库，并配置 cuDNN 为确定性模式。

### 核心功能展示

#### 1. `inspect(tensor, name)` —— 张量体检神器

不再被 `print(tensor)` 的海量数据刷屏，只看最关键的元数据（Shape, Dtype, Device, Grad）。

```python
x = torch.randn(3, 4, requires_grad=True, device=DEVICE)
inspect(x, "Input Tensor")

```

**输出示例：**

```text
[Inspect] Input Tensor: 
   Shape:  (3, 4)
   Dtype:  torch.float32
   Device: cuda:0
   Grad:   True (Grad Fn: False)
------------------------------

```

#### 2. `time_block(label)` —— 代码块计时器

使用 `with` 语句包裹代码块，自动计算并打印运行耗时（基于 `perf_counter` 的高精度计时）。

```python
with time_block("Matrix Multiplication"):
    # 你的计算逻辑
    y = torch.matmul(x, x.T)
    time.sleep(0.1)

```

**输出示例：**

```text
[Time] Matrix Multiplication cost: 0.100452 sec

```

#### 3. 全局变量 `DEVICE`

无需每次手写 `torch.device('cuda' if ... else ...)`，直接使用全局变量。

```python
model = MyModel().to(DEVICE)
data = data.to(DEVICE)

```

---

### 📝 Debug 检查清单

遇到报错或 Loss 异常时，请对照 [utils/debug_checklist.md](https://www.google.com/search?q=./utils/debug_checklist.md) 进行排查。

```

### 💡 一个小建议

既然你把文件移到了 `utils` 文件夹，为了让 Python 把它识别为一个包（Package），并且方便你在 `28days_challenges` 里面的代码调用它，建议你在 `utils` 文件夹里新建一个空的 `__init__.py` 文件。

**文件结构变成：**
```text
utils/
├── __init__.py  <-- 新建这个空文件
├── torch_playground.py
└── debug_checklist.md

```
