
```markdown
# 🚀 PyTorch 深度学习 & Python 算法双修实战

> **目标**：以**工程化**的思维构建深度学习知识体系（PyTorch），同时夯实编程内功（Python 算法），最终具备手写 Transformer 和独立解决复杂工程问题的能力。

这是我的个人全栈 AI 学习仓库。本项目采用**双线程**学习模式：
1.  🔥 **PyTorch 主线**：结合《PyTorch深度学习实践》与 Kaggle 实战，从 Tensor 基础到 LLM 实现。
2.  🐍 **Python 支线**：系统性训练数据结构与算法，提升代码的 **"Pythonic"** 程度与运行效率。

---

## 📂 项目结构 (Project Structure)

```text
AI_FULLSTACK/
├── 📜 .env                  # [关键] 环境变量配置 (PYTHONPATH=.)
├── 📂 28days_challenges/    # 🔥 PyTorch 深度学习进阶代码
├── 📂 python_gym/           # 🐍 Python 算法与数据结构特训 (New!)
│   ├── 📜 templates.py      # 刷题常用模板 (I/O, 二分, 堆等)
│   └── ...
├── 📂 lecture/              # 刘二大人《PyTorch深度学习实践》课程代码
├── 📂 kaggle/               # Kaggle 实战演练 (Titanic, MNIST 等)
├── 📂 utils/                # 🛠️ 通用工具箱 (两个计划通用)
│   ├── 📜 __init__.py       # 标识 utils 为 Python 包
│   ├── 📜 torch_playground.py   # 核心工具：环境配置、计时、张量体检
│   └── 📜 debug_checklist.md    # 🚑 救命指南：Debug 检查清单
└── 📜 README.md             # 项目说明文档

```

---

## 🛠️ 环境快速启动 (Quick Start)

为了避免 `ModuleNotFoundError` 并确保 `utils` 库能在所有子文件夹中被调用，请务必执行以下配置：

1. **创建 `.env` 文件**：在项目根目录下新建名为 `.env` 的文件，写入：
```properties
PYTHONPATH=.

```


2. **配置 VS Code**：确保安装了 Python 插件，并使用 `.vscode/settings.json` 强制终端加载环境变量（详见工程笔记）。

---

## 📅 路线图 A：PyTorch 深度学习 (28 Days)

> **核心目标**：独立写出训练闭环，掌握 Debug 技巧，手写 Transformer。

### Week 1: 维度直觉与张量操作

* [x] **Day 0**: 🛠️ 环境搭建与 utils 工具库封装
* [x] **Day 1**: Tensor 基础 + Device 管理 (CPU/GPU)
* [x] **Day 2**: 维度变换大通关 (reshape/view/permute)
* [x] **Day 3**: 广播机制 (Broadcasting) 深度解析
* [x] **Day 4**: 高级索引 (Advanced Indexing) & Mask
* [ ] **Day 5**: Einsum 爱因斯坦求和约定
* [ ] **Day 6**: NumPy 与 PyTorch 的互转陷阱
* [ ] **Day 7**: 🔄 **复盘**：输出维度变换速查表

### Week 2: 自动化梯度与训练闭环

* [ ] **Day 8**: Autograd 机制 (requires_grad, detach)
* [ ] **Day 9**: Loss 函数的输入细节 (Shape/Dtype)
* [ ] **Day 10**: Optimizer 与梯度管理 (zero_grad, clip_grad)
* [ ] **Day 11**: Train/Eval 模式的副作用 (Dropout/BN)
* [ ] **Day 12**: DataLoader 与 Dataset 自定义
* [ ] **Day 13**: 模型保存、加载与断点续训
* [ ] **Day 14**: 🔄 **复盘**：输出最小可运行训练模板

### Week 3: Debug 技巧与性能优化

* [ ] **Day 15**: NaN / Loss 爆炸的定位与修复
* [ ] **Day 16**: 显存管理与 Batch Size 调优
* [ ] **Day 17**: 混合精度训练 (AMP) 初探
* [ ] **Day 18**: DataLoader 加速 (num_workers)
* [ ] **Day 19**: 简单的性能分析 (Profiling)
* [ ] **Day 20**: 综合排错演练
* [ ] **Day 21**: 🔄 **复盘**：完善 `debug_checklist.md`

### Week 4: 对齐 LLM 项目 (Transformer)

* [ ] **Day 22**: Embedding + Positional Encoding
* [ ] **Day 23**: Self-Attention 实现 (Q, K, V)
* [ ] **Day 24**: Transformer Block (ResNet + LayerNorm)
* [ ] **Day 25**: 语言模型 (LM) 训练闭环
* [ ] **Day 26**: 理论面试题准备
* [ ] **Day 27**: 🎓 **验收**：手写最小 LM 并解释 Tensor Shape
* [ ] **Day 28**: 🎉 **总复盘**

---

## 📅 路线图 B：Python 算法健身房 (Python Gym)

> **核心目标**：提升代码“手感”与运行效率，积累常用算法模板。
> **每日标准**：AC 2 题 + 3 行复盘总结 + 维护 `templates.py`。

### Week 1: Collections 模块特训 (手感拉满)

* [x] **Day 1**: Counter (频次统计/异位词)
* [x] **Day 2**: defaultdict (分组/建图技巧)
* [ ] **Day 3**: deque (队列/BFS/滑窗) —— *使用 `time_block` 验证 O(1) 优势*
* [ ] **Day 4**: 排序技巧 (sorted key/lambda/元组)
* [ ] **Day 5**: 综合小测 (Collections 混用)
* [ ] **Day 6**: 错题回炉 + 模板固化
* [ ] **Day 7**: 补齐遗漏点

### Week 2: Itertools 与 基础算法

* [ ] **Day 8**: itertools 基础 (product/permutations)
* [ ] **Day 9**: groupby 思想 (数据压缩/连续段)
* [ ] **Day 10**: heapq (TopK/最小堆/模拟)
* [ ] **Day 11**: bisect (二分查找/维护有序序列)
* [ ] **Day 12**: 综合训练 (堆 + 二分)
* [ ] **Day 13**: 速度训练 (60分钟限时)
* [ ] **Day 14**: 整理堆与二分模板

### Week 3: 函数式编程与性能优化

* [ ] **Day 15**: 迭代器与生成器 (Yield/Map/Filter)
* [ ] **Day 16**: functools (lru_cache 记忆化搜索)
* [ ] **Day 17**: 快速 I/O 与常数优化 (sys.stdin)
* [ ] **Day 18**: 字符串与解析专项
* [ ] **Day 19**: 图/树模板巩固 (BFS/DFS)
* [ ] **Day 20**: 综合限时 (90分钟)
* [ ] **Day 21**: 产出“常用库速查表”

### Week 4: 稳定性与模拟面试

* [ ] **Day 22**: 双指针/滑窗专题
* [ ] **Day 23**: 二分查找专题
* [ ] **Day 24**: 单调栈/队列专题
* [ ] **Day 25**: 堆/TopK 专题
* [ ] **Day 26**: DP 动态规划入门
* [ ] **Day 27**: 模拟面试 (口播思路)
* [ ] **Day 28**: 统计错题与下阶段规划

---

## 🛠️ 工具库使用说明 (utils/torch_playground.py)

`utils` 库不仅服务于 PyTorch，其中的计时器和环境配置对 Python 算法练习同样有效。

**导入方式 (确保已配置 .env)：**

```python
from utils.torch_playground import *

```

### 核心功能

#### 1. `time_block(label)` —— 算法效率对比神器

在 Python 刷题时，用它来对比不同写法的速度差异（例如 list vs deque）。

```python
# 验证 list 在头部插入的性能劣势
with time_block("List Insert"):
    arr.insert(0, x) 

# 验证 deque 的 O(1) 优势
with time_block("Deque AppendLeft"):
    deq.appendleft(x)

```

#### 2. `inspect(tensor)` —— PyTorch 调试

查看 Tensor 的 Shape, Dtype, Device 和 Grad 状态。

---

## 🚑 Debug 快速通道

遇到报错、Loss 不下降或维度对不上时，请优先查阅：
👉 **[点击查看 PyTorch Debug Checklist](./utils/debug_checklist.md)**

