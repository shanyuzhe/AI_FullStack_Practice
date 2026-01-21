>#大模型开发工程师进阶路线图 (Hardcore Mode)
 **Warning**: 此路线图不包含任何"速成"或"科普"内容。目标是达到工业界核心研发岗（DeepSeek/OpenAI）的入职门槛。每日预计投入 8-10 小时。
>

> 📘 **For AI Assistants**: See [CLAUDE.md](./CLAUDE.md) for codebase architecture, development setup, and engineering guidelines.

## 🎯 核心竞争力 Checkbox

*(每阶段结束必须通过的自测题)*

- [ ]  **Math**: 能否手推 Softmax 的 Jacobian 矩阵？能否证明 KL 散度非负？
- [ ]  **Arch**: 能否徒手写出 RoPE 的旋转矩阵并解释其远程衰减特性？
- [ ]  **Sys**: 能否解释 ZeRO-3 相比 ZeRO-2 多切分了什么？通信量有何变化？
- [ ]  **CUDA**: 能否用 Triton 写一个简单的 Vector Add 或 Softmax 算子？
- [ ]  **KvCache**: 为什么 PagedAttention 能解决显存碎片化？Block Table 是怎么维护的？

---

## 📅 Stage 1: Foundation & Math (彻底打通底层)

**Goal**: 能够不依赖 PyTorch Autograd 手写神经网络反向传播。

### 1.1 硬核数学 (Math for DL)

- **重点**: 矩阵微积分 (Matrix Calculus)、信息论。
- **资料**:
    - Paper: *The Matrix Calculus You Need For Deep Learning* (Parr & Howard).
    - Concept: Jacobian, Hessian, KL Divergence, Cross Entropy vs MSE.
- **Task**:
    - [ ]  手推 Transformer 中 Multi-Head Attention 的反向传播梯度。

### 1.2 深度 NLP 基础 (CS224N)

- **课程**: **Stanford CS224n: Natural Language Processing with Deep Learning**
- **重点**: Word2Vec, RNN/LSTM/GRU (理解序列建模的历史), Attention Mechanism.
- **Task**:
    - [ ]  完成 Assignment 3 (Dependency Parsing) & Assignment 4 (NMT with Attention).

### 1.3 系统雏形 (Build Micrograd)

- **项目**: 参考 Karpathy 的 `micrograd`，但用 Python + NumPy 实现一个支持 Tensor 的 Autograd 引擎。
- **Task**:
    - [ ]  实现 `Tensor` 类，支持 `matmul`, `conv2d` 的 `backward`。
    - [ ]  用你写的引擎训练一个 MLP 拟合 `sin(x)`。

---

## 📅 Stage 2: Architecture & Systems (Llama from Scratch)

**Goal**: 徒手复现 Llama 架构，并理解分布式训练原理。

### 2.1 架构拆解 (CS336)

- **课程**: **Stanford CS336: Language Modeling from Scratch** (最硬核，必看)
- **重点**: Pre-training, Post-training, Scaling Laws.
- **组件手撸**:
    - [ ]  **RoPE**: 旋转位置编码 (Rotary Positional Embeddings).
    - [ ]  **SwiGLU**: 激活函数实现与导数。
    - [ ]  **RMSNorm**: 相比 LayerNorm 的区别与实现。
    - [ ]  **GQA/MQA**: Grouped Query Attention 代码实现。

### 2.2 分布式训练基础 (3D Parallelism)

- **理论**: Data Parallelism (DP), Tensor Parallelism (TP), Pipeline Parallelism (PP).
- **论文**: *Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism*.
- **Task**:
    - [ ]  阅读 DeepSpeed ZeRO Paper (ZeRO-1/2/3 的区别).
    - [ ]  在单机多卡环境配置 `torch.distributed.launch`，跑通 DDP 训练。

### 2.3 显存优化 (FlashAttention)

- **理论**: IO-Awareness, Tiling, Recomputation.
- **论文**: *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*.
- **Task**:
    - [ ]  阅读 FlashAttention V1/V2 源码 (CUDA/Triton 部分).
    - [ ]  **入门 Triton**: 用 OpenAI Triton 写一个简单的矩阵乘法算子。

---

## 📅 Stage 3: Full-stack LLM & Research (全栈实战)

**Goal**: 训练 Tiny-Llama 并复现 R1 推理策略。

### 3.1 Pre-train & SFT Pipeline

- **项目**: 从零构建 `min-llama` (参考 `lit-gpt` 或 `nanoGPT`).
- **数据**: 使用 `TinyStories` 数据集训练一个 15M-50M 参数的模型。
- **Task**:
    - [ ]  实现完整的 Pre-training Loop (WandB 监控 Loss, Grad Norm).
    - [ ]  实现 SFT (Supervised Fine-Tuning) 流程，使用 Alpaca 格式数据微调。

### 3.2 Alignment (RLHF/DPO)

- **课程**: **Berkeley CS285** (选修 RL 基础部分) 或直接看 DPO 论文.
- **论文**: *Direct Preference Optimization: Your Language Model is Secretly a Reward Model*.
- **Task**:
    - [ ]  在你的 Tiny-Llama 上实现 DPO Loss。
    - [ ]  对比 SFT 模型与 DPO 模型的输出差异。

### 3.3 推理与推理优化 (Inference)

- **策略**: DeepSeek-R1 的冷启动策略 (Cold Start).
    - **Concept**: Chain of Thought (CoT), Reasoning Tokens.
    - **Task**: 构造一个简单的 CoT 数据集 (Math word problems)，微调你的模型，观察是否出现"思考过程"。
- **系统**: **vLLM & PagedAttention**.
    - **Task**: 阅读 vLLM 源码，理解 Block Manager 如何管理 KV Cache 物理块与逻辑块的映射。

---

---

# 🪜 Stage 1: Daily Execution Plan (Start Now)

> **Execution Rule**: 每天核心 4 小时深度工作（不包含看视频发呆的时间）。Talk is cheap, show me the code.
> 

## Week 1: 重新发明轮子 (Autograd & Backprop)

目标：彻底祛魅 PyTorch。不依赖 `.backward()` 手写神经网络训练。

- [ ]  **Day 1: 矩阵微积分 (Matrix Calculus)**
    - **Input**: 阅读 *The Matrix Calculus You Need For Deep Learning* (Parr & Howard) 前 3 章。
    - **Code**: 手推 $Y = WX + b$ 和 Softmax 的 Jacobian 矩阵。
    - **Output**: 在纸上写出 CrossEntropyLoss 对 Logits 的梯度推导过程（你会发现它惊人的简单：$P - Y$）。
- [ ]  **Day 2: Micrograd (Scalar Autograd)**
    - **Input**: Andrej Karpathy 的 YouTube 视频 *"The spelled-out intro to neural networks..."*
    - **Code**: 跟写 `micrograd`。实现 `Value` 类，支持 `+`, `*`, `pow`, `relu`。
    - **Output**: 用你的引擎训练一个 MLP 拟合 `f(x) = 3x^2 - 4x + 5`。
- [ ]  **Day 3: Tensor Autograd (NumPy Edition)**
    - **Task**: 将 Micrograd 升级为张量版本。
    - **Code**: 实现 `Tensor` 类。难点：处理 `matmul` 的反向传播（注意转置和形状匹配）。
    - **Check**: 你的 `tensor.grad` 必须和 PyTorch 的结果完全一致（`torch.allclose`）。
- [ ]  **Day 4: 神经网络层 (Layers)**
    - **Code**: 基于 Day 3 的 Tensor，实现 `Linear`, `ReLU`, `Sequential`。
    - **Task**: 实现 SGD 优化器 (`step`, `zero_grad`)。
    - **Output**: 跑通 MNIST 手写数字识别（Acc > 90%）。
- [ ]  **Day 5: 卷积与池化 (Hard Mode)**
    - **Theory**: 理解 Convolution 是矩阵乘法的稀疏形式 (Toeplitz Matrix)。
    - **Code**: 手写 `im2col` (Image to Column) 实现高效卷积。
    - **Output**: 在你的引擎中添加 `Conv2d` 层。
- [ ]  **Day 6: 初始化与归一化 (Init & Norm)**
    - **Theory**: 为什么需要 Xavier/Kaiming 初始化？BatchNorm 的 $\gamma, \beta$ 怎么求导？
    - **Code**: 手写 `BatchNorm1d` 的 forward 和 backward。
    - **Check**: 观察加了 BN 后 Loss 下降速度的变化。
- [ ]  **Day 7: Week 1 复盘 (Code Review)**
    - **Self-Review**: 对比你的 Autograd 和 PyTorch 源码 (C++层面不用看，看逻辑)。
    - **Output**: 整理一篇笔记 "PyTorch Autograd 的 5 个设计细节"。

---

## Week 2: 序列建模与 Attention (NLP Foundation)

目标：手搓 Transformer 组件，为 Stage 2 做铺垫。

- [ ]  **Day 8: Word2Vec & Embedding**
    - **Theory**: Skip-gram vs CBOW。
    - **Code**: 用 `torch.einsum` 实现 Skip-gram 的负采样 Loss。
    - **Check**: 训练后，`King - Man + Woman` 真的等于 `Queen` 吗？
- [ ]  **Day 9: RNN/LSTM 及其反向传播**
    - **Theory**: BPTT (Backprop Through Time) 的梯度消失/爆炸问题。
    - **Code**: 手写一个单层 RNN 训练字符级语言模型 (Char-RNN)。
    - **Check**: 梯度截断 (Gradient Clipping) 的代码实现。
- [ ]  **Day 10: Attention Is All You Need (Part 1)**
    - **Reading**: 原论文精读。理解 Query, Key, Value 的物理含义（数据库检索视角）。
    - **Math**: 为什么除以 $sqrt{d_k}$？（推导方差变化）。
    - **Code**: 实现 `ScaledDotProductAttention`。
- [ ]  **Day 11: Multi-Head Attention (MHA)**
    - **Code**: 实现 MHA。注意 `transpose` 和 `view` 的操作顺序（Day 2 笔记的内容！）。
    - **Task**: 实现 Causal Mask (用于 GPT 解码)。
- [ ]  **Day 12: Transformer Block**
    - **Code**: 拼装 `MHA`, `LayerNorm`, `FFN`。实现 Residual Connection。
    - **Task**: 理解 Pre-Norm vs Post-Norm 的区别（DeepSeek/Llama 全是用 Pre-Norm，为什么？）。
- [ ]  **Day 13: Positional Encoding (PE)**
    - **Code**: 实现 Sinusoidal PE (绝对位置编码)。
    - **Math**: 证明 Sinusoidal PE 具有相对位置性质（线性变换）。
    - **Preview**: 预习 RoPE (旋转位置编码)，这是现在的标配。
- [ ]  **Day 14: Week 2 复盘 (Mini-Project)**
    - **Project**: **NanoGPT (Part 1)**。
    - **Task**: 复现 Karpathy 的 `nanoGPT` 代码，在莎士比亚数据集上训练一个小型 Transformer。
    - **Goal**: Loss 降到 1.5 以下，能生成像样的伪莎士比亚戏剧。