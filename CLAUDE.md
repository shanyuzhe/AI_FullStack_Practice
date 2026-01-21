# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a PyTorch deep learning learning repository focused on building foundational understanding of LLMs and Transformers from scratch. The project follows a "hardcore mode" approach - reimplementing core components rather than relying on high-level abstractions. The goal is to reach industrial R&D level (DeepSeek/OpenAI entry-level competency).

**Language**: Chinese (Mandarin) - README and most documentation are in Chinese. Code comments are primarily in Chinese.

## Repository Structure

```
.
├── archive/                    # Learning progression & completed exercises
│   ├── 28days_challenges/      # PyTorch fundamentals (Tensor ops, einsum, etc.)
│   ├── Introductory learning/  # Linear models → MLP → Transformer → LLMs
│   ├── Let's_build_GPT/        # Karpathy-style GPT implementation
│   ├── kaggle/                 # Kaggle competition experiments
│   └── liu2_lecture/           # Course materials
├── data_pipeline/              # (Planned) Data cleaning & tokenization
│   ├── cleaning/
│   └── tokenization/
├── inference_engine/           # (Planned) Inference optimization
│   ├── kernels/
│   └── quantization/
├── model_lab/                  # (Planned) Model components & architectures
│   ├── 01_components/          # RoPE, SwiGLU, RMSNorm, GQA/MQA, etc.
│   └── 02_architectures/       # Llama, etc.
├── training_sys/               # (Planned) Distributed training & optimization
│   ├── distributed/            # DP, TP, PP, ZeRO
│   └── optimization/
├── utils/                      # Engineering utilities
│   ├── torch_playground.py     # Device mgmt, seeding, timing, tensor inspection
│   └── debug_checklist.md      # PyTorch debugging workflow
├── checkpoints/                # Model checkpoints
└── data/                       # Training data (e.g., MNIST)
```

## Development Setup

### Environment
- **Python**: 3.x with virtual environment in `.venv/`
- **Key Dependencies**: PyTorch 2.6.0+cu124, torchvision, numpy, pandas, matplotlib, google-genai
- **GPU**: CUDA 12.4 support enabled

### PYTHONPATH Configuration
The project uses a custom `utils` module that must be importable from any subdirectory.

**Critical**: The `.env` file in the root directory sets `PYTHONPATH=.` and VSCode settings (`.vscode/settings.json`) enforce this for all terminals and the Python language server. Do NOT run scripts without this configuration or imports will fail.

### Install Dependencies
```bash
# Activate virtual environment
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# Install requirements
pip install -r requirements.txt
```

## Common Commands

### Running Scripts
```bash
# Always run from project root to ensure PYTHONPATH is correct
python archive/Let\'s_build_GPT/let\'s_build_gpt.py
python archive/28days_challenges/day01_tensor_basics.py

# Use utils in any script
from utils.torch_playground import get_device, setup_seed, time_block, inspect
```

### Testing
No formal test suite currently exists. The project uses exploratory scripts in `archive/` for validation.

## Core Utilities (`utils/torch_playground.py`)

This module provides essential engineering infrastructure used across all experiments:

### Key Functions
- **`get_device()`**: Auto-select CUDA/CPU with GPU info printout
- **`setup_seed(seed=277527)`**: Fix all random seeds (torch, numpy, random, cudnn)
- **`time_block(label)`**: Context manager for timing code blocks
  ```python
  with time_block("Matrix Multiplication"):
      result = a @ b
  ```
- **`inspect(tensor, name)`**: Print tensor shape, dtype, device, grad status, and value (if scalar)

### Global Exports
- `DEVICE`: Pre-initialized device (automatically selected)
- Seed is auto-fixed on import for reproducibility

## Architecture Philosophy

### Learning Progression
The codebase follows a structured learning path (see README.md for full roadmap):

1. **Stage 1: Foundation** (Weeks 1-2)
   - Manual autograd implementation (micrograd → tensor autograd)
   - Matrix calculus, backprop from scratch
   - RNN/LSTM, Word2Vec, Attention mechanism

2. **Stage 2: Architecture & Systems** (Currently in progress)
   - Transformer components: RoPE, SwiGLU, RMSNorm, GQA/MQA
   - Distributed training: DDP, ZeRO, Megatron-LM
   - FlashAttention, Triton kernels

3. **Stage 3: Full-stack LLM** (Planned)
   - Pre-training & SFT pipelines
   - RLHF/DPO alignment
   - vLLM inference optimization

### Code Style
- **Emphasis on clarity over brevity**: Extensive comments explaining "why" not just "what"
- **From-scratch implementations**: Avoid high-level APIs when learning core concepts
- **Mathematical rigor**: Hand-derive gradients, prove properties (e.g., KL divergence non-negativity)
- **Engineering discipline**: Use timing, seeding, and inspection tools consistently

## Key Implementation Examples

### GPT Implementation (`archive/Let's_build_GPT/let's_build_gpt.py`)
- Character-level language model on Shakespeare text (`input.txt`)
- Multi-head self-attention with causal masking
- Uses `register_buffer` for efficient mask storage
- Hyperparameters: 32 batch size, 8 block size, 32 embedding dim, 4 heads

### Learning Checkpoints (`archive/`)
- `28days_challenges/`: Tensor fundamentals (broadcast, einsum, gather, mask operations)
- `Introductory learning/01_Linear_Models/`: Linear regression with MSE loss
- `Introductory learning/03_Transformer/`: Attention mechanism implementations

## Debugging Workflow

When encountering errors, follow the checklist in `utils/debug_checklist.md`:

1. **Shape Mismatch**: Check (Batch, Time, Channel) ordering, broadcasting behavior, `.contiguous()` before `.view()`
2. **Dtype Issues**: Input should be `float32`, labels should be `long` for CrossEntropyLoss
3. **Device Errors**: Ensure model, input, and labels are all on same device
4. **Gradient Issues**: Check `optimizer.zero_grad()`, loss is scalar, `model.train()`/`model.eval()` switching
5. **NaN/Inf**: Check learning rate, division by zero (add epsilon), log domain

## Common Pitfalls

1. **PYTHONPATH not set**: Will cause `ModuleNotFoundError: No module named 'utils'`
2. **Running from wrong directory**: Always execute from project root
3. **Forgetting `.to(device)`**: Model and tensors must be on same device (CUDA/CPU)
4. **Shape confusion**: Pay attention to Transformer dimension ordering (B, T, C) vs CNN (B, C, H, W)
5. **Mask indexing**: Use `.contiguous()` after `permute`/`transpose` before calling `.view()`

## Current State & Roadmap

**Current Focus**: Transitioning from foundational learning (`archive/`) to production-style implementations in:
- `model_lab/01_components/`: Llama components (RoPE, GQA, RMSNorm)
- `training_sys/`: Distributed training infrastructure

**Note**: The main directories (`data_pipeline/`, `model_lab/`, `training_sys/`, `inference_engine/`) are currently scaffolded but empty. Active development is in `archive/` which contains working implementations and learning exercises.

## Additional Notes

- **No CI/CD**: This is a learning repository, not production code
- **No versioning**: Checkpoints are saved locally in `checkpoints/`
- **Data**: Small datasets (TinyStories, Shakespeare) for quick iteration
- **Compute**: Designed for single-GPU development (uses `torch.cuda` not multi-GPU by default)
