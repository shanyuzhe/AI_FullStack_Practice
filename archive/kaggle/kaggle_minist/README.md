# Kaggle MNIST Practice

这是一个用于用 PyTorch 练习 MNIST（手写数字）分类的示例仓库。包含数据加载、模型训练、评估与推理的简易示例，适合学习与实验。

目录概览
- `Softmax_Classifier.py`：Softmax/线性分类器示例训练脚本。
- `utils/Minist_loader.py`：数据加载器（注意命名中的拼写历史性问题）。
- `utils/verify_dataset.py`：数据集完整性与格式校验工具。

依赖（示例）
- Python 3.8+
- torch, torchvision, numpy, pandas

快速开始
1. 安装依赖（推荐使用虚拟环境）：
```bash
pip install -r requirements.txt
```
2. 运行训练/评估脚本：
```bash
python Softmax_Classifier.py
```

说明
- 本仓库为练习用途，代码以可读与教学为主；若用于生产或更大规模实验，请按需改进模型与数据处理流程。
- 建议将文件/变量名中的 `Minist` 统一为 `MNIST` 以避免混淆。
