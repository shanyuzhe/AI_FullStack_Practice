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