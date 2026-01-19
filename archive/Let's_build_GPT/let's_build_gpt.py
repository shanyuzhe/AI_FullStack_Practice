import os
import torch
import torch.nn as nn
from torch.nn import functional as F

# -----------------------------------------------------------------------------
# 1. 超参数设置 (Hyperparameters)
# -----------------------------------------------------------------------------
batch_size = 32      # 一次并行处理多少个序列
block_size = 8       # 上下文长度 (Time steps)
max_iters = 3000     # 训练步数
eval_interval = 300  # 评估间隔
learning_rate = 1e-3 # 学习率 (稍微调低一点，加了Attention后网络变深了)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embed = 32         # 嵌入维度 (每个token变成32个数字)

print(f"Using device: {device}")

# -----------------------------------------------------------------------------
# 2. 数据准备 (Data Preparation)
# -----------------------------------------------------------------------------
torch.manual_seed(1337)

script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, 'input.txt')

try:
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
except FileNotFoundError:
    print(f"Error: 找不到文件 {file_path}，使用测试数据。")
    text = "Hello world! This is a test dataset for attention mechanism." * 100

chars = sorted(list(set(text)))
vocab_size = len(chars)
print(f"Vocab size: {vocab_size}")

stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}
encode = lambda s : [stoi[c] for c in s]
decode = lambda l: "".join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# -----------------------------------------------------------------------------
# 3. 数据加载函数 (Data Loading)
# -----------------------------------------------------------------------------
def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,)) 
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+1+block_size] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train','val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# -----------------------------------------------------------------------------
# 4. 单头注意力模块 (Single Head Attention) - **新增部分**
# -----------------------------------------------------------------------------
class Head(nn.Module):
    """一个标准的自注意力头"""

    def __init__(self, head_size):
        super().__init__()
        # 定义 Q, K, V 的线性变换
        # bias=False 是为了简化，通常也会加 bias
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        
        # 定义掩码 (Tril)，用于不让模型看见“未来”
        # register_buffer 表示这不是一个需要训练的参数，但它是模型状态的一部分
        #定义一个下三角矩阵
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        # 1. 生成 Key 和 Query
        k = self.key(x)   # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)

        # 2. 计算注意力得分 (Scores)
        # 这里的 transpose(-2, -1) 是为了把 (B, T, head_size) 转成 (B, head_size, T) 方便矩阵乘法
        # * C**-0.5 是缩放因子 (Scaled Dot-Product Attention)
        # "q" 和 "k" 的点积 表示它们的相似度 即查询与键的匹配程度（注意力得分）
        wei = q @ k.transpose(-2, -1) * C**-0.5 # (B, T_q, T_k)
        #wei(1, 2, 3)表示第一批 序列第二个字的查询 对 第三个字的键的注意力得分
        #但是其实是在向未来看 所以要掩码
        

        # 3. 掩码 (Masking)
        # 将上三角区域（未来）设置为负无穷，softmax 后变为 0
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))

        # 4. 归一化 (Softmax)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
   
        #下三角矩阵的含义是累加变换
        
        #按照key的维度进行归一化 对于每一个query，所有key的权重和为1
        # Query (当前字) \ Key (历史字),
        # 我 (0),爱 (1),吃 (2),饭 (3),行总和
        #        1.0,0.0,0.0,0.0,1.0
        # 爱 (1),0.4,0.6,0.0,0.0,1.0
        # 吃 (2),0.1,0.2,0.7,0.0,1.0
        # 饭 (3),0.1,0.4,0.3,0.2,1.0


#这里面是注意力得分矩阵 就是它告诉你：
#为了读懂当前的这个字，我需要从前面已经看过的字里，分别提取多少信息？
#wei实际上就是一个加权累加变换 只是这里是用softmax实现的

# 参考：
# #矩阵运算技巧
# wei = torch.tril(torch.ones(T,T))#生成下三角矩阵 左乘的效果是按列累加和
# wei = wei / wei.sum(1,keepdim=True)#使得行和唯一 左乘效果变成按列累加均值
# xbow2 = wei @ x
# torch.allclose(xbow,xbow2)
# # 计算两个结果的最大差值
# diff = (xbow - xbow2).abs().max()
# print(f"最大误差: {diff.item()}")

#这里是求均值 但是注意力机制是加权均值
# 也就是说 每个词对当前词的贡献度不一样






        # 5. 聚合 Value（按照注意力得分加权求和）
        v = self.value(x) # (B, T, head_size)
        out = wei @ v     # (B, T, head_size)
        return out

# -----------------------------------------------------------------------------
# 5. 主模型定义 (Updated Model)
# -----------------------------------------------------------------------------
class BigramLanguageModel(nn.Module):
    #这里面定义的线性变换矩阵都是模型的参数，会被optimizer更新
    def __init__(self, vocab_size):
        super().__init__()
        # 1. Token Embedding: 把每个 ID 变成向量 (B, T, n_embed)
        # embed表格逻辑 是：有vocab_size个词，每个词用n_embed维向量表示
        # 送进来的Batch个序列，每个序列block_size个词
        # 最终变成 (B, T, n_embed) :每个词变成n_embed维向量表示
        
        # 你眼中的 词w对应的 向量(w)：[w_1, w_2, w_3, ... w_32]这 32 个 $w$ 都是独立的、活生生的参数。
        # 当你反向传播时，会有 32 条不同的链路（偏导数）分别去指挥这 32 个数字如何变化。
        
        
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        
        # 2. Position Embedding: 记住每个字在句子里的位置 (T, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        
        # 3. Self-Attention Head: **核心修改**
        # 这里创建一个注意力头，传入的参数是这个head要处理的向量维度
        #这里由于只有一个head，所以head_size=n_embed
        #如果有多个head 则head_size = n_embed // num_heads
        self.sa_head = Head(n_embed)
        
        # 4. Language Model Head: 也就是最后的线性层，把向量变回词表概率
        # 输入 n_embed，输出 vocab_size
        self.lm_head = nn.Linear(n_embed, vocab_size)
        
    def forward(self, idx, targets=None):
        B, T = idx.shape

        # --- Embedding 阶段 ---
        # 获取 token 嵌入
        tok_emb = self.token_embedding_table(idx) # (B, T, n_embed)
        # 获取 位置 嵌入 (0, 1, 2... T-1)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, n_embed)
        # 将两者相加，现在 x 既包含了“是什么字”，也包含了“在哪个位置”
        # 最开始pos_emb是随机初始化的，经过训练会学到位置编码
        
        x = tok_emb + pos_emb # (B, T, n_embed)
        
        # --- Attention 阶段 (新增) ---
        # 让 token 之间开始交流，x 经过这层后，融合了上下文信息
        x = self.sa_head(x) # (B, T, n_embed)
        
        # --- Decoding 阶段 ---
        # 映射回词汇表大小，得到 Logits
        logits = self.lm_head(x) # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices
        for _ in range(max_new_tokens):
            # --- 关键修改：截断上下文 ---
            # 因为我们有 position embedding，最大只能处理 block_size 这么长
            # 如果 idx 超过了 8 个字，我们就只取最后 8 个字喂给模型
            idx_cond = idx[:, -block_size:] 
            
            # 这里的输入变成了 idx_cond
            logits, loss = self(idx_cond)
            
            # 后面逻辑不变
            logits = logits[:, -1, :] 
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            
        return idx

# -----------------------------------------------------------------------------
# 6. 训练主循环
# -----------------------------------------------------------------------------
# **注意**：实例化时必须传入 vocab_size
model = BigramLanguageModel(vocab_size)
m = model.to(device)

optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

print(f"开始训练 (max_iters={max_iters})...")

for iter in range(max_iters):

    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch("train")

    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
# 模型里定义的都是参数，一起训练 比如token_embedding_table,position_embedding_table,sa_head,lm_head    
# 前馈：7000 个参数像齿轮一样咬合在一起，算出结果。
# 反馈：算出每个齿轮对错误的“贡献度”（梯度）。
# 优化：把这 7000 个齿轮，同时朝着各自正确的方向，微调一点点。
# -----------------------------------------------------------------------------
# 7. 生成测试
# -----------------------------------------------------------------------------
print("\n训练完成，生成示例文本：")
context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated_indices = m.generate(idx=context, max_new_tokens=500)
print(decode(generated_indices[0].tolist())  )  