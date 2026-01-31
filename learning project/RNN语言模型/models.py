import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from constant import EMBEDDING_LENGTH, LETTER_LIST, LETTER_MAP

# 只用线性层、自动求导机制来从头实现一个RNN
class RNN1(nn.Module):
    def __init__(self, hidden_units=32):
        super().__init__()
        self.hidden_units = hidden_units
        # 输入 x EMBEDDING_LENGTH h hidden_units
        self.linear_h = nn.Linear(EMBEDDING_LENGTH + hidden_units, hidden_units)
        # 输出部分 输出的是字母的概率
        self.linear_y = nn.Linear(hidden_units, EMBEDDING_LENGTH)
        # 把数值限制在[-1. 1]防止梯度爆炸
        self.tanh = nn.Tanh()

    # 当一个单词进入模型，具体怎么处理
    def forward(self, word:torch.Tensor):
        # word shape [batch, max_word_length, embedding_length] (256, 20, 27)
        batch, Tx = word.shape[0:2]
        # 转换为 [max_word_length, batch, embedding_length]，便于循环
        word = torch.transpose(word, 0, 1)
        output = torch.empty_like(word)
        h = torch.zeros(batch, self.hidden_units, device=word.device)
        x = torch.zeros(batch, EMBEDDING_LENGTH, device=word.device)
        for i in range(Tx):
            next_h = self.tanh(self.linear_h(torch.cat((h, x),1)))
            hat_y = self.linear_y(next_h)
            output[i] = hat_y
            x = word[i]
            h = next_h
        return torch.transpose(output, 0, 1)

    @torch.no_grad() # 不需要推理，不需要计算梯度
    def language_model(self, word:torch.Tensor):
        batch, Tx = word.shape[0:2]
        # 转换为 [max_word_length, batch, embedding_length]
        word = torch.transpose(word, 0, 1)
        # one-hot 转为编号
        word_label = torch.argmax(word, 2)
        # output shape [batch]
        output = torch.ones(batch, device=word.device)
        # 初始化h、x
        h = torch.zeros(batch, self.hidden_units, device=word.device)
        x = torch.zeros(batch, EMBEDDING_LENGTH, device=word.device)
        for i in range(Tx):
            next_h = self.tanh(self.linear_h(torch.cat((h, x),1)))
            # 想要一个具体的概率
            temp = self.linear_y(next_h)
            # tmp的形状是？ [batch, EMBEDDING_LENGTH]
            tmp = temp
            # 注意 经过softmax不会降维 而是把每个样本的每个类别概率分布都计算出来【0~1】
            # hat_y形状：[batch, EMBEDDING_LENGTH]，即每个样本在当前时刻的所有类别概率分布
            hat_y = F.softmax(tmp, dim=1)
            # 查一下batch中的每一个样本，取出模型在当前时间i上，对真实字符的预测概率
            probs = hat_y[torch.arange(batch), word_label[i]]
            output *= probs
            x = word[i]
            h = next_h
        return output

# GRU RNN2
class RNN2(torch.nn.Module):

    def __init__(self, hidden_units=64, embeding_dim=64, dropout_rate=0.2):
        super().__init__()
        self.drop = nn.Dropout(dropout_rate)
        self.encoder = nn.Embedding(EMBEDDING_LENGTH, embeding_dim)
        self.rnn = nn.GRU(embeding_dim, hidden_units, 1, batch_first=True)
        self.decoder = torch.nn.Linear(hidden_units, EMBEDDING_LENGTH)
        self.hidden_units = hidden_units

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.bias)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, word: torch.Tensor):
        # word shape: [batch, max_word_length]
        batch, Tx = word.shape[0:2]  # batch: 批量样本数量（多少个单词/序列）；Tx: 每个单词/序列的最大长度（时间步数）

        # 这里 first_letter 用来给每个序列的第一个输入位置补0，实现错位
        first_letter = word.new_zeros(batch, 1)
        # x 是将 first_letter 拼在原序列前面，并去掉原序列最后一个字符，实现“错位输入”
        x = torch.cat((first_letter, word[:, 0:-1]), 1)

        # 初始化 RNN 的隐藏状态: 1（单层）× batch × 隐层维度
        hidden = torch.zeros(1, batch, self.hidden_units, device=word.device)

        emb = self.drop(self.encoder(x))          # 先查embedding再dropout
        # output: (batch, Tx, hidden_units)；hidden: (1, batch, hidden_units)
        output, hidden = self.rnn(emb, hidden)    
        # 输入两个参数 emb和hidden，返回output和hidden emb是错位的输入 防止偷看
        # emb: (batch, Tx, embeding_dim)
        # hidden: (1, batch, hidden_units)
        # output: (batch, Tx, hidden_units)
        # hidden: (1, batch, hidden_units)
        # output: 所有时间步输出（全程记忆）, hidden: 最后时刻隐藏状态

        # 每个时间步都把output展平，方便decoder处理
        # 展平后丢到decoder中，生成每个字符/时间步的分类分布
        y = self.decoder(output.reshape(batch * Tx, -1))

        return y.reshape(batch, Tx, -1)

    @torch.no_grad()
    def language_model(self, word: torch.Tensor):
        batch, Tx = word.shape[0:2]
        hat_y = self.forward(word)
        hat_y = F.softmax(hat_y, 2)
        output = torch.ones(batch, device=word.device)
        for i in range(Tx):
            # 取出第i个时间步对应的预测概率
            # hat_y的形状是 (batch, Tx, EMBEDDING_LENGTH)，
            # word[:, i] 是每个样本在第i个时间步的真实词ID（shape: [batch]）
            # hat_y[torch.arange(batch), i, word[:, i]] 相当于批量地，
            # 从每个样本的第i步输出概率分布里，取出真实单词/字符对应的概率
            # 得到一个形状为[batch]的向量
    
            probs = hat_y[torch.arange(batch), i, word[:, i]]
            # 合起来就是：
            # 取出 (第0个样本, 第0时刻, 第0个字母A的概率)
            # 取出 (第1个样本, 第0时刻, 第0个字母C的概率)
            # 取出 (第2个样本, 第0时刻, 第0个字母B的概率)
            output *= probs

        return output