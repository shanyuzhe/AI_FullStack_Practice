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
        # 第一维和第二维
        batch, Tx = word.shape[0:2]
        # word shape [max_word_length, batch, embedding_length]
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
        word = torch.transpose(word, 0, 1)
        # 吧one-hot编码转化为编号
        word_label = torch.argmax(word, 2)
        # output shape [batch]
        output = torch.ones(batch, device=word.device)
        # 把h、x初始化
        h = torch.zeros(batch, self.hidden_units, device=word.device)
        x = torch.zeros(batch, EMBEDDING_LENGTH, device=word.device)
        for i in range(Tx):
            next_h = self.tanh(self.linear_h(torch.cat((h, x),1)))
            # 想要一个具体的概率
            temp = self.linear_y(next_h)
            hat_y = F.softmax(temp)
            # 对batch中的每一个样本，取出模型在当前时间i上，对真实字符的预测概率
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
        batch, Tx = word.shape[0:2]
        first_letter = word.new_zeros(batch, 1)
        x = torch.cat((first_letter, word[:, 0:-1]), 1)
        hidden = torch.zeros(1, batch, self.hidden_units, device=word.device)
        emb = self.drop(self.encoder(x))
        output, hidden = self.rnn(emb, hidden)
        y = self.decoder(output.reshape(batch * Tx, -1))

        return y.reshape(batch, Tx, -1)

    @torch.no_grad()
    def language_model(self, word: torch.Tensor):
        batch, Tx = word.shape[0:2]
        hat_y = self.forward(word)
        hat_y = F.softmax(hat_y, 2)
        output = torch.ones(batch, device=word.device)
        for i in range(Tx):
            probs = hat_y[torch.arange(batch), i, word[:, i]]
            output *= probs

        return output