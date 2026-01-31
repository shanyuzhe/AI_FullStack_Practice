from typing import Sequence, Tuple
import warnings
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from constant import EMBEDDING_LENGTH, LETTER_MAP
from read_imdb import read_imdb_vocab, read_imdb_words
from models import RNN1, RNN2

# 把字符串形式的单词列表，转换成神经网络可以直接训练的张量数据集
class WordDataset(Dataset):
    def __init__(self, words, max_length, is_onehot=True):
        super().__init__()
        n_words = len(words)
        self.n_words = n_words
        self.words = words
        self.max_length = max_length
        self.is_onehot = is_onehot

    def __len__(self):
        return self.n_words

    # 把一个单词字符串，转换成固定长度的字符编码张量
    def __getitem__(self, index):
        word = self.words[index] + ' '
        word_length = len(word)
        # 如果是one-hot方法
        if self.is_onehot:
            tensor = torch.zeros(self.max_length, EMBEDDING_LENGTH)
            # max_length 单词的最大长度
            for i in range(self.max_length):
                if i < word_length:
                    tensor[i][LETTER_MAP[word[i]]] = 1
                else:
                    tensor[i][0] = 1
        # 如果是普通标签
        else:
            tensor = torch.zeros(self.max_length, dtype=torch.long)
            for i in range(word_length):
                tensor[i] = LETTER_MAP[word[i]]

        return tensor

# words = ['the', 'apple']
# dataset = WordDataset(words, 20, True)
# print(dataset[0])

# 构建PyTorch Dataset 和 Dataloader
# 返回能够直接用于训练的数据加载器和实际使用的最大序列长度

def get_dataloader_and_max_length(limit_length=None, is_onehot=True, is_vocab=True):
    if is_vocab:
        words = read_imdb_vocab()
    else:
        words = read_imdb_words(n_files=200)

    max_length = 0
    # 获取最大长度
    for word in words:
        max_length = max(max_length, len(word))
    if limit_length is not None and max_length > limit_length:
        words = [w for w in words if len(w) <= limit_length]
        max_length = limit_length
    max_length += 1
    # WordDataset类实例化
    dataset = WordDataset(words, max_length, is_onehot=is_onehot)
    return DataLoader(dataset, batch_size=256), max_length

# dataloader, max_length = get_dataloader_and_max_length(19)
# print(max_length)

def train_rnn1():
    device = 'cpu'
    dataloader, max_length = get_dataloader_and_max_length(19)
    model = RNN1().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    for epoch in range(5):
        loss_sum = 0
        dataset_len = len(dataloader.dataset)
        for y in dataloader:
            y = y.to(device)
            hat_y = model(y)
            n, Tx, _ = hat_y.shape
            # hat_y y [batch, max_word_length, EMBEDDING_LENGTH] [batch * max_word_length, EMBEDDING_LENGTH]
            # 为了符合损失函数的结构，我们把前两维合并，
            hat_y = torch.reshape(hat_y, (n * Tx, -1))
            y = torch.reshape(y, (n * Tx, -1))
            # one-hot [0, 0, 1, 0, 0] -> 类别2
            label_y = torch.argmax(y, dim=1)
            loss = criterion(hat_y, label_y)
            optimizer.zero_grad()
            loss.backward()
            # 截取梯度的最大值，防止RNN梯度过大
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            loss_sum += loss
        print(f'Epoch {epoch}. Loss: {loss_sum / dataset_len}')
    torch.save(model.state_dict(), 'rnn1.pth')
    return model

Test_words = [
    'apple', 'appll', 'appla', 'apply', 'bear', 'beer', 'berr', 'beee', 'car',
    'cae', 'cat', 'cac', 'caq', 'query', 'queee', 'queue', 'queen', 'quest',
    'quess', 'quees'
]

# 把字符转换成整数索引 根据LETTER_MAP转换成对应的数字
def words_to_label_array(words:Tuple[str, Sequence[str]], max_length):
    # 无论是字符还是列表，都把它转化成列表
    if isinstance(words, str):
        words = [words]
    words = [word + ' ' for word in words]
    batch = len(words)
    tensor = torch.zeros(batch, max_length, dtype=torch.long)
    for i in range(batch):
        for j, letter in enumerate(words[i]):
            tensor[i][j] = LETTER_MAP[letter]
    return tensor

# test_words = words_to_label_array(Test_words, 20)

def words_to_one_hot(words:Tuple[str, Sequence[str]], max_length):
    if isinstance(words, str):
        words = [words]
    words = [word + ' ' for word in words]
    batch = len(words)
    tensor = torch.zeros(batch, max_length, EMBEDDING_LENGTH)
    for i in range(batch):
        word_length = len(words[i])
        for j in range(max_length):
            if j < word_length:
                tensor[i][j][LETTER_MAP[words[i][j]]] = 1
            else:
                tensor[i][j][0] = 0
    return tensor

# test_word = words_to_one_hot(Test_words, 20)

def test_language_model(model, is_onehot=True, device='cpu'):
    _, max_length = get_dataloader_and_max_length(19)
    if is_onehot:
        test_word = words_to_one_hot(Test_words, max_length)
    else:
        test_word = words_to_label_array(Test_words, max_length)
    test_word = test_word.to(device)
    probs = model.language_model(test_word)
    for word, prob in zip(Test_words, probs):
        print(f'{word}:{prob}')

def train_rnn2():
    device = 'cpu'
    dataloader, max_length = get_dataloader_and_max_length(19, is_onehot=False)

    model = RNN2().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    citerion = torch.nn.CrossEntropyLoss()
    for epoch in range(5):

        loss_sum = 0
        dataset_len = len(dataloader.dataset)

        for y in dataloader:
            y = y.to(device)
            hat_y = model(y)
            n, Tx, _ = hat_y.shape
            hat_y = torch.reshape(hat_y, (n * Tx, -1))
            label_y = torch.reshape(y, (n * Tx, ))
            loss = citerion(hat_y, label_y)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            loss_sum += loss

        print(f'Epoch {epoch}. loss: {loss_sum / dataset_len}')

    torch.save(model.state_dict(), 'rnn2.pth')
    return model



def rnn1():
    # rnn1 = train_rnn1()
    state_dict = torch.load('checkpoints/rnn1.pth', map_location='cpu')
    rnn1 = RNN1().to('cpu')
    rnn1.load_state_dict(state_dict)
    # 切换到评估模式
    rnn1.eval()
    test_language_model(rnn1)

def rnn2():
    # rnn2 = train_rnn2()

    # Or load the models
    state_dict = torch.load('checkpoints/rnn2.pth', map_location='cpu')
    rnn2 = RNN2().to('cpu')
    rnn2.load_state_dict(state_dict)

    rnn2.eval()
    test_language_model(rnn2, False)



if __name__ == '__main__':
    # rnn1()
    rnn2()