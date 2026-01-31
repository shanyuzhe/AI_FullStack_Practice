import os
import re

# 全量获取，从原文读取句子
def read_imdb(dir='data/aclImdb', split='pos', is_train=True):
    subdir = 'train' if is_train else 'test'
    dir = os.path.join(dir, subdir, split)
    lines = []
    for file in os.listdir(dir):
        # rb二进制读取 f自定义变量名，方便操作
        with open(os.path.join(dir, file), 'rb') as f:
            # utf-8 通用的标准规则
            line = f.read().decode('utf-8')
            lines.append(line)
    return lines

# lines = read_imdb()
# print(len(lines))
# print(lines[0])
# print(lines[1])
# print(lines[2])

# 采样读取，从原文中获取单词，有重复，带有权重，学习到哪里单词组合出现的频率更高
def read_imdb_words(dir='data/aclImdb', split='pos', is_train=True, n_files=200):
    subdir = 'train' if is_train else 'test'
    dir = os.path.join(dir, subdir, split)
    all_str = ''
    for file in os.listdir(dir):
        if n_files <= 0:
            break
        with open(os.path.join(dir, file), 'rb') as f:
            line = f.read().decode('utf-8')
            all_str += line
        n_files -= 1
        # 只保留空格 和 26个小些字母  \u0020空格 \u0061-\u0071 a-z
        words = re.sub(u'^\u0020\u0061-\u007a', '', all_str.lower()).split(' ')
        return words

# words = read_imdb_words()
# print(words)


# 从单词表里面读取单词（没有重复，短时间看到更多种类单词拼写）
def read_imdb_vocab(dir='data/aclImdb'):
    fn = os.path.join(dir, 'imdb.vocab')
    with open(fn, 'rb') as f:
        word = f.read().decode('utf-8').replace('\n', ' ')
        words = re.sub(u'[^\u0020\u0061-\u007a]', '', word.lower()).split(' ')
        # hello!! \n\n
        filtered_words = [w for w in words if len(w) > 0]
    return filtered_words

filtered_words = read_imdb_vocab()
print(len(filtered_words))