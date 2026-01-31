import os
import re

# 全量获取，从原文读取句子
#相对路径 我的工作区就是在AI_FULLSTACK，这个是从工作区往下找
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
        # 解释正则表达式：[^\u0020\u0061-\u007a]
        # 这个正则的意思是：匹配所有“不是空格(Unicode码0020)和不是小写字母a-z(Unicode码0061-007a)”的字符。
        # [^ ] 表示取反，\u0020是空格，\u0061-\u007a是a到z的小写字母区间。
        # 所以这个表达式的作用是：保留所有空格和a-z小写字母，去除其它字符（包括标点符号、大写字母、数字和其它非英语字符）。
        words = re.sub(u'[^\u0020\u0061-\u007a]', '', word.lower()).split(' ')
        # hello!! \n\n
        filtered_words = [w for w in words if len(w) > 0]
    return filtered_words

filtered_words = read_imdb_vocab()
print(len(filtered_words))