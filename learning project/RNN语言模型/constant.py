import torch
# 26个字母和空格
EMBEDDING_LENGTH = 27
# 正向编码
LETTER_MAP = {' ': 0}
# 反向编码
ENCODING_MAP = [' ']

for i in range(26):
    LETTER_MAP[chr(ord('a') + i)] = i + 1
    ENCODING_MAP.append(chr(ord('a') + i))

# print(LETTER_MAP)
# print(ENCODING_MAP)
LETTER_LIST = list(LETTER_MAP.keys())
# print(LETTER_LIST)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"Using device: {DEVICE}")