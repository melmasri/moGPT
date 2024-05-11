import os
import requests
import tiktoken
import numpy as np


DATA_URL = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
# download the tiny shakespeare dataset

# getting the files
input_path = os.path.join(os.path.dirname(__file__), 'input.txt')
if not os.path.exists(input_path):
    data_url = DATA_URL
    with open(input_path, 'w', encoding='utf-8') as f:
        f.write(requests.get(data_url).text)

# readiing the files
with open(input_path, 'r', encoding='utf-8') as f:
    data = f.read()
n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

print('Snapshot of train data:')
print(train_data[:100])

# encode with tiktoken gpt2
encoding = tiktoken.get_encoding("gpt2")
train_tokens = encoding.encode_ordinary(train_data)
val_tokens = encoding.encode_ordinary(val_data)

print(f"train has {len(train_tokens):,} tokens")
print(f"val has {len(val_tokens):,} tokens")

print('Snapshot of train tokens:')
print(train_tokens[:100])

# export to bin files
train_ids = np.array(train_tokens, dtype=np.uint16)
val_ids = np.array(val_tokens, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

print(f"# train has {len(train_ids):,} tokens, val has {len(val_ids):,} tokens")

# train.bin has 301,966 tokens
# val.bin has 36,059 tokens