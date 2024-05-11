import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch

from constants import DATASET, INPUT_DATA_FOLDER, BATCH_SIZE, BLOCK_SIZE, DEVICE_TYPE, DEVICE


def get_batch(split):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == 'train':
        data = np.memmap(os.path.join(INPUT_DATA_FOLDER, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(INPUT_DATA_FOLDER, 'val.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([torch.from_numpy((data[i:i+BLOCK_SIZE]).astype(np.int64)) for i in ix])
    print(x)
    y = torch.stack([torch.from_numpy((data[i+1:i+1+BLOCK_SIZE]).astype(np.int64)) for i in ix])
    if DEVICE_TYPE == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(DEVICE, non_blocking=True), y.pin_memory().to(DEVICE, non_blocking=True)
    else:
        x, y = x.to(DEVICE), y.to(DEVICE)
    return x, y