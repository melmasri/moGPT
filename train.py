import os
import time
import math
import pickle
from contextlib import nullcontext
from torch.nn import functional as F
import wandb

from constants import WANDB_PROJECT, WANDB_LOG,WANDB_KEY, WANDB_RUN_NAME


import numpy as np
import torch

from constants import DATASET, INPUT_DATA_FOLDER, BATCH_SIZE, BLOCK_SIZE, DEVICE_TYPE, DEVICE, VOCAB_SIZE


def wandb_init(model, run_name=None):
    if WANDB_LOG and WANDB_KEY:
        print("Logging to wandb with key")
        if run_name is None:
            run_name = WANDB_RUN_NAME 
        wandb.login(key=WANDB_KEY)
        wandb.init(project=WANDB_PROJECT, name=run_name, entity='elmasri-m', reinit=True)
        wandb.config.update({
            'DATASET': DATASET,
            'BATCH_SIZE': BATCH_SIZE,
            'BLOCK_SIZE': BLOCK_SIZE,
            'VOCAB_SIZE': VOCAB_SIZE,
            'DEVICE_TYPE': DEVICE_TYPE,
            'DEVICE': DEVICE,
        })
        wandb.watch(model)
        

def get_batch(split, config=None):
    block_size = BLOCK_SIZE
    batch_size = BATCH_SIZE
    if config:
        block_size = config.get('block_size', BLOCK_SIZE)
        batch_size = config.get('batch_size', BATCH_SIZE)
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == 'train':
        data = np.memmap(os.path.join(INPUT_DATA_FOLDER, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(INPUT_DATA_FOLDER, 'val.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if DEVICE_TYPE == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(DEVICE, non_blocking=True), y.pin_memory().to(DEVICE, non_blocking=True)
    else:
        x, y = x.to(DEVICE), y.to(DEVICE)
    return x, y

@torch.no_grad()
def evaluate(model):
    model.eval()
    eval_iter = 100
    metrics = {}
    config = model.get_config()
    mode = ['train', 'eval']
    for split in mode:
        total_loss = 0.
        for i in range(eval_iter):
            data, targets = get_batch(split, config)
            logits = model(data)
            B, T, C = logits.shape
            targets = targets.view(B*T)
            logits = logits.view(B*T, C)
            total_loss += F.cross_entropy(logits, targets).item()
        metrics[split] = total_loss / eval_iter
    return metrics

def train(model, optimizer, num_epochs, run_name=None):
    wandb_init(model, run_name=run_name)
    eval_interval = 100
    iterations = 1000
    step = 0
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.
        start_time = time.time()
        for batch, i in enumerate(range(0, iterations, BATCH_SIZE)):
            step += 1
            data, targets = get_batch('train',  model.get_config())
            optimizer.zero_grad()
            logits = model(data)
            B, T, C = logits.shape
            targets = targets.view(B*T)
            logits = logits.view(B*T, C)
            loss = F.cross_entropy(logits, targets)
            loss.backward()
            optimizer.step()
            # logging evaluations
            if batch % eval_interval == 0:
                metrics = evaluate(model)
                if WANDB_LOG:
                    wandb.log({'train/loss': metrics['train'],
                               'eval/loss': metrics['eval'],
                               'step': step,
                               })
            total_loss += loss.item()
            if batch % 10 == 0 and batch > 0:
                cur_loss = total_loss / 10
                elapsed = time.time() - start_time
                print(f"{epoch=}, {batch=}, {elapsed=},{cur_loss=:0.3f}")
                total_loss = 0
                start_time = time.time()
