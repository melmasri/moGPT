import os
import torch

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
DEVICE_TYPE = 'cuda' if 'cuda' in DEVICE else 'cpu'

DATASET = 'shakespeare'
INPUT_DATA_FOLDER = os.path.join('data', DATASET)

BATCH_SIZE = 8
BLOCK_SIZE = 4
VOCAB_SIZE = 50304  
N_LAYERS = 12
N_EMBED = 768
N_HEADS = 12
BIAS = False
DROP_OUT = 0.1


## GPT-2 Hyperparameters -- 117M parameters
## VOCAB_SIZE = 50257
## BLOCK_SIZE = 1024
## BATCH_SIZE = 12
## N_LAYERS = 12
## N_HEADS = 12
## N_EMBED = 768



#### -- wandb -- ####
WANDB_PROJECT = 'gpt2'
WANDB_LOG = False
WANDB_KEY = ''
WANDB_RUN_NAME = 'mogpt'