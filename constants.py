import os


DEVICE = 'cpu'
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


