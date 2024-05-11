import os


DEVICE = 'cpu'
DEVICE_TYPE = 'cuda' if 'cuda' in DEVICE else 'cpu'

DATASET = 'shakespeare'
INPUT_DATA_FOLDER = os.path.join('data', DATASET)

BATCH_SIZE = 8
BLOCK_SIZE = 4