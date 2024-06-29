import torch

if torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

EMBEDDINGS_DIM = 256
MAX_SEQ_LEN = 120
MODELS_BASE_DIR = '../models/'
PLOTTING_BASE_DIR = '../plots/'
NUM_EPOCHS = 10
BATCH_SIZE = 128

PREPROCESSED_DATA_PATH = '../data/processed'
CLEANED_FILE_NAME = 'Cleaned Tweets.csv'