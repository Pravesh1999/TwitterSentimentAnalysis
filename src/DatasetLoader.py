import numpy as np
from tokenizers import BertWordPieceTokenizer
from transformers import BertTokenizer, BertModel
from transformers import DistilBertTokenizer, DistilBertModel
from transformers import AutoTokenizer
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import os
import random
import torch.nn as nn
import torch.optim as optim
import time
import torch.nn.functional as F
from src.settings import *

class LoadDataset(Dataset):
    
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer  # Save the tokenizer to the instance.
        assert len(texts) == len(labels), "Length of texts and labels must be the same"
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        encoded_text = self.tokenizer.encode_plus(
            self.texts[idx],
            add_special_tokens=True,
            max_length=MAX_SEQ_LEN,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        label = torch.tensor(self.labels[idx])
        
        return (encoded_text['input_ids'].squeeze(0), encoded_text['attention_mask'].squeeze(0)), label

