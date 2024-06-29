import pandas as pd
import numpy as np
from tokenizers import BertWordPieceTokenizer
from transformers import BertTokenizer, BertModel
from transformers import AutoTokenizer
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from DatasetLoader import LoadDataset
import os
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from settings import *
import yaml
from settings import *
from utils import *


def split_data(data):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    texts = data['Cleaned Content'].values.tolist()
    labels = data[['angry','disappointed','happy']].values.tolist()
    set_seed(seed_value=42)
    train_texts, temp_texts, train_labels, temp_labels = train_test_split(texts, labels, test_size=0.3, random_state=42)
    valid_texts, test_texts, valid_labels, test_labels = train_test_split(temp_texts, temp_labels, test_size=0.5, random_state=42)
    
    train_texts = np.array(train_texts)
    train_labels = np.array(train_labels)
    valid_texts = np.array(valid_texts)
    valid_labels = np.array(valid_labels)
    test_texts = np.array(test_texts)
    test_labels = np.array(test_labels)
    
    np.save('train_texts.npy', train_texts)
    np.save('train_labels.npy', train_labels)
    np.save('valid_texts.npy', valid_texts)
    np.save('valid_labels.npy', valid_labels)
    np.save('test_texts.npy', test_texts)
    np.save('test_labels.npy', test_labels)
    
    
    
