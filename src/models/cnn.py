import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from src.settings import *

class CNNModel(nn.Module):
    def __init__(self,
                 vocab_size,
                 embed_dim=256, 
                 filter_sizes=[3, 4, 5],
                 num_filters=[200, 250, 200],
                 num_classes=3,
                 dropout=0.5):
        """
        Initialize the CNNModel model.

        Parameters:
        - vocab_size: Size of the vocabulary.
        - embed_dim: Dimension of the input embeddings.
        - filter_sizes: The size of the filter in each layer of convolution.
        - num_filters: The number of filters to apply in each layer of convolution.
        - num_classes: Number of classes in the output layer.
        - dropout: Dropout rate for regularization.
        """
        super(CNNModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.conv1d_list = nn.ModuleList([
            nn.Conv1d(in_channels=embed_dim, out_channels=num_filters[i],
                     kernel_size=filter_sizes[i])
            for i in range(len(filter_sizes))
        ])
        self.fc1 = nn.Linear(np.sum(num_filters), 256)     
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, x):
        x_embed = self.embedding(x)
        x_reshaped = x_embed.permute(0, 2, 1)
        x_conv_list = [F.relu(conv1d(x_reshaped)) for conv1d in self.conv1d_list]
        x_pool_list = [F.max_pool1d(x_conv, kernel_size=x_conv.shape[2])
                       for x_conv in x_conv_list]
        x_fc = torch.cat([x_pool.squeeze(dim=2) for x_pool in x_pool_list], dim=1)
        fc1 = self.dropout(F.relu(self.fc1(x_fc)))
        logits = self.fc2(fc1)
        return logits
