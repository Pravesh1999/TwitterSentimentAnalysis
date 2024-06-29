import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np


class MLPClassifier(nn.Module):
    def __init__(self, vocab_size, input_dim, hidden_size, num_classes=2, dropout=0.5):
        """
        Initialize the MLPClassifier model.

        Parameters:
        - vocab_size: Size of the vocabulary.
        - input_dim: Dimension of the input embeddings.
        - hidden_dim: Dimension of the hidden state in the LSTM.
        - num_classes: Number of classes in the output layer.
        - dropout: Dropout rate for regularization.
        """
        super(MLPClassifier, self).__init__()
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, input_dim)
        self.fc1 = nn.Linear(input_dim * 120, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, hidden_size // 4)
        self.fc4 = nn.Linear(hidden_size // 4, num_classes)
        # Dropout layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        embeddings = self.embedding(x)
        # Get embeddings flattened
        x = embeddings.view(embeddings.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.dropout(x)
        # final output layer
        x = self.fc4(x)
        return x
