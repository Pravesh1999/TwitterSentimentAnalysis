import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np


class LSTMTextClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout):
        """
        Initialize the LSTMTextClassifier model.

        Parameters:
        - embedding_dim: Dimension of the input embeddings.
        - hidden_dim: Dimension of the hidden state in the LSTM.
        - output_dim: Number of classes in the output layer.
        - n_layers: Number of layers in the LSTM.
        - bidirectional: If True, initializes a bidirectional LSTM.
        - dropout: Dropout rate for regularization.
        """
        super(LSTMTextClassifier, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # LSTM layer
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers,
                            bidirectional=bidirectional, dropout=dropout, batch_first=True)
        
        # Fully connected layer
        # If the LSTM is bidirectional, it doubles the output features since it
        # concatenates the hidden states from both directions
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Forward pass through the model.

        Parameters:
        - x: input_ids

        Returns:
        - The logits for each class.
        """
        # Embeddings
        text_embeddings = self.embedding(x)
        # LSTM
        lstm_out, (hidden,cell) = self.lstm(text_embeddings)
        
        # Concatenate the final forward and backward hidden states
        if self.lstm.bidirectional:
            hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        else:
            hidden = hidden[-1,:,:]
        
        # Apply dropout
        hidden = self.dropout(hidden)
        
        # Fully connected layer
        logits = self.fc(hidden)
        
        return logits