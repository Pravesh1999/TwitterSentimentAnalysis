import torch
import torch.nn as nn
import torch.nn.functional as F
from src.settings import *

class LSTMMultiHeadAttention(nn.Module):
    def __init__(self, vocab_size, input_dim, hidden_dim, output_dim, num_layers, bidirectional , dropout,
                num_heads):
        """
        Initialize the LSTMMultiHeadAttention model.

        Parameters:
        - vocab_size: Size of the vocabulary.
        - embedding_dim: Dimension of the input embeddings.
        - hidden_dim: Dimension of the hidden state in the LSTM.
        - output_dim: Number of classes in the output layer.
        - n_layers: Number of layers in the LSTM.
        - bidirectional: If True, initializes a bidirectional LSTM.
        - dropout: Dropout rate for regularization.
        """
        super(LSTMMultiHeadAttention, self).__init__()
        self.embedding = nn.Embedding(vocab_size, input_dim)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.bidirectional = bidirectional
        
        # LSTM
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, bidirectional=self.bidirectional, 
                            batch_first=True, dropout=dropout)
        
        # Attention Layer
        if self.bidirectional:
            self.head_dim = hidden_dim * 2 // num_heads
        else:
            self.head_dim = hidden_dim // num_heads
        assert self.hidden_dim % num_heads == 0, "hidden_dim must be divisible by the number of heads"
        if self.bidirectional:
            # Q, K, V layers are initialised
            self.query_layers = nn.ModuleList([nn.Linear(hidden_dim * 2, self.head_dim) for _ in range(self.num_heads)])
            self.key_layers = nn.ModuleList([nn.Linear(hidden_dim * 2, self.head_dim) for _ in range(self.num_heads)])
            self.value_layers = nn.ModuleList([nn.Linear(hidden_dim * 2, self.head_dim) for _ in range(self.num_heads)])
        elif not self.bidirectional:
            self.query_layers = nn.ModuleList([nn.Linear(hidden_dim, self.head_dim) for _ in range(self.num_heads)])
            self.key_layers = nn.ModuleList([nn.Linear(hidden_dim, self.head_dim) for _ in range(self.num_heads)])
            self.value_layers = nn.ModuleList([nn.Linear(hidden_dim, self.head_dim) for _ in range(self.num_heads)])
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        
        # Attention weights
        self.last_attention_weights = None
        self.dropout = nn.Dropout(p=dropout)

    def attention_net(self, lstm_outputs):
        """
        Multi-Head Attention mechanism
        """
        attention_outputs = []
        attention_scores = []
        for i in range(self.num_heads):
            Q = self.query_layers[i](lstm_outputs)
            Q = self.dropout(Q)
            K = self.key_layers[i](lstm_outputs)
            K = self.dropout(K)
            V = self.value_layers[i](lstm_outputs)
            V = self.dropout(V)
            scores = torch.bmm(Q, K.transpose(1, 2)) / (self.head_dim ** 0.5)
            scores = F.softmax(scores, dim=-1)
            output = torch.bmm(scores, V)
            attention_scores.append(scores)
            attention_outputs.append(output)
        final_output = torch.cat(attention_outputs, dim=-1)
        self.last_attention_weights = attention_scores
        return final_output, attention_scores
            
        
    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_dim).requires_grad_().to(DEVICE)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_dim).requires_grad_().to(DEVICE)
        
        x = self.embedding(x)
        
        # Forward propagate LSTM
        lstm_out, (hn, cn) = self.lstm(x, (h0.detach(),c0.detach()))
        
        # Attention layer
        attention_output, attention_weights = self.attention_net(lstm_out)
        # Pool over sequence
        final_attention_output = torch.mean(attention_output, dim=1)
        # Fully connected layer
        # print(attention_output.shape)
        out = self.fc(final_attention_output)
        
        return out