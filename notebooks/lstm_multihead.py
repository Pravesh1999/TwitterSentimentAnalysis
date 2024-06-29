import torch
import torch.nn as nn
import torch.nn.functional as F
from settings import *

class LSTMMultiHeadAttention(nn.Module):
    def __init__(self, vocab_size, input_dim, hidden_dim, output_dim, num_layers, bidirectional , dropout,
                num_heads):
        super(LSTMMultiHeadAttention, self).__init__()
        self.embedding = nn.Embedding(vocab_size, input_dim)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, bidirectional=True, batch_first=True, dropout=dropout)
        
        # Attention Layer
        self.attention_heads = nn.ModuleList([
            nn.Linear(hidden_dim * 2, 1) for _ in range(num_heads)])
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim * 2 * num_heads, output_dim)
        
        # Attention weights
        self.last_attention_weights = None

    def attention_net(self, lstm_outputs):
        """
        Multi-Head Attention mechanism.
        """
        context_vectors = []
        attention_weights_list = []
        for head in self.attention_heads:
            attention_weights = torch.tanh(head(lstm_outputs))
            attention_weights = F.softmax(attention_weights, dim=1)
            context_vector = torch.sum(attention_weights * lstm_outputs, dim=1)
            context_vectors.append(context_vector)
            attention_weights_list.append(attention_weights)
        combined_context = torch.cat(context_vectors, dim=-1).squeeze(-1)
        self.last_attention_weights = attention_weights
        return combined_context, attention_weights_list
        
    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_dim).requires_grad_().to(DEVICE)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_dim).requires_grad_().to(DEVICE)
        
        x = self.embedding(x)
        
        # Forward propagate LSTM
        lstm_out, (hn, cn) = self.lstm(x, (h0.detach(),c0.detach()))
        
        # Attention layer
        combined_context, attention_weights = self.attention_net(lstm_out)
        
        # Fully connected layer (readout)
        out = self.fc(combined_context)
        
        return out