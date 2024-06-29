import torch
import torch.nn as nn
import torch.nn.functional as F
from settings import *

class LSTMAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, bidirectional ,dropout):
        super(LSTMAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, bidirectional=True, batch_first=True, dropout=dropout)
        
        # Attention Layer
        self.attention_layer1 = nn.Linear(hidden_dim * 2, 1)
        self.attention_layer2 = nn.Linear(hidden_dim * 2, 1)
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        
    def attention_net(self, lstm_outputs):
        """
        Attention mechanism.
        """
        attention_weights1 = torch.tanh(self.attention_layer1(lstm_outputs))
        attention_weights1 = F.softmax(attention_weights1, dim=1)
        attention_weights2 = torch.tanh(self.attention_layer2(lstm_outputs))
        attention_weights2 = F.softmax(attention_weights2, dim=2)
        weighted_attention_weights = (attention_weights1 + attention_weights2) / 2 
        context_vector = torch.sum(weighted_attention_weights * lstm_outputs, dim=1)
        return context_vector, weighted_attention_weights
        
    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_dim).requires_grad_().to(DEVICE)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_dim).requires_grad_().to(DEVICE)
        
        # Forward propagate LSTM
        lstm_out, (hn, cn) = self.lstm(x, (h0.detach(),c0.detach()))
        
        # Attention layer
        context_vector, attention_weights = self.attention_net(lstm_out)
        
        # Fully connected layer (readout)
        out = self.fc(context_vector)
        
        return out