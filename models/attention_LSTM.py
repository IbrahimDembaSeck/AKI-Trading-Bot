import torch
import torch.nn as nn
from lstm import MultiInputLSTM  # Import the core LSTM model
import math


class Attention(nn.Module):
    def __init__(self, hidden_sz):
        super(Attention, self).__init__()
        self.W_b = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_b = nn.Parameter(torch.Tensor(hidden_sz))
        self.v_b = nn.Parameter(torch.Tensor(hidden_sz))
        self.init_weights()

    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.v_b.size(0))
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, hidden_seq):
        # Compute attention scores
        attn_weights = torch.tanh(torch.matmul(hidden_seq, self.W_b) + self.b_b)
        attn_weights = torch.matmul(attn_weights, self.v_b.unsqueeze(-1)).squeeze(-1)
        attn_weights = torch.softmax(attn_weights, dim=1)

        # Apply attention weights to hidden sequence
        attended_output = torch.sum(hidden_seq * attn_weights.unsqueeze(-1), dim=1)
        return attended_output


class Net(nn.Module):
    def __init__(self, input_sz, hidden_sz, num_aux_factors=3):
        super(Net, self).__init__()
        self.multi_input_lstm = MultiInputLSTM(input_sz, hidden_sz, num_aux_factors)
        self.attention = Attention(hidden_sz)
        self.fc = nn.Linear(hidden_sz, 1)  # Final output layer

    def forward(self, Y, aux_inputs):
        _, hidden_seq = self.multi_input_lstm(Y, aux_inputs)
        attended_output = self.attention(hidden_seq)
        output = torch.relu(self.fc(attended_output))
        return output
