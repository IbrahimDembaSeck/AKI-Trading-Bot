import torch
import torch.nn as nn
import math


class MultiInputLSTM(nn.Module):
    def __init__(self, input_sz: int, hidden_sz: int, num_aux_factors=3):
        super(MultiInputLSTM, self).__init__()

        # Parameters
        self.hidden_size = hidden_sz
        self.num_aux_factors = num_aux_factors

        # Mainstream (Y) input gates
        self.W_i = nn.Parameter(torch.Tensor(input_sz, hidden_sz))
        self.U_i = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_i = nn.Parameter(torch.Tensor(hidden_sz))

        self.W_f = nn.Parameter(torch.Tensor(input_sz, hidden_sz))
        self.U_f = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_f = nn.Parameter(torch.Tensor(hidden_sz))

        self.W_c = nn.Parameter(torch.Tensor(input_sz, hidden_sz))
        self.U_c = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_c = nn.Parameter(torch.Tensor(hidden_sz))

        self.W_o = nn.Parameter(torch.Tensor(input_sz, hidden_sz))
        self.U_o = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_o = nn.Parameter(torch.Tensor(hidden_sz))

        # Auxiliary (P and N) input gates
        self.aux_gates = nn.ModuleList([
            nn.Linear(input_sz, hidden_sz) for _ in range(num_aux_factors)
        ])

        # Attention weights for combining cell states of main and auxiliary inputs
        self.W_a = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_a = nn.Parameter(torch.Tensor(hidden_sz))

        # Initialize weights
        self.init_weights()

    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, Y, aux_inputs):
        """
        Y: Mainstream input sequence (batch, seq_len, input_sz)
        aux_inputs: List of auxiliary input sequences [(batch, seq_len, input_sz), ...] for each auxiliary factor
        """
        bs, seq_len, _ = Y.size()

        # Initialize hidden and cell states
        h_t, c_t = (torch.zeros(bs, self.hidden_size).to(Y.device),
                    torch.zeros(bs, self.hidden_size).to(Y.device))

        hidden_seq = []

        # Iterate through each timestep
        for t in range(seq_len):
            Y_t = Y[:, t, :]

            # Mainstream LSTM gates
            i_t = torch.sigmoid(Y_t @ self.W_i + h_t @ self.U_i + self.b_i)
            f_t = torch.sigmoid(Y_t @ self.W_f + h_t @ self.U_f + self.b_f)
            C_tilde_t = torch.tanh(Y_t @ self.W_c + h_t @ self.U_c + self.b_c)
            o_t = torch.sigmoid(Y_t @ self.W_o + h_t @ self.U_o + self.b_o)

            # Auxiliary inputs processing
            aux_outputs = []
            for i, aux_input in enumerate(aux_inputs):
                aux_t = aux_input[:, t, :]
                aux_gate = self.aux_gates[i]
                aux_out = torch.tanh(aux_gate(aux_t))
                aux_outputs.append(aux_out)

            # Calculate weighted sum of cell states using attention
            aux_outputs = torch.stack(aux_outputs, dim=0)  # Shape: (num_aux_factors, batch, hidden_size)
            combined_states = torch.cat((C_tilde_t.unsqueeze(0), aux_outputs), dim=0)
            weights = torch.softmax(torch.matmul(combined_states, self.W_a) + self.b_a, dim=0)
            L_t = torch.sum(weights * combined_states, dim=0)

            # Cell and hidden state updates
            c_t = f_t * c_t + i_t * L_t
            h_t = o_t * torch.tanh(c_t)

            hidden_seq.append(h_t.unsqueeze(0))

        # Stack the hidden states for all timesteps
        hidden_seq = torch.cat(hidden_seq, dim=0).transpose(0, 1)  # Shape: (batch, seq_len, hidden_size)
        return h_t, hidden_seq  # Return the final hidden state and sequence of hidden states
