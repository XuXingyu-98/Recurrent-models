import torch
import torch.nn as nn
from LSTM_cell import LSTMCell
from Basic_RNN_cell import BasicRNNCell
from GRU_cell import GRUCell


class RNNModel(nn.Module):
    def __init__(self, mode, input_size, hidden_size, num_layers, bias, output_size):
        super(RNNModel, self).__init__()
        self.mode = mode
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.output_size = output_size

        self.rnn_cell_list = nn.ModuleList()

        if mode == 'LSTM':

            self.rnn_cell_list.append(LSTMCell(self.input_size,
                                               self.hidden_size,
                                               self.bias))
            for l in range(1, self.num_layers):
                self.rnn_cell_list.append(LSTMCell(self.hidden_size,
                                                   self.hidden_size,
                                                   self.bias))

        elif mode == 'GRU':

            self.rnn_cell_list.append(GRUCell(self.input_size,
                                              self.hidden_size,
                                              self.bias))
            for l in range(1, self.num_layers):
                self.rnn_cell_list.append(GRUCell(self.hidden_size,
                                                  self.hidden_size,
                                                  self.bias))

        elif mode == 'RNN_TANH':

            self.rnn_cell_list.append(BasicRNNCell(self.input_size,
                                                   self.hidden_size,
                                                   self.bias,
                                                   "tanh"))
            for l in range(1, self.num_layers):
                self.rnn_cell_list.append(BasicRNNCell(self.hidden_size,
                                                       self.hidden_size,
                                                       self.bias,
                                                       "tanh"))

        elif mode == 'RNN_RELU':

            self.rnn_cell_list.append(BasicRNNCell(self.input_size,
                                                   self.hidden_size,
                                                   self.bias,
                                                   "relu"))
            for l in range(1, self.num_layers):
                self.rnn_cell_list.append(BasicRNNCell(self.hidden_size,
                                                       self.hidden_size,
                                                       self.bias,
                                                       "relu"))

        else:
            raise ValueError("Invalid RNN mode selected.")

        self.att_fc = nn.Linear(self.hidden_size, 1)
        self.fc = nn.Linear(self.hidden_size, self.output_size)


    def forward(self, input, hx=None):

        h0 = [None] * self.num_layers if hx is None else list(hx)

        X = list(input.permute(1, 0, 2))
        for j, l in enumerate(self.rnn_cell_list):
            hx = h0[j]
            for i in range(input.shape[1]):
                hx = l(X[i], hx)
                X[i] = hx if self.mode != 'LSTM' else hx[0]
        outs = X

        out = outs[-1].squeeze()

        out = self.fc(out)

        return out