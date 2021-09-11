import torch
import torch.nn as nn
from LSTM_cell import LSTMCell
from Basic_RNN_cell import BasicRNNCell
from GRU_cell import GRUCell


class BidirRecurrentModel(nn.Module):
    def __init__(self, mode, input_size, hidden_size, num_layers, bias, output_size):
        super(BidirRecurrentModel, self).__init__()
        self.mode = mode
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.output_size = output_size

        self.rnn_cell_list = nn.ModuleList()
        self.rnn_cell_list_rev = nn.ModuleList()

        if mode == 'LSTM':
            self.rnn_cell_list.append(LSTMCell(self.input_size,
                                               self.hidden_size,
                                               self.bias))
            for l in range(1, self.num_layers):
                self.rnn_cell_list.append(LSTMCell(self.hidden_size,
                                                   self.hidden_size,
                                                   self.bias))

            self.rnn_cell_list_rev.append(LSTMCell(self.input_size,
                                                   self.hidden_size,
                                                   self.bias))
            for l in range(1, self.num_layers):
                self.rnn_cell_list_rev.append(LSTMCell(self.hidden_size,
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

            self.rnn_cell_list_rev.append(GRUCell(self.input_size,
                                                  self.hidden_size,
                                                  self.bias))
            for l in range(1, self.num_layers):
                self.rnn_cell_list_rev.append(GRUCell(self.hidden_size,
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

            self.rnn_cell_list_rev.append(BasicRNNCell(self.input_size,
                                                       self.hidden_size,
                                                       self.bias,
                                                       "tanh"))
            for l in range(1, self.num_layers):
                self.rnn_cell_list_rev.append(BasicRNNCell(self.hidden_size,
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
            self.rnn_cell_list_rev.append(BasicRNNCell(self.input_size,
                                                       self.hidden_size,
                                                       self.bias,
                                                       "relu"))
            for l in range(1, self.num_layers):
                self.rnn_cell_list_rev.append(BasicRNNCell(self.hidden_size,
                                                           self.hidden_size,
                                                           self.bias,
                                                           "relu"))

        else:
            raise ValueError("Invalid RNN mode selected.")

        self.fc = nn.Linear(2 * self.hidden_size, self.output_size)


    def forward(self, input, hx=None):

        # In this forward pass we want to create our Bidirectional RNN from the rnn cells,
        # ..taking the hidden states from the final layer with their reversed counterparts
        # .. before concatening these and running them through the fully connected layer (fc)

        # The multi-layered RNN should be able to run when the mode is either
        # .. LSTM, GRU, RNN_TANH or RNN_RELU.


        X = list(input.permute(1, 0, 2))

        X_rev = list(input.permute(1, 0, 2))
        X_rev.reverse()
        hi = [None] * self.num_layers if hx is None else list(hx)
        hi_rev = [None] * self.num_layers if hx is None else list(hx)
        for j in range(self.num_layers):
            hx = hi[j]
            hx_rev = hi_rev[j]
            for i in range(input.shape[1]):
                hx = self.rnn_cell_list[j](X[i], hx)
                X[i] = hx if self.mode != 'LSTM' else hx[0]
                hx_rev = self.rnn_cell_list_rev[j](X_rev[i], hx_rev)
                X_rev[i] = hx_rev if self.mode != 'LSTM' else hx_rev[0]
        outs = X
        outs_rev = X_rev

        out = outs[-1].squeeze()
        out_rev = outs_rev[0].squeeze()
        out = torch.cat((out, out_rev), 1)

        out = self.fc(out)
        return out