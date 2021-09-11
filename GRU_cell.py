import torch
import torch.nn as nn
import math


class GRUCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super(GRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        self.x2h = nn.Linear(input_size, 2 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 2 * hidden_size, bias=bias)

        self.x2r = nn.Linear(input_size, hidden_size, bias=bias)
        self.h2r = nn.Linear(hidden_size, hidden_size, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, input, hx=None):
        if hx is None:
            hx = input.new_zeros(input.size(0), self.hidden_size, requires_grad=False)

        z, r = torch.chunk(self.x2h(input) + self.h2h(hx), 2, -1)
        z = torch.sigmoid(z)
        r = torch.sigmoid(r)
        g = torch.tanh(self.h2r(hx) * r + self.x2r(input))
        hy = z * hx + (1 - z) * g

        return hy