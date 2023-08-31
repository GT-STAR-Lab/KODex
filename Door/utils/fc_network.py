import numpy as np
import torch
import torch.nn as nn


class FCNetwork(nn.Module):
    def __init__(self, model_size = None,
                 nonlinearity='tanh',   # either 'tanh' or 'relu'
                 seed = 100
                    ):
        super(FCNetwork, self).__init__()

        self.obs_dim = model_size[0]
        self.act_dim = model_size[-1]
        self.layer_sizes = model_size

        # hidden layers
        torch.manual_seed(seed)
        self.fc_layers = nn.ModuleList([nn.Linear(self.layer_sizes[i], self.layer_sizes[i+1]) \
                         for i in range(len(self.layer_sizes) -1)])  # stack severeal layers together.
        # The weights are initialzied in default by:
#        stdv = 1. / math.sqrt(self.weight.size(1))
#        self.weight.data.uniform_(-stdv, stdv)
#        if self.bias is not None:
#           self.bias.data.uniform_(-stdv, stdv)
        self.nonlinearity = torch.relu if nonlinearity == 'relu' else torch.tanh   

    def forward(self, x):
        if x.is_cuda:
            out = x.to('cpu')
        else:
            out = x
        out = out.to(torch.float32)
        for i in range(len(self.fc_layers)-1):
            out = self.fc_layers[i](out)
            out = self.nonlinearity(out)
        out = self.fc_layers[-1](out)
        return out