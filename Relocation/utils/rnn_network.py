import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

class RNNNetwork(nn.Module):
    def __init__(self, rnn_cell,
                 m,
                 n):
        """
        :param rnn_cell: RNN (NCP) module
        """
        # nn.Module that unfolds a RNN cell into a sequence
        super(RNNNetwork, self).__init__()
        self.rnn_cell = rnn_cell
        self.obs_dim = m
        self.act_dim = n
        # follow by a fc_layer
        self.fc_input_layer = nn.Linear(in_features=self.obs_dim, out_features=rnn_cell.input_size)  # map observation_dim to RNN_input dim
        self.fc_output_layer = nn.Linear(in_features=rnn_cell.hidden_size, out_features=self.act_dim)

    def forward(self, x): # this is used for training
        device = x.device
        batch_size = x.size(0)
        # hidden states were set to be zeros at time 0
        hidden_state = (  # h_{0} and c_{0}
            torch.zeros((self.rnn_cell.num_layers, batch_size, self.rnn_cell.hidden_size), device=device),
            torch.zeros((self.rnn_cell.num_layers, batch_size, self.rnn_cell.hidden_size), device=device)
        )
        x = self.fc_input_layer(x.float())  # (batch_size, seq_len, self.rnn_cell.input_size)
        outputs, hidden_state = self.rnn_cell(x, hidden_state)  
        # output.shape = (batch_size, seq_len, self.rnn_cell.hidden_size)    
        # output -> hidden_state at each time step (the top layer if multiple layers), and we use this as the output data
        # hidden_state = (hidden_state[0] = h_{T} and hidden_state[1] = c_{T}) at last time step T, if there are multiple layers, output are the h_{t} for the top layers
        outputs = self.fc_output_layer(outputs)  #(batch_size, seq_len, act_dim)        
        return outputs

    def predict(self, observation, hidden_state):
        # for execution, so batch_size is always 1. Not for training
        observation = self.fc_input_layer(observation)
        output, hidden_state = self.rnn_cell(observation.view(1, 1, -1), hidden_state) 
        output = self.fc_output_layer(output)
        return output, hidden_state