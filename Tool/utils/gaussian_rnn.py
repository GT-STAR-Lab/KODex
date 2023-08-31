from utils.rnn_network import RNNNetwork
import numpy as np
import torch
from torch.autograd import Variable
# import kerasncp as kncp
# from kerasncp.torch import LTCCell

class RNN:
    def __init__(self,
                 m,
                 n,
                 input_sizes=64,
                 hidden_state = 64,
                 LSTM_layer = 1,
                 seed=None):
        """
        :param m: NN_input size
        :param n: NN_output size 
        :param hidden_sizes: network hidden layer sizes (currently 2 layers only)
        :param seed: random seed
        """
        # Set seed
        # ------------------------
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        # Policy network
        # ------------------------
        #  If batch_first is True, then the input and output tensors are provided as (batch, seq, feature) instead of (seq, batch, feature). 
        rnn_cell = torch.nn.LSTM(input_sizes, hidden_state, batch_first=True, num_layers = LSTM_layer) 
        # self.model = NCPNetwork(ltc_cells, env)
        self.model = RNNNetwork(rnn_cell, m, n)
        # make weights smaller
        # for param in list(self.model.parameters())[-2:]:  # only last layer
        #    param.data = 1e1 * param.data
        self.trainable_params = list(self.model.parameters())

        # Placeholders
        # ------------------------
        self.obs_var = Variable(torch.randn(m), requires_grad=False)
        self.hidden_state_var = Variable(torch.randn( # generate an variable to store values
            (LSTM_layer, 1, self.model.rnn_cell.hidden_size)), requires_grad=False)
        self.cell_state_var = Variable(torch.randn(
            (LSTM_layer, 1, self.model.rnn_cell.hidden_size)), requires_grad=False)
    
    # get_action is used for executions
    def get_action(self, observation, hidden_state):
        o = np.float32(observation.reshape(1, 1, -1))
        self.obs_var.data = torch.from_numpy(o)
        # print(hidden_state[0].shape)
        self.hidden_state_var.data = torch.from_numpy(np.float32(hidden_state[0]))
        self.cell_state_var.data = torch.from_numpy(np.float32(hidden_state[1]))
        # print(self.hidden_state_var.shape)
        mean, hidden_state = self.model.predict(self.obs_var, (self.hidden_state_var, self.cell_state_var))
        mean = mean.data.numpy().ravel()
        new_hidden_state = hidden_state[0].data.numpy()
        new_cell_state = hidden_state[1].data.numpy()
        hidden_state = (new_hidden_state, new_cell_state)  # these are needed for the next time step.
        #Since the input is the obs at each time step, we have to manually pass the hidden state and the cell state
        return mean, hidden_state
