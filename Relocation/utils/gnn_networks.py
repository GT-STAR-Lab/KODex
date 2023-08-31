import numpy as np
import torch
import torch.nn as nn

class RelationalModel(nn.Module):
    def __init__(self, input_size, model_size, nonlinearity='tanh', seed = 100):
        super(RelationalModel, self).__init__()
        
        self.layer_sizes = model_size
        self.layer_sizes.insert(0, input_size)
        torch.manual_seed(seed)
        
        self.fc_layers = nn.ModuleList([nn.Linear(self.layer_sizes[i], self.layer_sizes[i+1]) \
                    for i in range(len(self.layer_sizes) -1)])  # stack severeal layers together.
        self.nonlinearity = torch.relu if nonlinearity == 'relu' else torch.tanh  
    
    def forward(self, x):
        '''
        Args:
            x: [batch_size, n_relations, input_size]
        Returns:
            [batch_size, n_relations, output_size]
        '''
        if x.is_cuda:
                out = x.to('cpu')
        else:
            out = x
        for i in range(len(self.fc_layers)):
            out = self.fc_layers[i](out)
            out = self.nonlinearity(out)
        return out

class ObjectModel(nn.Module):
    def __init__(self, input_size, model_size, nonlinearity='tanh', seed = 100):
        super(ObjectModel, self).__init__()
        
        self.layer_sizes = model_size
        self.layer_sizes.insert(0, input_size)
        torch.manual_seed(seed)

        self.fc_layers = nn.ModuleList([nn.Linear(self.layer_sizes[i], self.layer_sizes[i+1]) \
                    for i in range(len(self.layer_sizes) -1)])  # stack severeal layers together.
        self.nonlinearity = torch.relu if nonlinearity == 'relu' else torch.tanh 

    def forward(self, x):
        '''
        Args:
            x: [batch_size, n_objects, input_size]
        Returns:
            [batch_size * n_objects, 2] speedX and speedY
        '''
        if x.is_cuda:
            out = x.to('cpu')
        else:
            out = x
        for i in range(len(self.fc_layers) - 1):
            out = self.fc_layers[i](out)
            out = self.nonlinearity(out)
        out = self.fc_layers[-1](out)
        return out

class InteractionNetwork(nn.Module):
    def __init__(self, object_dim, relation_dim, relation_model, object_model, nonlinearity, seed):
        super(InteractionNetwork, self).__init__()
        self.relational_model = RelationalModel(2 * object_dim + relation_dim, relation_model, nonlinearity, seed)
        # object_model defines the output dimension of the encoder(dim of lifted space)/decoder(dim of original space)
        self.object_model     = ObjectModel(object_dim + relation_model[-1], object_model, nonlinearity, seed)
        
    def forward(self, objects, sender_relations, receiver_relations, relation_info, flag):
        # in our case, the variable objects consists of palm, each finger and the manipulated object
        if not flag:
            z_t = objects[:,0,:]
            z_t_1 = objects[:,1,:]
            senders_t   = sender_relations.permute(0, 2, 1).bmm(z_t)
            senders_t_1   = sender_relations.permute(0, 2, 1).bmm(z_t_1)
            receivers_t = receiver_relations.permute(0, 2, 1).bmm(z_t)
            receivers_t_1 = receiver_relations.permute(0, 2, 1).bmm(z_t_1)
            effects_t = self.relational_model(torch.cat([senders_t, receivers_t, relation_info], 2))  # cat: 2 -> Concatenate along the second axis
            effects_t_1 = self.relational_model(torch.cat([senders_t_1, receivers_t_1, relation_info], 2))  # cat: 2 -> Concatenate along the second axis
            effect_receivers_t = receiver_relations.bmm(effects_t)
            effect_receivers_t_1 = receiver_relations.bmm(effects_t_1)
            predicted_t = self.object_model(torch.cat([z_t, effect_receivers_t], 2))[:,None,:]
            predicted_t_1 = self.object_model(torch.cat([z_t_1, effect_receivers_t_1], 2))[:,None,:]
            return torch.cat([predicted_t, predicted_t_1], 1)
        else:
            z_t = objects
            senders_t   = sender_relations.permute(0, 2, 1).bmm(z_t)
            receivers_t = receiver_relations.permute(0, 2, 1).bmm(z_t)
            effects_t = self.relational_model(torch.cat([senders_t, receivers_t, relation_info], 2))  # cat: 2 -> Concatenate along the second axis
            effect_receivers_t = receiver_relations.bmm(effects_t)
            predicted_t = self.object_model(torch.cat([z_t, effect_receivers_t], 2))
            return predicted_t
        
