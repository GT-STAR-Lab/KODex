"""
Define the functions that are used for koopman lifting
"""
from cmath import cos
import numpy as np
import math
from utils.fc_network import FCNetwork
from utils.gnn_networks import InteractionNetwork
import copy
import torch

class DraftedObservable(object):
    def __init__(self, num_handStates, num_objStates):
        self.num_ori_handStates = num_handStates
        self.num_ori_objStates = num_objStates
        self.num_ori_states = num_handStates + num_objStates
    
    def z(self, handState, objStates):  
        """
        Inputs: hand states(pos, vel) & object states(pos, vel)
        Outputs: state in lifted space
        Note: velocity is optional
        """
        obs = np.zeros(self.compute_observable(self.num_ori_handStates, self.num_ori_objStates))
        index = 0
        for i in range(self.num_ori_handStates):
            obs[index] = handState[i]
            index += 1
        for i in range(self.num_ori_handStates):
            obs[index] = handState[i] ** 2
            index += 1
        for i in range(self.num_ori_objStates):
            obs[index] = objStates[i]
            index += 1
        for i in range(self.num_ori_objStates):
            obs[index] = objStates[i] ** 2
            index += 1    
        for i in range(self.num_ori_objStates):
            for j in range(i + 1, self.num_ori_objStates):
                obs[index] = objStates[i] * objStates[j]
                index += 1            
        for i in range(self.num_ori_handStates):
            for j in range(i + 1, self.num_ori_handStates):
                obs[index] = handState[i] * handState[j]
                index += 1
        # can add handState[i] ** 3
        for i in range(self.num_ori_handStates):
            obs[index] = handState[i] ** 3
            index += 1
        # for i in range(self.num_ori_objStates):
        #     obs[index] = objStates[i] ** 3
        #     index += 1   
        # add the third-order polynomial of the hand states (more observables)
        # for i in range(self.num_ori_handStates):
        #     for j in range(self.num_ori_handStates):
        #         obs[index] = handState[i] ** 2 * handState[j]
        #         index += 1
        for i in range(self.num_ori_objStates):
            for j in range(self.num_ori_objStates):
                obs[index] = objStates[i] ** 2 * objStates[j]
                index += 1
        return obs
    
    def compute_observable(self, num_hand, num_obj):
        """
        Observation functions: original states, original states^2, cross product of hand states
        """
        # return int(2 * (num_hand + num_obj) + (num_hand - 1) * num_hand / 2 + num_obj + num_hand ** 2)  # include the second-order cross-product polynomial terms for hand_states and cubic terms
        # return int(3 * (num_hand + num_obj) + (num_hand - 1) * num_hand / 2)  # include the second-order cross-product polynomial terms for hand_states and cubic terms
        # return int(2 * (num_hand + num_obj))  # simplest version
        # return int(2 * (num_hand + num_obj) + (num_hand - 1) * num_hand / 2)  # include the second-order cross-product polynomial terms for hand_states
        # return int(2 * (num_hand + num_obj) + (num_obj - 1) * num_obj / 2)  # include the second-order cross-product polynomial terms for hand_states
        return int(2 * (num_hand + num_obj) + (num_obj - 1) * num_obj / 2 + (num_hand - 1) * num_hand / 2 + num_hand + num_obj ** 2)  
        # include the second-order cross-product polynomial terms for hand_states
        # return np.array([x[0], x[1], x[0]**2, (x[0]**2)*x[1], u[0]])

class MLPObservable(object):
    def __init__(self, num_handStates, num_objStates, param):
        self.param = param
        self.num_ori_states = num_handStates + num_objStates
        self.encoder = [i for i in param['encoder']]
        self.encoder.insert(0, self.num_ori_states)
        self.decoder = [i for i in param['decoder']]
        self.decoder.append(self.num_ori_states)
        self.encoder_NN = FCNetwork(self.encoder, param['nonlinearity'], param['encoder_seed'])
        self.decoder_NN = FCNetwork(self.decoder, param['nonlinearity'], param['decoder_seed'])
        self.trainable_params = list(self.encoder_NN.parameters()) + list(self.decoder_NN.parameters()) 

    def count_parameters(self):
        return sum(p.numel() for p in self.trainable_params if p.requires_grad)

    def z(self, input_state):
        """
        Inputs: original states: hand states(pos, vel) & object states(pos, vel)
        Outputs: state in lifted space
        Note: velocity is optional
        """ 
        z = self.encoder_NN(input_state)
        return z

    def z_inverse(self, liftedStates):
        """
        Inputs: state in lifted space
        Outputs: original states: hand states(pos, vel) & object states(pos, vel) 
        Note: velocity is optional
        """ 
        x = self.decoder_NN(liftedStates)
        return x
    
    def train(self): # set the models in training mode
        self.encoder_NN.train()
        self.decoder_NN.train() 
    
    def eval(self):  # set the models in evaluation mode
        self.encoder_NN.eval()
        self.decoder_NN.eval()

    def loss(self, batch_data, encoder_lambda, pred_lambda): # the loss consists of three part: 
        # 1). Intrinsic coordinates transformation loss, 2). Linear dynamics loss, 3). some others
        out = {}
        lifted_data = self.z(batch_data)
        batch_data_inverse = self.z_inverse(lifted_data)
        self.encoder_loss_criterion = torch.nn.MSELoss()
        self.encoder_loss = encoder_lambda * self.encoder_loss_criterion(batch_data_inverse, batch_data.detach().to(torch.float32))  # the auto-encoder loss
        out['Encoder_loss_torch'] = self.encoder_loss.item()
        batch_data_inverse_numpy = batch_data_inverse.detach().numpy()
        batch_data_numpy = batch_data.detach().numpy()
        Encoder_loss = np.abs(batch_data_numpy - batch_data_inverse_numpy)
        out['Encoder_loss_numpy'] = Encoder_loss
        self.pred_loss_criterion = torch.nn.MSELoss()
        for i in range(lifted_data.shape[0]):
            if i == 0:
                A = torch.outer(lifted_data[i,1,:], lifted_data[i,0,:]) / lifted_data.shape[0]
                G = torch.outer(lifted_data[i,0,:], lifted_data[i,0,:]) / lifted_data.shape[0]
            else:
                A += torch.outer(lifted_data[i,1,:], lifted_data[i,0,:]) / lifted_data.shape[0]
                G += torch.outer(lifted_data[i,0,:], lifted_data[i,0,:]) / lifted_data.shape[0]
        koopman_operator = torch.matmul(A, torch.linalg.pinv(G)) # using the batch data, compute the koopman matrix
        Linear_evol = torch.matmul(lifted_data[:,0,:], koopman_operator)
        self.pred_loss = pred_lambda * self.pred_loss_criterion(Linear_evol, lifted_data[:,1,:])  # the prediction loss
        out['Pred_loss_torch'] = self.pred_loss.item()
        Linear_evol_numpy = Linear_evol.detach().numpy()
        lifted_data_numpy = lifted_data[:,1,:].detach().numpy()
        pred_loss = np.abs(lifted_data_numpy - Linear_evol_numpy)
        out['Pred_loss_numpy'] = pred_loss
        loss = self.encoder_loss + self.pred_loss # loss is the sum of each component -> this one is used for pytorch training
        return loss, out, koopman_operator

    def load(self, encoder_path, decoder_path):
        self.encoder_NN.load_state_dict(torch.load(encoder_path))
        self.decoder_NN.load_state_dict(torch.load(decoder_path))
        
class GNNObservable(object):
    def __init__(self, num_objStates, param):
        self.gnn_encoder = InteractionNetwork(num_objStates, param['relation_domain'], param['relation_encoder'], param['object_encoder'], param['gnn_nonlinearity'], param['gnn_encoder_seed'])
        tmp = param['object_decoder']
        tmp.append(num_objStates)
        self.gnn_decoder = InteractionNetwork(param['object_encoder'][-1], param['relation_domain'], param['relation_decoder'], tmp, param['gnn_nonlinearity'], param['gnn_decoder_seed'])
        self.trainable_params = list(self.gnn_encoder.parameters()) + list(self.gnn_decoder.parameters())
        self._rollout = False

    def count_parameters(self):
        return sum(p.numel() for p in self.trainable_params if p.requires_grad)

    def z(self, input_state, relations, flag):
        """
        Inputs: original states: hand states(pos, vel) & object states(pos, vel); defined relations & relation types
        Outputs: state in lifted space
        Note: velocity is optional
        """ 
        z = self.gnn_encoder(input_state, relations['sender_relations'], relations['receiver_relations'], relations['relation_info'], flag) 
        return z

    def z_inverse(self, liftedStates, relations, flag):
        """
        Inputs: state in lifted space; defined relations & relation types
        Outputs: original states: hand states(pos, vel) & object states(pos, vel) 
        Note: velocity is optional
        """ 
        x = self.gnn_decoder(liftedStates, relations['sender_relations'], relations['receiver_relations'], relations['relation_info'], flag)
        return x

    def train(self): # set the models in training mode
        self.gnn_encoder.train()
        self.gnn_decoder.train() 
    
    def eval(self):  # set the models in evaluation mode
        self.gnn_encoder.eval()
        self.gnn_decoder.eval()

    def set_status(self, flag):
        if flag == 0:
            self._rollout = True

    def loss(self, batch_data, relations, encoder_lambda, pred_lambda): # the loss consists of three part: 
        batch_size = batch_data.shape[0]
        out = {}
        lifted_data = self.z(batch_data, relations, self._rollout)
        # .view(effect_receivers_t_1.shape[0], -1)
        batch_data_inverse = self.z_inverse(lifted_data, relations, self._rollout)
        self.encoder_loss_criterion = torch.nn.MSELoss()
        self.encoder_loss = encoder_lambda * self.encoder_loss_criterion(batch_data_inverse, batch_data.detach())  # the auto-encoder loss
        out['Encoder_loss_torch'] = self.encoder_loss.item()
        batch_data_inverse_numpy = batch_data_inverse.detach().numpy()
        batch_data_numpy = batch_data.detach().numpy()
        Encoder_loss = np.abs(batch_data_numpy - batch_data_inverse_numpy)
        out['Encoder_loss_numpy'] = Encoder_loss
        self.pred_loss_criterion = torch.nn.MSELoss()
        lifted_data = lifted_data.view(batch_size, 2, -1)  # Squeeze along the last 2 axis
        for i in range(lifted_data.shape[0]):
            if i == 0:
                A = torch.outer(lifted_data[i,1,:], lifted_data[i,0,:]) / lifted_data.shape[0]
                G = torch.outer(lifted_data[i,0,:], lifted_data[i,0,:]) / lifted_data.shape[0]
            else:
                A += torch.outer(lifted_data[i,1,:], lifted_data[i,0,:]) / lifted_data.shape[0]
                G += torch.outer(lifted_data[i,0,:], lifted_data[i,0,:]) / lifted_data.shape[0]
        koopman_operator = torch.matmul(A, torch.linalg.pinv(G)) # using the batch data, compute the koopman matrix
        Linear_evol = torch.matmul(lifted_data[:,0,:], koopman_operator)
        self.pred_loss = pred_lambda * self.pred_loss_criterion(Linear_evol, lifted_data[:,1,:])  # the prediction loss
        out['Pred_loss_torch'] = self.pred_loss.item()
        Linear_evol_numpy = Linear_evol.detach().numpy()
        lifted_data_numpy = lifted_data[:,1,:].detach().numpy()
        pred_loss = np.abs(lifted_data_numpy - Linear_evol_numpy)
        out['Pred_loss_numpy'] = pred_loss
        loss = self.encoder_loss + self.pred_loss # loss is the sum of each component -> this one is used for pytorch training
        return loss, out, koopman_operator

    def load(self, encoder_path, decoder_path):
        self.gnn_encoder.load_state_dict(torch.load(encoder_path))
        self.gnn_decoder.load_state_dict(torch.load(decoder_path))

    def Create_relations(self, batch_size, gnn_num_obj, gnn_num_relation, param, eval_length):
        receiver_relations = np.zeros((batch_size, gnn_num_obj, gnn_num_relation))
        sender_relations = np.zeros((batch_size, gnn_num_obj, gnn_num_relation))
        relation_info = np.zeros((batch_size, gnn_num_relation, param['relation_domain']))
        receiver_relations[:, 0, 0:5] = 1
        cnt = 0
        for i in [1,2,3,4,5]:
            sender_relations[:, 0, (cnt * gnn_num_obj) + i] = 1
            cnt += 1
        receiver_relations[:, 1, 6] = 1 
        receiver_relations[:, 1, 11] = 1
        sender_relations[:, 1, 0] = 1
        sender_relations[:, 1, 41] = 1
        receiver_relations[:, 2, 12] = 1 
        receiver_relations[:, 2, 17] = 1
        sender_relations[:, 2, 0] = 1
        sender_relations[:, 2, 41] = 1 
        receiver_relations[:, 3, 18] = 1 
        receiver_relations[:, 3, 23] = 1  
        sender_relations[:, 3, 0] = 1
        sender_relations[:, 3, 41] = 1
        receiver_relations[:, 4, 24] = 1 
        receiver_relations[:, 4, 29] = 1  
        sender_relations[:, 4, 0] = 1
        sender_relations[:, 4, 41] = 1
        receiver_relations[:, 5, 30] = 1 
        receiver_relations[:, 5, 35] = 1  
        sender_relations[:, 5, 0] = 1
        sender_relations[:, 5, 41] = 1
        receiver_relations[:, 6, 37:42] = 1 
        cnt = 0
        for i in [1,2,3,4,5]:
            sender_relations[:, 6, (cnt * gnn_num_obj) + i] = 1
            cnt += 1
        relation_info[:, 0:5, 0] = 1
        relation_info[:, 6, 0] = 1
        relation_info[:, 11, 1] = 1
        relation_info[:, 12, 0] = 1
        relation_info[:, 17, 1] = 1
        relation_info[:, 18, 0] = 1
        relation_info[:, 23, 1] = 1
        relation_info[:, 24, 0] = 1
        relation_info[:, 29, 1] = 1
        relation_info[:, 30, 0] = 1
        relation_info[:, 35, 1] = 1
        relation_info[:, -5:-1, 1] = 1
        relation_info[:, -1, 1] = 1
        # Todo: check if this part is implemented correctly
        relations = {}
        relations['receiver_relations'] = torch.from_numpy(receiver_relations).to(torch.float32)
        relations['sender_relations'] = torch.from_numpy(sender_relations).to(torch.float32)
        relations['relation_info'] = torch.from_numpy(relation_info).to(torch.float32)
        relations_eval = {}
        receiver_relations_eval = relations['receiver_relations'][0][None,:].repeat(eval_length, 1, 1)
        sender_relations_eval = relations['sender_relations'][0][None,:].repeat(eval_length, 1, 1)
        relation_info_eval = relations['relation_info'][0][None,:].repeat(eval_length, 1, 1)
        relations_eval['receiver_relations'] = receiver_relations_eval
        relations_eval['sender_relations'] = sender_relations_eval
        relations_eval['relation_info'] = relation_info_eval
        return relations, relations_eval

    def Create_states(self):
        hand_dict = {}
        hand_dict['palm'] = [0, 3] # or [0, 2]
        hand_dict['forfinger'] = [3, 7]
        hand_dict['middlefinger'] = [7, 11] 
        hand_dict['ringfinger'] = [11, 15]
        hand_dict['littlefinger'] = [15, 19]
        hand_dict['thumb'] = [19, 24]
        gnn_num_obj = 7 # palm, five fingers, manipulated object
        gnn_num_relation = gnn_num_obj * (gnn_num_obj - 1) 
        return hand_dict, gnn_num_obj, gnn_num_relation