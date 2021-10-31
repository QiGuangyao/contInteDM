import torch
import numpy as np
from matplotlib import pyplot as plt
import pdb

class ApplyGradient(object):
    """The model"""

    def __init__(self, model, hp, is_cuda=True):

        if is_cuda:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        '''used during training for performance'''
        self._cost_reg_init = torch.tensor(0., device=self.device)
        self._max_norm = torch.tensor(1., device=self.device)
        self.hp = hp

        # used hyper-parameters during training
        self.alpha = torch.tensor(hp['alpha'], device=self.device)

        self.rnn_obj = model


        if is_cuda:
            self.rnn_obj.cuda(device=self.device)

        # weight list, used for regularization
        self.weight_list = [self.rnn_obj.weight_ih, self.rnn_obj.weight_hh, self.rnn_obj.weight_out]
        self.out_weight = self.rnn_obj.weight_out
        self.hidden_weight = self.rnn_obj.weight_hh

        self.out_b = self.rnn_obj.b_out
        self.activate_func = self.rnn_obj.activate_func

        # regularization parameters
        self.l1_weight = torch.tensor(hp['l1_weight'], device=self.device)
        self.l2_weight = torch.tensor(hp['l2_weight'], device=self.device)

        self.l2_activity = torch.tensor(hp['l2_activity'], device=self.device)
        self.l1_activity = torch.tensor(hp['l1_activity'], device=self.device)

        if hp['optimizer'] == 'adam':
            self.optimizer = torch.optim.Adam(self.rnn_obj.parameters(), lr=hp['learning_rate'])
        elif hp['optimizer'] == 'sgd':
            self.optimizer = torch.optim.SGD(self.rnn_obj.parameters(), lr=hp['learning_rate'])

    # calculate cost_reg
    def cost_regularization(self, seq_mask):
        cost_reg = self._cost_reg_init
        # for weight regular
        if self.l1_weight > 0:
            temp = self._cost_reg_init
            for v in self.weight_list:
                temp = temp + torch.mean(torch.abs(v))
            cost_reg = cost_reg + temp * self.l1_weight

        if self.l2_weight > 0:
            temp = self._cost_reg_init  #
            for x in self.weight_list:
                temp = temp + torch.mean(x ** 2)
            cost_reg = cost_reg + temp * self.l2_weight

        # for neural activity regular
        if self.l2_activity > 0:
            seq_mask_n_element = torch.sum(seq_mask, dim=0)
            cost_reg = cost_reg + torch.mean(torch.sum((self.activity_shaped * seq_mask) ** 2,
                                                                 dim=0) / seq_mask_n_element) * self.l2_activity

        if self.l1_activity > 0:
            seq_mask_n_element = torch.sum(seq_mask, dim=0)
            cost_reg = cost_reg + torch.mean(torch.sum(torch.abs(self.activity_shaped * seq_mask),
                                                                 dim=0) / seq_mask_n_element) * self.l1_activity
        return cost_reg


    def compute_cost(self, **kwargs):

        inputs = kwargs['inputs']
        target_outputs = kwargs['target_outputs']
        cost_mask = kwargs['cost_mask']
        cost_start_time = kwargs['cost_start_time']
        cost_end_time = kwargs['cost_end_time']
        initial_state = kwargs['initial_state']
        seq_mask = kwargs['seq_mask'].type(torch.float32).unsqueeze(2)

        self.batch_size, self.n_hidden = initial_state.shape

        #forward
        self.state_collector = self.rnn_obj(inputs, initial_state)
        self.state_shaped = torch.cat(self.state_collector[cost_start_time + 1:cost_end_time + 1], dim=0).view(-1, self.batch_size, self.n_hidden)
        self.activity_shaped = self.activate_func(self.state_shaped)
        self.outputs = torch.matmul(self.activity_shaped, self.out_weight) + self.out_b
        cost_mask_length = torch.sum(cost_mask, dim=0)


        # cost_lsq
        self.cost_lsq = torch.mean(
            torch.sum(((self.outputs - target_outputs) ** 2) * cost_mask, dim=0) / cost_mask_length)
        # cost regularization
        self.cost_reg = self.cost_regularization(seq_mask)
        self.cost = self.cost_lsq + self.cost_reg
        self.cost = self.cost_lsq + self.cost_reg


    def stepper(self, **kwargs):

        self.optimizer.zero_grad()
        self.compute_cost(**kwargs)
        self.cost.backward()

        #grad clip
        if self.cost > 0.1:
            torch.nn.utils.clip_grad_value_(self.rnn_obj.parameters(), self._max_norm)
        else:
            torch.nn.utils.clip_grad_norm_(self.rnn_obj.parameters(), self._max_norm, 2)

        self.optimizer.step()

        self.rnn_obj.self_weight_clipper()
