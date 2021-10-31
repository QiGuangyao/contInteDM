"""Definition of the network model and various RNN cells"""
from __future__ import division

import torch
from torch import nn

import os
import math
import numpy as np


# Create Network
class RNN(nn.Module):
    def __init__(self, hp, rule_trains, is_cuda=True):
        super(RNN, self).__init__()

        n_input = hp['n_input']
        n_hidden = hp['n_rnn']
        n_output = hp['n_output']
        alpha = hp['alpha']
        sigma_rec = hp['sigma_rec']
        activate_func = hp['activation']

        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.hp = hp
        self.rule_trains = rule_trains#



        if is_cuda:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        if activate_func == 'relu':
            self.activate_func = lambda x: nn.functional.relu(x)
        elif activate_func == 'softplus':
            self.activate_func = lambda x: nn.functional.softplus(x)

        # basic timing task
        if self.rule_trains == 'integrate_production':
            if n_input != 2:
                raise Exception('n_input should be 2 for integrate_production')
            self.weight_ih = nn.Parameter(torch.empty(n_input, n_hidden).uniform_(-1./np.sqrt(1), 1./np.sqrt(1)))

        elif self.rule_trains == 'decision_making':
            if n_input != 2:
                raise Exception('n_input should be 2 for decision_making')
            weight_ih = torch.empty(n_input, n_hidden).uniform_(-1./np.sqrt(2.), 1./np.sqrt(2.))
            self.weight_ih = nn.Parameter(weight_ih)

        elif self.rule_trains == 'ctx_decision_making':
            if n_input != 4:
                raise Exception('n_input should be 4 for ctx_decision_making')
            weight_ih = torch.empty(n_input, n_hidden).uniform_(-1./np.sqrt(2.), 1./np.sqrt(2.))
            weight_ih[2:4, :].uniform_(-1./np.sqrt(1.), 1./np.sqrt(1.))
            self.weight_ih = nn.Parameter(weight_ih)

        elif self.rule_trains == 'contextInteg_decision_making':
            if n_input != 6:
                raise Exception('n_input should be 6 for contextInteg_decision_making_variable_delay')
            weight_ih = torch.empty(n_input, n_hidden).uniform_(-1./np.sqrt(2.), 1./np.sqrt(2.))
            weight_ih[4:6, :].uniform_(-1./np.sqrt(1.), 1./np.sqrt(1.))
            self.weight_ih = nn.Parameter(weight_ih)

        hh_mask = torch.ones(n_hidden, n_hidden) - torch.eye(n_hidden)
        non_diag = torch.empty(n_hidden, n_hidden).normal_(0, hp['initial_std']/np.sqrt(n_hidden))
        weight_hh = torch.eye(n_hidden)*0.999 + hh_mask * non_diag

        self.weight_hh = nn.Parameter(weight_hh)
        self.b_h = nn.Parameter(torch.zeros(1, n_hidden))

        self.weight_out = nn.Parameter(torch.empty(n_hidden, n_output).normal_(0., 0.4/math.sqrt(n_hidden)))
        self.b_out = nn.Parameter(torch.zeros(n_output,))

        self.alpha = torch.tensor(alpha, device=self.device)
        self.sigma_rec = torch.tensor(math.sqrt(2./alpha) * sigma_rec, device=self.device)


    def forward(self, inputs, initial_state):
        #print("network/forward inputs",inputs.size())

        """Most basic RNN: output = new_state = W_input * input + W_rec * act(state) + B + noise """

        state = initial_state
        state_collector = [state]

        for input_per_step in inputs:
            state_new = torch.matmul(self.activate_func(state), self.weight_hh) + self.b_h + \
                        torch.matmul(input_per_step, self.weight_ih) + torch.randn_like(state, device=self.device) * self.sigma_rec

            state = (1 - self.alpha) * state + self.alpha * state_new #alpha=1
            state_collector.append(state)


        return state_collector

    def out_weight_clipper(self):
        self.weight_out.data.clamp_(0.)

    def self_weight_clipper(self):
        diag_element = self.weight_hh.diag().data.clamp_(0., 1.)
        self.weight_hh.data[range(self.n_hidden), range(self.n_hidden)] = diag_element

    def save(self, model_dir):
        save_path = os.path.join(model_dir, 'model.pth')
        torch.save(self.state_dict(), save_path)

    def load(self, model_dir):
        if model_dir is not None:
            save_path = os.path.join(model_dir, 'model.pth')
            if os.path.isfile(save_path):
                self.load_state_dict(torch.load(save_path, map_location=lambda storage, loc: storage), strict=False)



def get_perf_multi_decision(output, target_choices, fix_start, fix_end, fix_strength=0.2, action_threshold=0.5, response_duration=int(300/20)):
    # print('output',output,'target', target_choices)

    batch_size = output.shape[1]
    action_at_fix = np.array([np.sum(output[fix_start[i]+1:fix_end[i]-1, i, :] > fix_strength) > 0 for i in range(batch_size)])
    no_action_at_motion = np.array([np.sum(output[fix_end[i]:fix_end[i]+response_duration, i, :] > action_threshold) == 0 for i in range(batch_size)])
    redundant_action_at_motion = np.array([np.sum(np.sum(output[fix_end[i]:fix_end[i]+response_duration, i, :] > action_threshold, axis=0) > 0) > 1 for i in range(batch_size)])
    fail_action = action_at_fix + no_action_at_motion + redundant_action_at_motion

    action_time = np.array([np.argmax(np.sum(output[fix_end[i]:fix_end[i]+response_duration, i, :] > action_threshold, axis=1) > 0) for i in range(batch_size)])
    action = np.concatenate([(output[[fix_end[i] + action_time[i]], i, :] > action_threshold).astype(np.int) for i in range(batch_size)], axis=0)

    #find actual choices
    actual_choices = []
    for i in range(batch_size):
        if np.max(action[i, :]) == 1:
            actual_choice = np.argmax(action[i, :] == 1) + 1
        else:
            actual_choice = 0
        actual_choices.append(actual_choice)

    #compara actual_choices with target_choices
    actual_choices = np.array(actual_choices)
    target_choices = target_choices.reshape(batch_size,).astype(np.int)
    choice_correct = ((actual_choices - target_choices) == 0).reshape(batch_size,)

    success_action_prob = 1 - np.nansum(fail_action)/batch_size
    success_action = 1 - fail_action

    mean_choice_correct = np.nanmean(choice_correct[np.argwhere(success_action)])
    # print('success_action',success_action, 'fail_action',fail_action)
    return success_action_prob, 1-mean_choice_correct
