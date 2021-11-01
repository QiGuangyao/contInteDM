#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 31 15:44:30 2021

@author: qiguangyao
"""
#import torch
#from torch.utils.data import Dataset, DataLoader
import sys
import numpy as np
sys.path.append('..')
from src import task_reve
from src import task
#from src import dataset
#import pickle as pkl
# from src import generate_trials
#%% functions
def input_output_n(rule_trains):
    # basic timing tasks
    if rule_trains == 'contextInteg_decision_making':
        return 4+2, 4
def get_default_hp(rule_trains, random_seed=None):
    '''Get a default hp.

    Returns:
        hp : a dictionary containing training hpuration
    '''

    n_input, n_output = input_output_n(rule_trains)

    # default seed of random number generator
    if random_seed is None:
        seed = np.random.randint(1000000)
    else:
        seed = random_seed
    #seed = 321985
    hp = {
        'rule_trains': rule_trains,
        # batch size for training
        'batch_size_train': 64,#64, #128,#64,
        # batch_size for testing
        'batch_size_test': 512,#512,#512
        # Type of RNNs: RNN
        'rnn_type': 'RNN',
        # Optimizer adam or sgd
        'optimizer': 'adam',
        # Type of activation functions: relu, softplus
        'activation': 'softplus',

        # Time constant (ms)
        'tau': 20,
        # discretization time step (ms)
        'dt': 20,
        # discretization time step/time constant
        'alpha': 1,
        # initial standard deviation of non-diagonal recurrent weights

        'initial_std': 0.3,#0.25,#0.27,#0.3,
        # recurrent noise
        'sigma_rec': 0.05,
        # input noise
        'sigma_x': 0.01,#when traning sigma_x=0.01,
        # a default weak regularization prevents instability
        'l1_activity': 0,
        # l2 regularization on activity
        'l2_activity': 0,
        # l1 regularization on weight#
        'l1_weight': 0,
        # l2 regularization on weight
        'l2_weight': 0,
        # number of input units
        'n_input': n_input,
        # number of output units
        'n_output': n_output,
        # number of recurrent units
        'n_rnn': 256,
        # learning rate
        'learning_rate': 0.0005,#0.0005,
        # random number generator
        'seed': seed,
        'rng': np.random.RandomState(seed),
    }
    return hp
#%% generate trial
rule_trains = 'contextInteg_decision_making'
hp = get_default_hp(rule_trains, random_seed=None)
is_cuda = False
trial = task.generate_trials(rule_trains, hp, 'random', noise_on=True, batch_size=1)

trial_reve = task_reve.generate_trials(rule_trains, hp, 'random', noise_on=True, batch_size=1)

