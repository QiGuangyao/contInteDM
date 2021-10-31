"""Main training loop"""

from __future__ import division

import numpy as np
import torch
import torch
from torch.utils.data import DataLoader
from collections import defaultdict
import time
import sys
import os
sys.path.append('..')

from src import apply_gradients
from src import network
from src import tools
from src import dataset
import pdb
import pickle as pkl

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



def do_eval(rule_trains=None, model=None, hp=None, is_cuda=True):

    """Do evaluation.
    Args:
        model: Model class instance
        rule_train: string or list of strings, the rules being trained
    """

    # trainner stepper
    train_stepper_eval = apply_gradients.ApplyGradient(model, hp, is_cuda)

    def collate_fn(batch):
        return batch[0]

    # get samples for testing
    dataset_test = dataset.SampleSetForTrain(rule_trains, hp, mode='test', is_cuda=is_cuda)
    dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn)


    for i_batch, sample_batched in enumerate(dataloader_test):
        print('i_batch',i_batch)

        cost_lsq_tmp = list()
        cost_reg_tmp = list()

        if is_cuda:
            sample_batched['inputs'] = sample_batched['inputs'].cuda()
            sample_batched['target_outputs'] = sample_batched['target_outputs'].cuda()
            sample_batched['cost_mask'] = sample_batched['cost_mask'].cuda()
            sample_batched['seq_mask'] = sample_batched['seq_mask'].cuda()
            sample_batched['initial_state'] = sample_batched['initial_state'].cuda()

        sample_batched['rule_trains'] = rule_trains
        with torch.no_grad():
            train_stepper_eval.compute_cost(**sample_batched)

        cost_lsq_tmp.append(train_stepper_eval.cost_lsq.detach().cpu().numpy())
        cost_reg_tmp.append(train_stepper_eval.cost_reg.detach().cpu().numpy())

        print('| mean cost {:0.6f}'.format(np.mean(cost_lsq_tmp)) +
              '| mean c_reg {:0.6f}'.format(np.mean(cost_reg_tmp)))

        min_cost = np.inf

        if cost_lsq_tmp[-1] < min_cost:
            min_cost = cost_lsq_tmp[-1]

        if rule_trains == 'contextInteg_decision_making':
            actual_output = train_stepper_eval.outputs.detach().cpu().numpy()
            cue = sample_batched['cue']#[0,1]=[left,right]
            #target_choice = [0,1,2,3]
            target_choice = (1 - cue) * [1 * (sample_batched['strength1_color'] > sample_batched['strength2_color']) + 2 * (
                        sample_batched['strength1_color'] <= sample_batched['strength2_color'])] \
                            + cue * [3 * (sample_batched['strength1_motion'] > sample_batched['strength2_motion']) + 4 * (
                        sample_batched['strength1_motion'] <= sample_batched['strength2_motion'])]  #


            fix_start = sample_batched['epochs']['stim'][0]
            fix_off = sample_batched['epochs']['integrate'][1]

            # fix_start = sample_batched['epochs']['cue'][0]
            # fix_off = sample_batched['epochs']['integrate'][1]
            #print("fix_start,fix_off",fix_start,fix_off)

            success_action_prob, mean_choice_error = network.get_perf_multi_decision(actual_output, target_choice, fix_start, fix_off)

            success_action_prob = success_action_prob.tolist()
            mean_choice_error = mean_choice_error.tolist()

            log = dict()
            log['cost'] = cost_lsq_tmp[-1].tolist()
            log['creg'] = cost_reg_tmp[-1].tolist()
            log['success_action_prob'] = success_action_prob
            log['mean_choice_error'] = mean_choice_error


            print('| success_action_prob {:0.7f}'.format(success_action_prob) +
                  '| mean_choice_error {:0.7f}'.format(mean_choice_error))

        if rule_trains == 'contextInteg_decision_making':
            return log, cost_lsq_tmp[-1], success_action_prob, mean_choice_error


#save trained model
def save_final_result(model=None, model_dir=None,hp=None, log=None):
    save_path = os.path.join(model_dir, 'finalResult')
    tools.mkdir_p(save_path)
    model.save(save_path)
    log['model_dir'] = save_path
    tools.save_log(log)
    tools.save_hp(hp, save_path)



def Trainer(max_samples=1e7,
            display_step=500,
            rule_trains=None,
            model=None,
            hp=None,
            model_dir=None,
            is_cuda=True):
    '''Train the network.
        Args:
            max_samples: int, maximum number of training samples
            display_step: int, display steps
            model_dir: str, training directory
            hp: dictionary of hyperparameters
        Returns:
            model is stored at model_dir/model.pth
            training configuration is stored at model_dir/hp.json
        '''
    print('***')
    # model directory
    tools.mkdir_p(model_dir)
    save_path = os.path.join(model_dir, 'finalResult')
    tools.mkdir_p(save_path)

    # GPU
    if is_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    hp['alpha'] = 1.0 * hp['dt'] / hp['tau']

    if model is None:
        model = network.RNN(hp, rule_trains, is_cuda)
    print('***')

    # # load or create log
    log = defaultdict(list)


    # collate_fn of dataloader
    def collate_fn(batch):
        return batch[0]
    # dataset_train is a list and each item is a dict
    dataset_train = dataset.SampleSetForTrain(rule_trains, hp, mode='train', is_cuda=is_cuda)
    dataloader_train = DataLoader(dataset_train, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn)

    # dataTrainColl = {}
    # for kkk in range(1,100):
    #     # print(kkk)
    #     dataTrainColl[kkk] = dataset_train[kkk]
    # pkl.dump({
    #     'dataTrainColl':dataTrainColl
    # },
    #          open('/Users/qiguangyao/github/contextInteg/src/saved_data/datasetTrain.pickle','wb'))



    #train stepper
    step_training = apply_gradients.ApplyGradient(model, hp, is_cuda)

    min_cost = np.inf
    model_save_idx = 0

    for key, val in hp.items():
        print('{:20s} = '.format(key) + str(val))

    # Record time
    t_start = time.time()

    success_action_prob_list=[]
    mean_choice_error_list=[]
    # Keep training until reach max iterations
    for step, trial_batched in enumerate(dataloader_train):
        # print(step)
        try:
            if is_cuda:

                trial_batched['inputs'] = trial_batched['inputs'].cuda()
                trial_batched['target_outputs'] = trial_batched['target_outputs'].cuda()
                trial_batched['cost_mask'] = trial_batched['cost_mask'].cuda()
                trial_batched['seq_mask'] = trial_batched['seq_mask'].cuda()
                trial_batched['initial_state'] = trial_batched['initial_state'].cuda()

            trial_batched['rule_trains'] = rule_trains

            if model_save_idx < 5:
                step_training.l2_activity = torch.tensor(1e-3, device=device)
            else:
                step_training.l2_activity = torch.tensor(hp['l2_activity'], device=device)

            # Train the model
            step_training.stepper(**trial_batched)

            if step % display_step == 0:
                print("Training step",step)
                print('Trial {:7d}'.format(step * hp['batch_size_train']) + '    | Time ', tools.elapsed_time(time.time() - t_start))

                if rule_trains == 'contextInteg_decision_making':
                    log, cost, success_action_prob, mean_choice_err = do_eval(rule_trains, model, hp, is_cuda)
                    success_action_prob_list.append(success_action_prob)
                    mean_choice_error_list.append(mean_choice_err)
                    if not np.isfinite(cost):
                        return 'error'
                    if success_action_prob > 0.95 and mean_choice_err < 0.02:
                        print(success_action_prob_list)
                        print(mean_choice_error_list)
                        save_final_result(model, model_dir, hp, log)
                        break

            if step * hp['batch_size_train'] > max_samples:
                do_eval(rule_trains, model, hp, is_cuda)#
                break

        except KeyboardInterrupt:
            print("Optimization interrupted by user")
            break

    print("Optimization Finished!")

    return 'OK'



def Runner(
           mode='test',
           rule_trains=None,
           model=None,
           hp=None,
           model_dir=None,
           is_cuda=True,
           noise_on=True,
           **kwargs):

    tools.mkdir_p(model_dir)
    model_dir = model_dir

    rule_trains = rule_trains
    is_cuda = is_cuda
    # load or create hyper-parameters
    if hp is None:
        hp = tools.load_hp(model_dir, rule_trains)

    # hyper-parameters for time scale
    hp['alpha'] = 1.0 * hp['dt'] / hp['tau']
    hp = hp

    noise_on = noise_on

    # load or create model
    if model is None:
        if hp['rnn_type'] == 'RNN':
            model = network.RNN(hp, rule_trains, is_cuda)
        model.load(model_dir)
    else:
        model = model

    if not noise_on:
        model.sigma_rec = 0

    # trainner stepper
    train_step = apply_gradients.ApplyGradient(model, hp, is_cuda)
    # data loader
    dataset_run = dataset.SampleSetForRun(rule_trains, hp, noise_on=noise_on, mode=mode, **kwargs)

    sample = dataset_run.__getitem__()

    sample['rule_trains'] = rule_trains

    with torch.no_grad():
        train_step.compute_cost(**sample)

    return dataset_run.trial, train_step, model