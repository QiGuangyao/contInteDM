import torch
from torch.utils.data import Dataset, DataLoader
import sys

sys.path.append('..')
# from src import task
from src import task_reve
from src import tools


class SampleSetForTrain(Dataset):

    def __init__(self, rule_trains, hp, mode='train', is_cuda=True, **kwargs):
        '''provide name of the rules'''
        self.rule_trains = rule_trains

        self.hp = hp

        self.is_cuda = is_cuda

        if mode == 'train':
            self.bach_size = hp['batch_size_train']
            self.task_mode = 'random'
        elif mode == 'test':
            self.bach_size = hp['batch_size_test']
            self.task_mode = 'random_validate'
        elif mode == 'test_generalize':
            self.bach_size = hp['batch_size_test']
            self.task_mode = 'test_generalize'
        else:
            raise ValueError('Unknown mode: ' + str(mode))

        self.counter = 0
        self.kwargs = kwargs

    def __len__(self):
        '''arbitrary'''
        return 1000000
    def __getitem__(self, index):

        self.trial = task_reve.generate_trials(self.rule_trains, self.hp, self.task_mode, batch_size=self.bach_size, **self.kwargs)#from task to task_reve

        sample = dict()
        sample['inputs'] = torch.as_tensor(self.trial.x)
        sample['target_outputs'] = torch.as_tensor(self.trial.y)
        sample['cost_mask'] = torch.as_tensor(self.trial.cost_mask)
        sample['cost_start_time'] = 0
        sample['cost_end_time'] = self.trial.max_seq_len
        sample['seq_mask'] = tools.sequence_mask(self.trial.seq_len)
        sample['initial_state'] = torch.zeros((self.trial.x.shape[1], self.hp['n_rnn']))
        sample['epochs'] = self.trial.epochs


        if self.rule_trains == 'contextInteg_decision_making':
            sample['stim1_duration'] = self.trial.stim1_duration
            sample['strength1_color'] = self.trial.strength1_color
            sample['strength2_color'] = self.trial.strength2_color
            sample['strength1_motion'] = self.trial.strength1_motion#
            sample['strength2_motion'] = self.trial.strength2_motion
            sample['cue'] = self.trial.cue

        return sample


class SampleSetForRun(object):

    def __init__(self, rule_trains, hp, noise_on=True, mode='test', **kwargs):
        self.rule_trains = rule_trains
        self.hp = hp
        self.kwargs = kwargs
        self.noise_on = noise_on
        self.mode = mode

    def __getitem__(self):

        self.trial = task_reve.generate_trials(self.rule_trains, self.hp, self.mode, noise_on=self.noise_on, **self.kwargs)#from task to task_reve

        sample = dict()
        sample['inputs'] = torch.as_tensor(self.trial.x)
        sample['target_outputs'] = torch.as_tensor(self.trial.y)
        sample['cost_mask'] = torch.as_tensor(self.trial.cost_mask)
        sample['cost_start_time'] = 0
        sample['cost_end_time'] = self.trial.max_seq_len
        sample['seq_mask'] = tools.sequence_mask(self.trial.seq_len)
        sample['initial_state'] = torch.zeros((self.trial.x.shape[1], self.hp['n_rnn']))

        return sample


