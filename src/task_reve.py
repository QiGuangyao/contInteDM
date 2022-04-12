from __future__ import division
import numpy as np
import math
import sys
import pdb


def get_dist(original_dist):
    '''Get the distance in periodic boundary conditions'''
    return np.minimum(abs(original_dist), 2*np.pi-abs(original_dist))


class Trial(object):
    """Class representing a batch of trials."""

    def __init__(self, config, xtdim, batch_size):
        """A batch of trials.

        Args:
            config: dictionary of configurations
            xtdim: int, number of total time steps
            batch_size: int, batch size
        """
        self.float_type = 'float32'
        self.config = config
        self.dt = self.config['dt']

        self.n_input = self.config['n_input']
        self.n_output = self.config['n_output']

        self.batch_size = batch_size
        self.xtdim = xtdim

        # time major
        self.x = np.zeros((xtdim, batch_size, self.n_input), dtype=self.float_type)
        self.y = np.zeros((xtdim, batch_size, self.n_output), dtype=self.float_type)
        self.cost_mask = np.zeros((xtdim, batch_size, self.n_output), dtype=self.float_type)
        # strength of input noise
        self._sigma_x = config['sigma_x'] * math.sqrt(2./self.config['alpha'])


    def expand(self, var):
        """Expand an int/float to list."""
        if not hasattr(var, '__iter__'):
            var = [var] * self.batch_size
        return var

    def add(self, loc_type, loc_idx, ons, offs, strengths, gaussian_center=None):

        for i in range(self.batch_size):
            if loc_type == 'input':
                self.x[ons[i]: offs[i], i, loc_idx] = strengths[i]

            elif loc_type == 'out':
                self.y[ons[i]: offs[i], i, loc_idx] = strengths[i]

            elif loc_type == 'cost_mask':
                self.cost_mask[ons[i]: offs[i], i, loc_idx] = strengths[i]

            else:
                raise ValueError('Unknown loc_type')

    def add_x_noise(self):
        """Add input noise."""
        self.x += self.config['rng'].randn(*self.x.shape) * self._sigma_x


def _multisensory_output_cue_20_delay_40(config, mode, **kwargs):

    dt = config['dt']
    cue_duration = int(400/dt)
    rng = config['rng']
    response_duration = int(300/dt)

    if mode == 'random':
        batch_size = kwargs['batch_size']

        stim1_duration = (rng.uniform(800, 800, batch_size)/dt).astype(int)
        gamma_bar_color = rng.uniform(0.8, 1.2, batch_size)
        gamma_bar_motion = rng.uniform(0.8, 1.2, batch_size)

        c_color = rng.choice([-0.08, -0.04, -0.02, -0.01, 0.01, 0.02, 0.04, 0.08], (batch_size,))
        c_motion = rng.choice([-0.08, -0.04, -0.02, -0.01, 0.01, 0.02, 0.04, 0.08], (batch_size,))
        
        cue = rng.choice([0., 1.], (batch_size,))
        
    elif mode == 'random_validate':
        batch_size = kwargs['batch_size']

        stim1_duration = (rng.uniform(800, 800, batch_size) / dt).astype(int)
        gamma_bar_color = rng.uniform(0.8, 1.2, batch_size)
        gamma_bar_motion = rng.uniform(0.8, 1.2, batch_size)

        c_color = rng.choice([-0.08, -0.04, -0.02, -0.01, 0.01, 0.02, 0.04, 0.08], (batch_size,))
        c_motion = rng.choice([-0.08, -0.04, -0.02, -0.01, 0.01, 0.02, 0.04, 0.08], (batch_size,))
        cue = rng.choice([0., 1.], (batch_size,))
    elif mode == 'random_test':
        batch_size = kwargs['batch_size']

        stim1_duration = (rng.uniform(800, 800, batch_size)/dt).astype(int)
        # gamma_bar_color = rng.choice([0.8], (batch_size,))
        # gamma_bar_motion = rng.choice([0.8], (batch_size,))
        gamma_bar_color = rng.uniform(0.8, 1.2, batch_size)
        gamma_bar_motion = rng.uniform(0.8, 1.2, batch_size)

        c_color = rng.choice([-0.04, -0.02, -0.01,-0.005,0,0.005,0.01, 0.02, 0.04], (batch_size,))# -0.005,0,0.005
        c_motion = rng.choice([ -0.04, -0.02, -0.01,-0.005,0,0.005, 0.01, 0.02, 0.04], (batch_size,))

        # c_color = rng.choice([-0.08, -0.04, -0.02, -0.01, 0.01, 0.02, 0.04, 0.08], (batch_size,))
        # c_motion = rng.choice([-0.08, -0.04, -0.02, -0.01, 0.01, 0.02, 0.04, 0.08], (batch_size,))
        
        cue = rng.choice([0., 1.], (batch_size,))
    elif mode == 'test':
        
        batch_size = kwargs['batch_size']
        stim1_duration = (rng.uniform(800, 800, batch_size)/dt).astype(int)
        gamma_bar_color = kwargs['gamma_bar_color']
        gamma_bar_motion = kwargs['gamma_bar_motion']


        if not hasattr(gamma_bar_color, '__iter__'):
            print('__iter__')
            gamma_bar_color = np.array([gamma_bar_color] * batch_size)

        if not hasattr(gamma_bar_motion, '__iter__'):
            gamma_bar_motion = np.array([gamma_bar_motion] * batch_size)

        cue = kwargs['cue']
        if not hasattr(cue, '__iter__'):
            cue = np.array([cue] * batch_size)

        c_color = kwargs['c_color']
        if not hasattr(c_color, '__iter__'):
            c_color = np.array([c_color] * batch_size)

        c_motion = kwargs['c_motion']
        if not hasattr(c_color, '__iter__'):
            c_motion = np.array([c_motion] * batch_size)
        
        print('test',type(gamma_bar_color), gamma_bar_motion, c_color, c_motion)
        
    
    else:
        raise ValueError('Unknown mode: ' + str(mode))

    

    strength1_color = gamma_bar_color + c_color
    strength2_color = gamma_bar_color - c_color
    strength1_motion = gamma_bar_motion + c_motion
    strength2_motion = gamma_bar_motion - c_motion
    
    # print('3',batch_size,strength1_color, strength1_motion)

    # task structure:
    """
    stim1_on: 100ms
    stim1_duration: 800ms
    stim_delay: 800ms
    cue_duration: 800ms
    """
    if kwargs['noise_on']:
        stim1_on = (rng.uniform(100, 100, batch_size)/dt).astype(int)
    else:
        stim1_on = (rng.uniform(100, 100, batch_size)/dt).astype(int)

    if kwargs['noise_on']:
        stim_delay = (rng.uniform(800, 800, batch_size)/dt).astype(int)
    else:
        stim_delay = (rng.uniform(800, 800, batch_size)/dt).astype(int)

    # the offset time of the first stimulus
    stim1_off = stim1_on + stim1_duration
    cue_on = stim1_off + stim_delay
    cue_off = cue_duration+cue_on

    response_on = cue_off
    # response end time
    response_off = response_on + response_duration
    xtdim = response_off

    trial = Trial(config, xtdim.max(), batch_size)

    cue_color_input = 1.0 - cue
    cue_motion_input = cue

    # input 0
    trial.add('input', 0, ons=stim1_on, offs=stim1_off, strengths=strength1_color)
    # input 1
    trial.add('input', 1, ons=stim1_on, offs=stim1_off, strengths=strength2_color)
    # input 3
    trial.add('input', 2, ons=stim1_on, offs=stim1_off, strengths=strength1_motion)
    # input 4
    trial.add('input', 3, ons=stim1_on, offs=stim1_off, strengths=strength2_motion)

    trial.add('input', 4, ons=cue_on, offs=cue_off, strengths=cue_color_input)
    trial.add('input', 5, ons=cue_on, offs=cue_off, strengths=cue_motion_input)


    output_color_target1 = cue_color_input * (strength1_color > strength2_color)
    output_color_target2 = cue_color_input * (strength1_color <= strength2_color)

    output_motion_target1 = cue_motion_input * (strength1_motion > strength2_motion)
    output_motion_target2 = cue_motion_input * (strength1_motion <= strength2_motion)

    trial.add('out', 0, ons=response_on, offs=response_off, strengths=output_color_target1)
    trial.add('out', 1, ons=response_on, offs=response_off, strengths=output_color_target2)
    trial.add('out', 2, ons=response_on, offs=response_off, strengths=output_motion_target1)
    trial.add('out', 3, ons=response_on, offs=response_off, strengths=output_motion_target2)
    #target output
    trial.add('cost_mask', 0, ons=stim1_on, offs=response_off, strengths=trial.expand(1.))
    trial.add('cost_mask', 1, ons=stim1_on, offs=response_off, strengths=trial.expand(1.))
    trial.add('cost_mask', 2, ons=stim1_on, offs=response_off, strengths=trial.expand(1.))
    trial.add('cost_mask', 3, ons=stim1_on, offs=response_off, strengths=trial.expand(1.))

    # trial.epochs = {'stim':(stim1_on, stim1_off),
    #                 'stim_delay':(cue_off,stim1_on),
    #                 'integrate': (cue_on,cue_off),
    #                 'response': (response_on, response_off)}



    trial.epochs = {'stim':(stim1_on, stim1_off),
                    'stim_delay':(stim1_off,cue_on),
                    'integrate': (cue_on,cue_off),
                    'response': (response_on, response_off)}


    trial.stim1_duration = stim1_duration
    trial.strength1_color = strength1_color
    trial.strength2_color = strength2_color
    trial.strength1_motion = strength1_motion
    trial.strength2_motion = strength2_motion
    trial.cue = cue
    
    trial.c_color = c_color
    trial.c_motion = c_motion

    trial.seq_len = xtdim
    trial.max_seq_len = xtdim.max()

    return trial


if __name__ == "__main__":
    import seaborn as sns
    from matplotlib import pyplot as plt
    import default
    fs = 10

    print(sys.argv)


    rule_trains = 'contextInteg_decision_making'

    train_time_integrate = np.array([600, 1200])

    hp = default.get_default_hp(rule_trains)
    trial = generate_trials(rule_trains, hp, 'random', noise_on=True, batch_size=1)
def contextInteg_decision_making(config, mode, **kwargs):
    return _multisensory_output_cue_20_delay_40(config, mode, **kwargs)



# map string to functions
rule_mapping = {
                'contextInteg_decision_making': contextInteg_decision_making
                }


def generate_trials(rule, hp, mode, noise_on=False, **kwargs):
    """Generate one batch of data.

    Args:
        hp: dictionary of hyperparameter
        mode: str, the mode of generating. Options: random, test, psychometric
        noise_on: bool, whether input noise is given

    Return:
        trial: Trial class instance, containing input and target output
    """
    # print(rule)
    config = hp
    kwargs['noise_on'] = noise_on
    trial = rule_mapping[rule](config, mode, **kwargs)

    if noise_on:
        trial.add_x_noise()

    return trial

###rule###################################################################
    fig1 = plt.figure(figsize=(2.0, 1.0))
    ax1 = fig1.add_axes([0.2, 0.3, 0.7, 0.6])
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    print(trial.x.shape)

    rule_color = trial.x[:, :, 4]
    rule_motion = trial.x[:, :, 5]

    plt.plot(np.arange(0,120)*20,rule_color, linewidth='2', color='tab:blue', alpha=1)  # #1f77b4
    plt.plot(np.arange(0,120)*20,rule_motion, linewidth='2', color='orange', alpha=1)  # rule1
    plt.xticks([0,25*20,120 * 20])
    plt.yticks([0, 1], fontsize=fs)
    #plt.xlabel('Time (ms)', fontsize=fs + 1)
    plt.xlim(0,120 * 20)
    plt.ylim(-0.05, 1.1)
    ax1.set_facecolor(color='whitesmoke')
    fig1.savefig("../figure/Appendix/plot_rule.pdf")
###color stimulus###################################################################
    fig2 = plt.figure(figsize=(2.0, 1.0))
    ax2 = fig2.add_axes([0.2, 0.3, 0.7, 0.6])
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    print(trial.x.shape)

    color_red = trial.x[:, :, 0]
    color_green = trial.x[:, :, 1]
    motion_left = trial.x[:, :, 2]
    motion_right = trial.x[:, :, 3]

    plt.plot(np.arange(0, 120) * 20, motion_left, linewidth='1', color='goldenrod', alpha=1)  # rule1
    plt.plot(np.arange(0, 120) * 20, motion_right, linewidth='1', color='olive', alpha=1)  # rule1

    plt.plot(np.arange(0, 120) * 20, color_red, linewidth='1', color='green', alpha=1)  # rule1
    plt.plot(np.arange(0, 120) * 20, color_green, linewidth='1', color='red', alpha=0.8)  # rule1
    plt.xticks([])#0, 65 * 20, 120 * 20
    plt.yticks([0, 1.2], fontsize=fs)
    #plt.xlabel('Time (ms)', fontsize=fs + 1)
    ax2.set_facecolor(color='whitesmoke')
    plt.xlim(0, 120 * 20)
    plt.ylim(-0.05, 1.3)
    fig2.savefig("../figure/Appendix/plot_color_sti.pdf")
###motion stimulus###################################################################
    fig3 = plt.figure(figsize=(2.0, 1.0))
    ax3 = fig3.add_axes([0.2, 0.3, 0.7, 0.6])
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    print(trial.x.shape)

    motion_left = trial.x[:, :, 2]
    motion_right = trial.x[:, :, 3]

    plt.plot(np.arange(0, 120) * 20, motion_left, linewidth='1', color='goldenrod', alpha=1)  # rule1
    plt.plot(np.arange(0, 120) * 20, motion_right, linewidth='1', color='olive', alpha=1)  # rule1
    plt.xticks([])#0, 65 * 20, 120 * 20
    plt.yticks([0, 1.2], fontsize=fs)
    # plt.xlabel('Time (ms)', fontsize=fs + 1)
    ax3.set_facecolor(color='whitesmoke')
    plt.xlim(0, 120 * 20)
    plt.ylim(-0.05, 1.3)
    fig3.savefig("../figure/Appendix/plot_motion_sti.pdf")


###response###################################################################
    fig4 = plt.figure(figsize=(2.0, 1.0))
    ax4 = fig4.add_axes([0.2, 0.3, 0.7, 0.6])
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    print(trial.x.shape)

    response_0 = trial.y[:, 0, 0]
    response_1 = trial.y[:, 0, 1]
    response_2 = trial.y[:, 0, 2]
    response_3 = trial.y[:, 0, 3]

    plt.plot(np.arange(0, 120) * 20, response_0, linewidth='1', color='green', alpha=1)  # rule1
    plt.plot(np.arange(0, 120) * 20, response_1, linewidth='1', color='red', alpha=1)  # rule1
    plt.plot(np.arange(0, 120) * 20, response_2, linewidth='1', color='goldenrod', alpha=1)  # rule1
    plt.plot(np.arange(0, 120) * 20, response_3, linewidth='1', color='olive', alpha=1)  # rule1
    plt.xticks([0, 120 * 20])
    plt.yticks([0, 1.2], fontsize=fs)
    ax4.set_facecolor(color='whitesmoke')
    # plt.xlabel('Time (ms)', fontsize=fs + 1)
    plt.xlim(0, 120 * 20)
    plt.ylim(-0.08, 1.3)
    fig4.savefig("../figure/Appendix/plot_response.pdf")
    plt.show()

'''
    # input1
    fig = plt.figure()
    for i in range(len(trial.x[0, 0, :])):

        data = trial.x[:, :, i]
        # plt.plot(data, color='blue')
        # print("data:",data)
        if i == 0:
            plt.plot(data, color='royalblue')  #
        elif i == 1:
            plt.plot(data, color='darkblue')  #

        elif i == 2:
            plt.plot(data, color='violet')  #
        elif i == 3:
            plt.plot(data, color='purple')  #
        elif i == 4:
            plt.plot(data, linewidth='2', color='blue', label='cue_color_blue')  # rule1
        elif i == 5:
            plt.plot(data, linewidth='2', color='red',label='cue_motion_red')  ##rule2

    for i in range(len(trial.y[0, 0, :])):
        data = trial.y[:, 0, i]

        if i == 0:
            plt.plot(data, '--', color='royalblue',label='out_color_left')  #
        elif i == 1:
            plt.plot(data, '--', color='darkblue',label='out_color_right')  #
        elif i == 2:
            plt.plot(data, '--', color='violet',label='out_motion_left')  #
        elif i == 3:
            plt.plot(data, '--', color='purple',label='out_motion_right')  #
    #plt.legend()

    # # cost_mask
    # data = trial.cost_mask[:, 0, 0]
    # plt.plot(range(len(data)), data, color='black')
    plt.show()

'''