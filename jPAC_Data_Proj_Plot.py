#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  2 16:06:29 2022

@author: qiguangyao
"""

#%% import data
import errno
import numpy as np
import os

import matplotlib.pyplot as plt
from pylab import mpl
from src import train
#% jPCA
import jPCA
# from jPCA.util import load_churchland_data, plot_projections
import sys
import random
import copy
#%%
def mkdir_p(path):
    """
    Portable mkdir -p

    """
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise
mkdir_p("../figure")
save_path = "../figure/"
sys.path.append('/Users/qiguangyao/github/contextInteg/')
save_path = "/Users/qiguangyao/github/contextInteg/figure/"
model_path = '/Users/qiguangyao/github/contextInteg/src/saved_model/contextInteg_decision_making/'
mpl.rcParams['axes.unicode_minus'] = False
mpl.rcParams['font.sans-serif'] = ['SimHei'] # 指定默认字体
#%%
# for mi in range(1,51):
"""
cue: 0--color; 1--motion
noise_on: if True, then add noise to x.

choice:
    1: color_red
    2: color_green
    3: motion_left
    4: motion_right
"""
mi = 103
rule_trains = "contextInteg_decision_making"
model_serial_idx='cue_20_delay_40/model_' + str(mi) + '/finalResult'
model_dir = './src/saved_model/' + rule_trains + '/' + str(model_serial_idx)
batch_size = 3600
noise_on = True
trial_sample, run_step, rnn_network = train.Runner(
                        mode = 'random_test',
                        model_dir=model_dir, rule_trains='contextInteg_decision_making',
                        is_cuda=False, 
                        noise_on=noise_on,
                        batch_size=batch_size)
neural_activity_cue = run_step.activity_shaped.detach().cpu().numpy()
output = run_step.outputs.detach().cpu().numpy()#real output (120, batch_size, 4)
# hp = tools.load_hp(model_dir, rule_trains)
#% get correct response
#------target choice------
#target_choice = [1,2,3,4]
cue = trial_sample.cue#[0,1]=[left,right]
strength1_color = trial_sample.strength1_color
strength2_color = trial_sample.strength2_color
strength1_motion = trial_sample.strength1_motion
strength2_motion = trial_sample.strength2_motion

for i in range(len(strength1_color)):
    if strength1_color[i] - strength2_color[i] == 0:
        
        strength1_color[i] = strength1_color[i]+np.random.choice((-1,1))

    if strength1_motion[i] - strength2_motion[i] == 0:
        
        strength1_motion[i] = strength1_motion[i]+np.random.choice((-1,1))
        
target_choice = (1 - cue) * [1 * (strength1_color > strength2_color) + 2 * (
            strength1_color <= strength2_color)] \
                + cue * [3 * (strength1_motion > strength2_motion) + 4 * (
            strength1_motion <= strength2_motion)]  #

#------actural choice------
fix_start = trial_sample.epochs['stim'][0]
fix_end = trial_sample.epochs['integrate'][1]
fix_strength=0.2
action_threshold=0.5
response_duration=int(300/20)
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
actual_choices = np.array(actual_choices)
#%% get neurActi_cont_cohe
"""
neural_activity_cue (120, 3600, 256)
cohe [-0.04 , -0.02 , -0.01 , -0.005,  0. , 0.005, 0.01 ,  0.02 ,  0.04]
cue [0,1]
"""
c_color = trial_sample.c_color
c_motion = trial_sample.c_motion
cohe = np.unique(c_color)
neurActi_cont_cohe = np.full((120,256,18),np.nan)

for i,j in enumerate(cohe):
    seleInde = [cu for cu in range(len(cue)) if (cue[cu] == 0 and c_color[cu] == j and actual_choices[cu] == target_choice[0][cu] and  fail_action[cu] == False)] 
    neurActi_cont_cohe[:,:,i] = np.nanmean(neural_activity_cue[:,seleInde,:],axis = 1)
    seleInde = [cu for cu in range(len(cue)) if (cue[cu] == 1 and c_motion[cu] == j and actual_choices[cu] == target_choice[0][cu] and fail_action[cu] == False)] 
    neurActi_cont_cohe[:,:,i+9] = np.nanmean(neural_activity_cue[:,seleInde,:],axis = 1)

#shuffle trial
neurActi_cont_cohe_shuf = np.full((120,256,18),np.nan)
indexTemp = [i for i in range(neural_activity_cue.shape[1])]
shuffleIndex = random.sample(indexTemp,len(indexTemp))

neural_activity_cue_shuf = neural_activity_cue[:,shuffleIndex,:]

for i,j in enumerate(cohe):
    seleInde = [cu for cu in range(len(cue)) if (cue[cu] == 0 and c_color[cu] == j and actual_choices[cu] == target_choice[0][cu] and  fail_action[cu] == False)] 
    neurActi_cont_cohe_shuf[:,:,i] = np.nanmean(neural_activity_cue_shuf[:,seleInde,:],axis = 1)
    seleInde = [cu for cu in range(len(cue)) if (cue[cu] == 1 and c_motion[cu] == j and actual_choices[cu] == target_choice[0][cu] and fail_action[cu] == False)] 
    neurActi_cont_cohe_shuf[:,:,i+9] = np.nanmean(neural_activity_cue_shuf[:,seleInde,:],axis = 1)
#%% jPCA
# Create a jPCA object
jpca = jPCA.JPCA(num_jpcs=6)
datas = [neurActi_cont_cohe[:,:,i] for i in range(18)]

times = [i for i in range(120)]
# Fit the jPCA object to data
(projected, 
 full_data_var,
 pca_var_capt,
 jpca_var_capt,
 jpca_, 
 processed_datas) = jpca.fit(datas, times=times, tstart=0, tend=119)


datas_shuf = [neurActi_cont_cohe_shuf[:,:,i] for i in range(18)]
# projected_shuf, jpca_var_capt_shuf = jpca.project(self,datas_shuf)
(projected_shuf, 
 full_data_var_shuf,
 pca_var_capt_shuf,
 jpca_var_capt_shuf,
 jpca_shuf, 
 processed_datas_shuf) = jpca.fit(datas_shuf, times=times, tstart=0, tend=119)

# projected_shuf, jpca_var_capt_shuf = jpca.project(jpca_,processed_datas_shuf)

projected_shuf = processed_datas_shuf @ jpca_
jpca_var_capt_shuf = np.var(projected_shuf, axis=(0,1))
projected_shuf = [x for x in projected_shuf]
#%% plot jPCs
s1 = 5
arrow_size=0.05/5#shuffle: arrow_size=0.05/5; normal: arrow_size=0.05
circle_size=0.05/2
# color="black"
outline="black"

inpuData = copy.deepcopy(projected_shuf)
f, ax1 = plt.subplots(ncols=1, nrows=1, sharey=True,figsize=[3.54/2,3.54/2])
colormap =plt.cm.RdBu
# colors = np.array([colormap(i) for i in np.linspace(0, 1, len(projected))])
colors = np.array([colormap(i) for i in np.linspace(0, 1, 9)])
# colors = np.array([colormap(i) for i in np.linspace(0, 1, 9)])
seleInd1= 2
seleInd2= 3

for i in range(9):
    ax1.plot(inpuData[i][:,seleInd1],inpuData[i][:,seleInd2], '-',color = colors[i],label = 'color: '+str(cohe[i]))
    ax1.plot(inpuData[i][0,seleInd1],inpuData[i][0,seleInd2], 'o',color = colors[i],markersize = circle_size)
    ax1.plot(inpuData[i][5,seleInd1],inpuData[i][5,seleInd2], '*',color = colors[i],markersize = circle_size*20)
    ax1.plot(inpuData[i][45,seleInd1],inpuData[i][45,seleInd2], '*',color = colors[i],markersize = circle_size*20)
    ax1.plot(inpuData[i][85,seleInd1],inpuData[i][85,seleInd2], '^',color = colors[i],markersize = circle_size*20)
    ax1.plot(inpuData[i][105,seleInd1],inpuData[i][105,seleInd2], 's',color = colors[i],markersize = circle_size*20)
    x = inpuData[i][:, seleInd1]
    y = inpuData[i][:, seleInd2]
    dx = x[-1] - x[-2]
    dy = y[-1] - y[-2]
    px, py = (x[-1], y[-1])
    ax1.arrow(px, 
              py , 
              dx, 
              dy, 
                facecolor=colors[i], 
                edgecolor=colors[i],
                length_includes_head=True,
                head_width=arrow_size
              )        
for i in range(9):
    ax1.plot(inpuData[i+9][:,seleInd1],inpuData[i+9][:,seleInd2], '--',color = colors[i])#,label = 'motion: '+str(cohe[i]))
    ax1.plot(inpuData[i+9][0,seleInd1],inpuData[i+9][0,seleInd2], 'o',color = colors[i],markersize = circle_size)
    ax1.plot(inpuData[i+9][5,seleInd1],inpuData[i+9][5,seleInd2], '*',color = colors[i],markersize = circle_size*20)
    ax1.plot(inpuData[i+9][45,seleInd1],inpuData[i+9][45,seleInd2], '*',color = colors[i],markersize = circle_size*20)
    ax1.plot(inpuData[i+9][85,seleInd1],inpuData[i+9][85,seleInd2], '^',color = colors[i],markersize = circle_size*20)
    ax1.plot(inpuData[i+9][105,seleInd1],inpuData[i+9][105,seleInd2], 's',color = colors[i],markersize = circle_size*20)

    x = inpuData[i+9][:, seleInd1]
    y = inpuData[i+9][:, seleInd2]
    dx = x[-1] - x[-2]
    dy = y[-1] - y[-2]
    px, py = (x[-1], y[-1])
    ax1.arrow(px, py , dx, dy, 
                facecolor=colors[i], 
                edgecolor=colors[i],#outline,
                length_includes_head=True,
                head_width=arrow_size
              )
plt.xlabel('Proj. onto jPC' +str(seleInd1+1))
plt.ylabel('Proj. onto jPC'+str(seleInd2+1))
# plt.title('model'+str(mi))
# plt.legend(ncol= 1,bbox_to_anchor=[0,1.3])
fileName = save_path+'model'+str(mi)+'jPCA_'+str(seleInd1+1)+'_'+str(seleInd2+1)+'_shuf.pdf'
plt.savefig(fileName,dpi = 600)
plt.show()
#%%










