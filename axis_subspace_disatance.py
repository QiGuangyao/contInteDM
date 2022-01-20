#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 12 15:08:17 2021

@author: qiguangyao
"""

#%%

import errno
import numpy as np
import os
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.io as sio
import copy
from src.analysis import neural_activity_analysis
from src.analysis import state_space_analysis
from src.analysis import find_fixed_point
from figtools import Figure
from src import train

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

import sys
sys.path.append('/Users/qiguangyao/github/contextInteg/')

save_path = "/Users/qiguangyao/github/contextInteg/figure/"
model_path = 'cue_20_delay_40/model_' + '103' + '/finalResult/'
model_path = '/Users/qiguangyao/github/contextInteg/src/saved_model/contextInteg_decision_making/'

from pylab import mpl
mpl.rcParams['axes.unicode_minus'] = False
mpl.rcParams['font.sans-serif'] = ['SimHei'] # 指定默认字体
#%% functions
def calcDist(transform_data):
    """
    Parameters
    ----------
    transform_data : array [conditions*time*PCs]
        pca components for each conditions
        
    Returns
    -------
    
    distAarry: array [conditions*conditions*time]
        the distance between each compotents: 
    """
    distAarry = np.full((transform_data.shape[0],transform_data.shape[0],transform_data.shape[1]),np.nan)
    for t in range(distAarry.shape[2]):
        for c1 in range(distAarry.shape[0]):
            for c2 in range(distAarry.shape[0]):
                distAarry[c1,c2,t] = np.sqrt(sum(np.power((transform_data[c1,t,:] - transform_data[c2,t,:]), 2)))
    return distAarry

def calcAngl(vector, subspace):
    """
    Parameters
    ----------
    vector : vector (256)
        axis
    subspace : array (4*256)
        PCA-subpace.

    Returns
    -------
        the angle between axis and plane (PC1-PC2, PC1-PC3, PC2-PC3)
    """
    projVect = np.dot(subspace,vector)
    x = projVect[range(3)]
    
    angle_d = np.full((3),np.nan)
    for i in range(3):
        y = copy.deepcopy(x)
        y[i] = 0
        # y = projVect[range(2)]
        l_x=np.sqrt(x.dot(x))
        l_y=np.sqrt(y.dot(y))
        
        dian=x.dot(y)
        cos_=dian/(l_x*l_y)
        angle_hu=np.arccos(cos_)
        angle_d[i]=angle_hu*180/np.pi
    return angle_d

def return_neurActi_cont_cohe(trial_sample, run_step, rnn_network
        ):
    
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
    # get neurActi_cont_cohe
    """
    neural_activity_cue (120, 3600, 256)
    cohe [-0.04 , -0.02 , -0.01 , -0.005,  0. , 0.005, 0.01 ,  0.02 ,  0.04]
    cue [0,1]
    """
    c_color = trial_sample.c_color
    c_motion = trial_sample.c_motion
    cohe = np.unique(c_color)
    neurActi_cont_cohe = np.full((120,256,18),np.nan)# color -- []; motion -- []
    for i,j in enumerate(cohe):
        seleInde = [cu for cu in range(len(cue)) if (cue[cu] == 0 and c_color[cu] == j and actual_choices[cu] == target_choice[0][cu] and  fail_action[cu] == False)] 
        neurActi_cont_cohe[:,:,i] = np.nanmean(neural_activity_cue[:,seleInde,:],axis = 1)
        seleInde = [cu for cu in range(len(cue)) if (cue[cu] == 1 and c_motion[cu] == j and actual_choices[cu] == target_choice[0][cu] and fail_action[cu] == False)] 
        neurActi_cont_cohe[:,:,i+9] = np.nanmean(neural_activity_cue[:,seleInde,:],axis = 1)
    return neurActi_cont_cohe,actual_choices,target_choice,fail_action


def return_epoch_wise_subspace(
        start_projection,
        end_projection,
        neurActi_cont_cohe
        ):
    """
    input:
        start_projection
        end_projection
        neurActi_cont_cohe: times*neurons*conditions(color -- motion)
    output:
        pca, concate_transform_split,concate_neural_activity_transform,concate_neural_activity
    
    """
    
    concate_neural_activity = neurActi_cont_cohe[range(start_projection,end_projection),:,:]
    #reshape concate_neural_activity
    concate_neural_activity_resh = np.full((concate_neural_activity.shape[0]*concate_neural_activity.shape[2],concate_neural_activity.shape[1]),np.nan)
    for i in range(concate_neural_activity.shape[2]):
        concate_neural_activity_resh[range(concate_neural_activity.shape[0]*i,
                                           concate_neural_activity.shape[0]*(i+1)),:] = concate_neural_activity[:,:,i]
    # concate_neural_activity = concate_neural_activity.reshape(concate_neural_activity.shape[0]*concate_neural_activity.shape[2],
    #                                                           concate_neural_activity.shape[1])
    # state space
    pca = PCA()#n_components=4
    pca.fit(concate_neural_activity_resh)
    concate_neural_activity_transform = pca.transform(concate_neural_activity_resh)
    # time_size = end_projection - start_projection
    # delim = np.cumsum(time_size)
    # concate_transform_split = np.split(concate_neural_activity_transform, delim[:-1], axis=0)
    concate_transform_split = []
    
    for i in range(concate_neural_activity.shape[2]):
        concate_transform_split.append(concate_neural_activity_transform[range(concate_neural_activity.shape[0]*i,
                                           concate_neural_activity.shape[0]*(i+1)),:])
    
    return pca,concate_transform_split,concate_neural_activity_transform,concate_neural_activity
#%%
"""
cue: 0--color; 1--motion
noise_on: if True, then add noise to x.

choice:
    1: color_red
    2: color_green
    3: motion_left
    4: motion_right
"""
#-----subspace------
#stimulus
pca_stim_subs_coll = {}
concate_transform_split_stim_subs_coll = {}
concate_neural_activity_transform_stim_subs_coll = {}
concate_neural_activity_stim_subs_coll = {}
#stimulus_delay
pca_stim_delay_subs_coll = {}
concate_transform_split_stim_delay_subs_coll = {}
concate_neural_activity_transform_stim_delay_subs_coll = {}
concate_neural_activity_stim_delay_subs_coll = {}
#integration
pca_inte_subs_coll = {}
concate_transform_split_inte_subs_coll = {}
concate_neural_activity_transform_inte_subs_coll = {}
concate_neural_activity_inte_subs_coll = {}
#response
pca_resp_subs_coll = {}
concate_transform_split_resp_subs_coll = {}
concate_neural_activity_transform_resp_subs_coll = {}
concate_neural_activity_resp_subs_coll = {}
#integration_response
pca_inte_resp_subs_coll = {}
concate_transform_split_inte_resp_subs_coll = {}
concate_neural_activity_transform_inte_resp_subs_coll = {}
concate_neural_activity_inte_resp_subs_coll = {}


#_stimulus_delay_integration_response
pca_all_subs_coll = {}
concate_transform_split_all_subs_coll = {}
concate_neural_activity_transform_all_subs_coll = {}
concate_neural_activity_all_subs_coll = {}

#-----axis------
color_cue_axis_stimu_coll = {}
motion_cue_axis_stimu_coll = {}
color_cue_axis_integ_coll = {}
motion_cue_axis_integ_coll = {}

batch_size = 3600
noise_on = True
for mi in range(101,151):
    print(mi)
    rule_trains = "contextInteg_decision_making"
    model_serial_idx='cue_20_delay_40/model_' + str(mi) + '/finalResult'
    model_dir = './src/saved_model/' + rule_trains + '/' + str(model_serial_idx)
    trial_sample, run_step, rnn_network = train.Runner(
                            mode = 'random_test',
                            model_dir=model_dir, rule_trains='contextInteg_decision_making',
                            is_cuda=False, 
                            noise_on=noise_on,
                            batch_size=batch_size)
    
    
    
    neurActi_cont_cohe,actual_choices,target_choice,fail_action = return_neurActi_cont_cohe(trial_sample, run_step, rnn_network
            )
    #subspace
    neurActi_cont_cohe_sele = copy.deepcopy(neurActi_cont_cohe)
    #stimulus subspace
    start_projection = 5
    end_projection = 45
    pca_stim_subs_coll[mi],concate_transform_split_stim_subs_coll[mi],concate_neural_activity_transform_stim_subs_coll[mi],concate_neural_activity_stim_subs_coll[mi] = return_epoch_wise_subspace(
            start_projection = start_projection,
            end_projection = end_projection,
            neurActi_cont_cohe = neurActi_cont_cohe_sele)
    
    #stimulus_delay subspace
    start_projection = 5
    end_projection = 85
    pca_stim_delay_subs_coll[mi],concate_transform_split_stim_delay_subs_coll[mi],concate_neural_activity_transform_stim_delay_subs_coll[mi],concate_neural_activity_stim_delay_subs_coll[mi] = return_epoch_wise_subspace(
            start_projection = start_projection,
            end_projection = end_projection,
            neurActi_cont_cohe = neurActi_cont_cohe_sele)
    
    #integration subspace
    start_projection = 85
    end_projection = 105
    pca_inte_subs_coll[mi],concate_transform_split_inte_subs_coll[mi],concate_neural_activity_transform_inte_subs_coll[mi],concate_neural_activity_inte_subs_coll[mi] = return_epoch_wise_subspace(
            start_projection = start_projection,
            end_projection = end_projection,
            neurActi_cont_cohe = neurActi_cont_cohe_sele)
    
    #response subspace
    start_projection = 105
    end_projection = 120
    pca_resp_subs_coll[mi],concate_transform_split_resp_subs_coll[mi],concate_neural_activity_transform_resp_subs_coll[mi],concate_neural_activity_resp_subs_coll[mi] = return_epoch_wise_subspace(
            start_projection = start_projection,
            end_projection = end_projection,
            neurActi_cont_cohe = neurActi_cont_cohe_sele)
    
    #integration_response subspace
    start_projection = 85
    end_projection = 120
    pca_inte_resp_subs_coll[mi],concate_transform_split_inte_resp_subs_coll[mi],concate_neural_activity_transform_inte_resp_subs_coll[mi],concate_neural_activity_inte_resp_subs_coll[mi] = return_epoch_wise_subspace(
            start_projection = start_projection,
            end_projection = end_projection,
            neurActi_cont_cohe = neurActi_cont_cohe_sele)

    #all subspace
    start_projection = 0
    end_projection = 120
    pca_all_subs_coll[mi],concate_transform_split_all_subs_coll[mi],concate_neural_activity_transform_all_subs_coll[mi],concate_neural_activity_all_subs_coll[mi] = return_epoch_wise_subspace(
            start_projection = start_projection,
            end_projection = end_projection,
            neurActi_cont_cohe = neurActi_cont_cohe_sele)
    # stimulus_axis
    start_projection = 5
    end_projection = 45
    #color
    neurActi_cont_cohe_sele_color = copy.deepcopy(neurActi_cont_cohe[:,:,range(9)])
    color_cue_axis_stimu_coll[mi], _,_,_ = return_epoch_wise_subspace(
            start_projection = start_projection,
            end_projection = end_projection,
            neurActi_cont_cohe = neurActi_cont_cohe_sele_color)
    #motion
    neurActi_cont_cohe_sele_motion = copy.deepcopy(neurActi_cont_cohe[:,:,range(9,18)])
    motion_cue_axis_stimu_coll[mi], _,_,_ = return_epoch_wise_subspace(
            start_projection = start_projection,
            end_projection = end_projection,
            neurActi_cont_cohe = neurActi_cont_cohe_sele_motion)
    #choice_axis
    start_projection = 85
    end_projection = 105
    #color
    neurActi_cont_cohe_sele_color = copy.deepcopy(neurActi_cont_cohe[:,:,range(9)])
    color_cue_axis_integ_coll[mi], _,_,_ = return_epoch_wise_subspace(
            start_projection = start_projection,
            end_projection = end_projection,
            neurActi_cont_cohe = neurActi_cont_cohe_sele_color)
    #motion
    neurActi_cont_cohe_sele_motion = copy.deepcopy(neurActi_cont_cohe[:,:,range(9,18)])
    motion_cue_axis_integ_coll[mi], _,_,_ = return_epoch_wise_subspace(
            start_projection = start_projection,
            end_projection = end_projection,
            neurActi_cont_cohe = neurActi_cont_cohe_sele_motion)
#%% stimulus subspcae; stimulus-delay subspcae; integration subspace
pca_coll = copy.deepcopy(pca_inte_resp_subs_coll)
concate_transform_split_coll = copy.deepcopy(concate_transform_split_inte_resp_subs_coll)
color_cue_axis_stimu_angl = []
motion_cue_axis_stimu_angl = []
motion_cue_axis_integ_angl = []
color_cue_axis_integ_angl = []
for i in range(101,151):
    print(i)
    color_cue_axis_stimu_angl.append(calcAngl(color_cue_axis_stimu_coll[i].components_[0], pca_coll[i].components_[range(4),:])[0])
    motion_cue_axis_stimu_angl.append(calcAngl(motion_cue_axis_stimu_coll[i].components_[0], pca_coll[i].components_[range(4),:])[0])
    motion_cue_axis_integ_angl.append(calcAngl(motion_cue_axis_integ_coll[i].components_[0], pca_coll[i].components_[range(4),:])[0])
    color_cue_axis_integ_angl.append(calcAngl(color_cue_axis_integ_coll[i].components_[0], pca_coll[i].components_[range(4),:])[0])

#%%
bottom = 0
# max_height = 50
fig = plt.figure(figsize=(4*3, 3.5))#figsize = [6,2]
theta = np.radians(color_cue_axis_stimu_angl)
# N = len(theta)
# radii = 20 * np.random.rand(N)
# width = np.pi / 4 * np.random.rand(N)

ax = plt.subplot(141, polar=True)
bins = 10
count, bins = np.histogram(theta, bins=bins)
ax.bar(bins[1:]-(bins[1]-bins[0])/2,count, width = (bins[1]-bins[0]),bottom=bottom)
ax.set_title('C-stimu-axis')

ax = plt.subplot(142, polar=True)
theta = np.radians(motion_cue_axis_stimu_angl)
bins = 10
count, bins = np.histogram(theta, bins=bins)
ax.bar(bins[1:]-(bins[1]-bins[0])/2,count, width = (bins[1]-bins[0]),bottom=bottom)
ax.set_title('M-stimu-axis')

ax = plt.subplot(143, polar=True)
theta = np.radians(motion_cue_axis_integ_angl)
bins = 10
count, bins = np.histogram(theta, bins=bins)
ax.bar(bins[1:]-(bins[1]-bins[0])/2,count, width = (bins[1]-bins[0]),bottom=bottom)
ax.set_title('C-choice-axis')

ax = plt.subplot(144, polar=True)
theta = np.radians(color_cue_axis_integ_angl)
bins = 10
count, bins = np.histogram(theta, bins=bins)
ax.bar(bins[1:]-(bins[1]-bins[0])/2,count, width = (bins[1]-bins[0]),bottom=bottom)
ax.set_title('M-choice-axis')
# # Use custom colors and opacity
# for r, bar in zip(radii, bars):
#     bar.set_facecolor('#1f77b4')
#     bar.set_alpha(0.8)
plt.tight_layout()
plt.savefig(save_path+"axis_angl_inte_resp_subspace.pdf",dpi = 300)
plt.show()
#%%
#plot
# inpuAxis = color_cue_axis_stimu
fig = plt.figure(figsize=(4, 3.5))
ax = fig.add_axes([0.15, 0.15, 0.8, 0.8])
_alpha_list_1=[0.1,0.2,0.4,0.6,0.8,0.1,0.2,0.4,0.6,0.8,0.1,0.2,0.4,0.6,0.8,0.1,0.2,0.4,0.6,0.8]
basecolor = Figure.colors('#1f77b4')
c_motions = np.array([ -0.02, -0.04, -0.06, -0.08,0.02, 0.04, 0.06, 0.08])
batch_size = len(c_motions)
color = '#1f77b4'
k0=0
k1=1
ind = 102
scalInd = 10
##projection
fs = 12 # font size
ax.plot([-scalInd*10 * np.sum(pca_coll[ind].components_[k0] * color_cue_axis_stimu_coll[ind].components_[0]), scalInd*10 * np.sum(pca_coll[ind].components_[k0] * color_cue_axis_stimu_coll[ind].components_[0])],
        [-scalInd*10 * np.sum(pca_coll[ind].components_[k1] * color_cue_axis_stimu_coll[ind].components_[0]), scalInd*10 * np.sum(pca_coll[ind].components_[k1] * color_cue_axis_stimu_coll[ind].components_[0])],
        '-', color='steelblue', alpha=0.8, linewidth=2.5)
ax.plot([-scalInd*20 * np.sum(pca_coll[ind].components_[k0] * motion_cue_axis_stimu_coll[ind].components_[0]), scalInd*20 * np.sum(pca_coll[ind].components_[k0] * motion_cue_axis_stimu_coll[ind].components_[0])],
        [-scalInd*20 * np.sum(pca_coll[ind].components_[k1] * motion_cue_axis_stimu_coll[ind].components_[0]), scalInd*20 * np.sum(pca_coll[ind].components_[k1] * motion_cue_axis_stimu_coll[ind].components_[0])],
        '-', color='grey', alpha=0.8, linewidth=2.5)

ax.plot([-scalInd*2 * np.sum(pca_coll[ind].components_[k0] * color_cue_axis_integ_coll[ind].components_[0]), scalInd*2 * np.sum(pca_coll[ind].components_[k0] * color_cue_axis_integ_coll[ind].components_[0])],
        [-scalInd*2 * np.sum(pca_coll[ind].components_[k1] * color_cue_axis_integ_coll[ind].components_[0]), scalInd*2 * np.sum(pca_coll[ind].components_[k1] * color_cue_axis_integ_coll[ind].components_[0])],
        '--', color='steelblue', alpha=0.8, linewidth=2.5)
ax.plot([-scalInd*2 * np.sum(pca_coll[ind].components_[k0] * motion_cue_axis_integ_coll[ind].components_[0]), scalInd*2 * np.sum(pca_coll[ind].components_[k0] * motion_cue_axis_integ_coll[ind].components_[0])],
        [-scalInd*2 * np.sum(pca_coll[ind].components_[k1] * motion_cue_axis_integ_coll[ind].components_[0]), scalInd*2 * np.sum(pca_coll[ind].components_[k1] * motion_cue_axis_integ_coll[ind].components_[0])],
        '--', color='grey', alpha=0.8, linewidth=2.5)

#plot
colormap =plt.cm.RdBu
    # colors = np.array([colormap(i) for i in np.linspace(0, 1, len(projected))])
    
colors = np.array([colormap(i) for i in np.linspace(0, 1, 9)])
for i in range(0, len(concate_transform_split_coll[ind])):
    if i<9:
        # print(1111)
        ax.plot(concate_transform_split_coll[ind][i][:, k0], concate_transform_split_coll[ind][i][:, k1],linewidth='1.8',color=colors[i])#,alpha=_alpha_list_1[i])
        ax.scatter(concate_transform_split_coll[ind][i][0, k0], concate_transform_split_coll[ind][i][0, k1], s=40,marker='*', color=colors[i])#,alpha=_alpha_list_1[i])
        ax.scatter(concate_transform_split_coll[ind][i][-1, k0], concate_transform_split_coll[ind][i][-1, k1], s=40,marker='o', color=colors[i])#,alpha=_alpha_list_1[i])
    else:
        print(i)
        ax.plot(concate_transform_split_coll[ind][i][:, k0], concate_transform_split_coll[ind][i][:, k1],'--',linewidth='1.8',color=colors[i-9])#,alpha=_alpha_list_1[i])
        ax.scatter(concate_transform_split_coll[ind][i][0, k0], concate_transform_split_coll[ind][i][0, k1], s=40,marker='*', color=colors[i-9])#,alpha=_alpha_list_1[i])
        ax.scatter(concate_transform_split_coll[ind][i][-1, k0], concate_transform_split_coll[ind][i][-1, k1], s=40,marker='o', color=colors[i-9])#,alpha=_alpha_list_1[i])

# ax.set_xlabel('cue-PC1', fontsize=fs+3)
# ax.set_ylabel('cue-PC2', fontsize=fs+3)

ax.set_xlabel('PC1', fontsize=fs+3)
ax.set_ylabel('PC2', fontsize=fs+3)

# ax.set_xticks([])
# ax.set_yticks([])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(False)
plt.tight_layout()
# plt.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
plt.savefig(save_path+"PCA_plot_2D_inte_resp_PC1_PC2.pdf",dpi = 300)
plt.show()
#%%
fig = plt.figure(figsize = [5,3.45])
ax = fig.gca(projection='3d')
k2 = 0
k2 = 1
k2 = 2
scalIndex = 10
colormap =plt.cm.RdBu
# colors = np.array([colormap(i) for i in np.linspace(0, 1, len(projected))])
colors = np.array([colormap(i) for i in np.linspace(0, 1, 9)])
#plot
for i in range(0, len(concate_transform_split_coll[ind])):
    print(i)
    if i<9:
        ax.plot(concate_transform_split_coll[ind][i][:, k0], concate_transform_split_coll[ind][i][:, k1],
                concate_transform_split_coll[ind][i][:, k2],
                linewidth='2',color=colors[i])#,alpha=_alpha_list_1[i])
        ax.scatter(concate_transform_split_coll[ind][i][0, k0], concate_transform_split_coll[ind][i][0, k1],
                   concate_transform_split_coll[ind][i][0, k2],
                   s=20,marker='*', color=colors[i])#,alpha=_alpha_list_1[i])
        ax.scatter(concate_transform_split_coll[ind][i][-1, k0], concate_transform_split_coll[ind][i][-1, k1],
                   concate_transform_split_coll[ind][i][-1, k2],
                   s=20,marker='o', color=colors[i])#,alpha=_alpha_list_1[i])
    else:
        ax.plot(concate_transform_split_coll[ind][i][:, k0], concate_transform_split_coll[ind][i][:, k1],
                concate_transform_split_coll[ind][i][:, k2],'--',
                linewidth='2',color=colors[i-9])#,alpha=_alpha_list_1[i])
        ax.scatter(concate_transform_split_coll[ind][i][0, k0], 
                   concate_transform_split_coll[ind][i][0, k1], 
                   concate_transform_split_coll[ind][i][0, k2],
                   s=10,marker='*', color=colors[i-9])#,alpha=_alpha_list_1[i])
        ax.scatter(concate_transform_split_coll[ind][i][-1, k0], 
                   concate_transform_split_coll[ind][i][-1, k1], 
                   concate_transform_split_coll[ind][i][-1, k2], 
                   s=10,marker='o', color=colors[i-9])#,alpha=_alpha_list_1[i])


    ax.plot([-scalIndex * np.sum(pca_coll[ind].components_[k0] * color_cue_axis_stimu_coll[ind].components_[0]), scalIndex * np.sum(pca_coll[ind].components_[k0] * color_cue_axis_stimu_coll[ind].components_[0])],
        [-scalIndex * np.sum(pca_coll[ind].components_[k1] * color_cue_axis_stimu_coll[ind].components_[0]), scalIndex * np.sum(pca_coll[ind].components_[k1] * color_cue_axis_stimu_coll[ind].components_[0])],
        [-scalIndex * np.sum(pca_coll[ind].components_[k2] * color_cue_axis_stimu_coll[ind].components_[0]), scalIndex * np.sum(pca_coll[ind].components_[k2] * color_cue_axis_stimu_coll[ind].components_[0])],
        '-', color='steelblue', alpha=0.8, linewidth=2.5)
    ax.plot([-scalIndex * np.sum(pca_coll[ind].components_[k0] * motion_cue_axis_stimu_coll[ind].components_[0]), scalIndex * np.sum(pca_coll[ind].components_[k0] * motion_cue_axis_stimu_coll[ind].components_[0])],
        [-scalIndex * np.sum(pca_coll[ind].components_[k1] * motion_cue_axis_stimu_coll[ind].components_[0]), scalIndex * np.sum(pca_coll[ind].components_[k1] * motion_cue_axis_stimu_coll[ind].components_[0])],
        [-scalIndex * np.sum(pca_coll[ind].components_[k2] * motion_cue_axis_stimu_coll[ind].components_[0]), scalIndex * np.sum(pca_coll[ind].components_[k2] * motion_cue_axis_stimu_coll[ind].components_[0])],
        '-', color='grey', alpha=0.8, linewidth=2.5)
    
    ax.plot([-scalIndex * np.sum(pca_coll[ind].components_[k0] * color_cue_axis_integ_coll[ind].components_[0]), scalIndex * np.sum(pca_coll[ind].components_[k0] * color_cue_axis_integ_coll[ind].components_[0])],
            [-scalIndex * np.sum(pca_coll[ind].components_[k1] * color_cue_axis_integ_coll[ind].components_[0]), scalIndex * np.sum(pca_coll[ind].components_[k1] * color_cue_axis_integ_coll[ind].components_[0])],
            [-scalIndex * np.sum(pca_coll[ind].components_[k2] * color_cue_axis_integ_coll[ind].components_[0]), scalIndex * np.sum(pca_coll[ind].components_[k2] * color_cue_axis_integ_coll[ind].components_[0])],
            '--', color='steelblue', alpha=0.8, linewidth=2.5)
    ax.plot([-scalIndex * np.sum(pca_coll[ind].components_[k0] * motion_cue_axis_integ_coll[ind].components_[0]), scalIndex * np.sum(pca_coll[ind].components_[k0] * motion_cue_axis_integ_coll[ind].components_[0])],
            [-scalIndex * np.sum(pca_coll[ind].components_[k1] * motion_cue_axis_integ_coll[ind].components_[0]), scalIndex * np.sum(pca_coll[ind].components_[k1] * motion_cue_axis_integ_coll[ind].components_[0])],
            [-scalIndex * np.sum(pca_coll[ind].components_[k2] * motion_cue_axis_integ_coll[ind].components_[0]), scalIndex * np.sum(pca_coll[ind].components_[k2] * motion_cue_axis_integ_coll[ind].components_[0])],
            '--', color='grey', alpha=0.8, linewidth=2.5)

ax.set_xlabel('stim_PC1')
ax.set_ylabel('stim_PC2')
ax.set_zlabel('stim_PC3')
# ax.set_xticks([-10,-5,0,5,10],[str(-10),-5,0,5,10])
# ax.set_ylabel('PC2')
# ax.set_zlabel('PC3')
plt.tight_layout()
# plt.savefig(save_path+"PCA_plot_3D_5_45_N.pdf",dpi = 300)

plt.show()
#%%
# indeNum = 50
# transform_data_test = np.full((len(concate_transform_split),concate_transform_split[0].shape[0],concate_transform_split[0].shape[1],indeNum),np.nan)
# for k in range(len(concate_transform_split)):
#     transform_data_test[k,:,:] = concate_transform_split[k]
# #%%

# sns.set_theme()
# distAarryTest = calcDist(transform_data_test)
# # for i in range(distAarryTest.shape[2]):
#     # plt.imshow(distAarryTest[:,:,i])
# # with plt.style.context('/Users/qiguangyao/neuralDataAnalysisProtocolData/style_paper.mplstyle'):
# sns.set_style("ticks", {"xtick.major.size": .01, "ytick.major.size": .01})
# # f, axs = plt.subplots(ncols=3, nrows=2, sharey=True,sharex=True,figsize=[4,3.54/2*2.5])
# f, axs = plt.subplots(ncols=1, nrows=1, sharey=True,sharex=True,figsize=[3.8,3.54])
# print(i)
# # sns.set(font_scale=5)
# # f, ax = plt.subplots(figsize=(9/1.5, 6/2))


# sns.heatmap(distAarryTest[:,:,i], 
#             # annot=True,
#             # cbar_kws={"fontsize":[5]},
#             # xticklabels = [-0.08, -0.04, -0.02, -0.01, 0.01, 0.02, 0.04, 0.08,-0.08, -0.04, -0.02, -0.01, 0.01, 0.02, 0.04, 0.08],
#             xticklabels = [-0.08,None ,None , None, None,None ,None , 0.08,-0.08,None ,None , None, None,None ,None , 0.08,],
#             yticklabels = [-0.08,None ,None , None, None,None ,None , 0.08,-0.08,None ,None , None, None,None ,None , 0.08,],
#             # yticklabels = [-0.08, -0.04, -0.02, -0.01, 0.01, 0.02, 0.04, 0.08,-0.08, -0.04, -0.02, -0.01, 0.01, 0.02, 0.04, 0.08],
#             square=True,
#             )

# # plt.xticks([i for i in range(16)], [-0.08, -0.04, -0.02, -0.01, 0.01, 0.02, 0.04, 0.08,-0.08, -0.04, -0.02, -0.01, 0.01, 0.02, 0.04, 0.08])
# # plt.yticks([i for i in range(16)], [-0.08, -0.04, -0.02, -0.01, 0.01, 0.02, 0.04, 0.08,-0.08, -0.04, -0.02, -0.01, 0.01, 0.02, 0.04, 0.08],rotation =90)
# # ax.set_xticks([])
# # ax.set_xticklabels([])
# # ax.set_yticks([])
# # ax.set_yticklabels([])
# # plt.xtick.major.size : 1.5
# # plt.ytick.major.size : 1.5
# plt.xticks(rotation=0,fontsize = 5)

# plt.tick_params(length=1)
# plt.yticks(fontsize = 5)
# cbar = axs.collections[0].colorbar
# cbar.set_label('Distance')
# # cbar.set_ticks(fontsize = 5)
# # cbar.set_tick_params(labelsize=5)
# plt.savefig(save_path+"distance_3D_5_45.pdf",dpi = 300)
# plt.show()
# #%%
# import seaborn as sns; sns.set_theme(color_codes=True)
# iris = sns.load_dataset("iris")
# species = iris.pop("species")
# g = sns.clustermap(distAarryTest[:,:,i])




