# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 12:43:25 2023

@author: bcary

Code to perform feature extraction on pose variables derived form Deeplabcut

TODO:
-consider data processing that will handle frames where DLC incorrectly labels pose and it jumps far away


"""



import numpy as np
import math
import os

import pandas as pd

from itertools import combinations

from matplotlib import pyplot as plt

# Add in a smoothing function in case I want to use it later
# https://stackoverflow.com/questions/5515720/python-smooth-time-series-data
def smooth(x,window_len=11,window='hanning'):
        if x.ndim != 1:
                raise ValueError("smooth only accepts 1 dimension arrays.")
        if x.size < window_len:
                raise ValueError("Input vector needs to be bigger than window size.")
        if window_len<3:
                return x
        if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
                raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")
        s=np.r_[2*x[0]-x[window_len-1::-1],x,2*x[-1]-x[-1:-window_len:-1]]
        if window == 'flat': #moving average
                w=np.ones(window_len,'d')
        else:  
                w=eval('np.'+window+'(window_len)')
        y=np.convolve(w/w.sum(),s,mode='same')
        return y[window_len:-window_len+1]

def angle_between_points(p1, p2):
    d1 = p2[0] - p1[0]
    d2 = p2[1] - p1[1]
    if d1 == 0:
        if d2 == 0:  # same points?
            deg = 0
        else:
            deg = 0 if p1[1] > p2[1] else 180
    elif d2 == 0:
        deg = 90 if p1[0] < p2[0] else 270
    else:
        deg = math.atan(d2 / d1) / math.pi * 180
        lowering = p1[1] < p2[1]
        if (lowering and deg < 0) or (not lowering and deg > 0):
            deg += 270
        else:
            deg += 90
    return deg


# # %matplotlib qt

# import yaml

# config_path = r"D:\Work\DATA\EleniB\scn2a_habit_config_BehCoder.yaml"

# # parses yaml file into config dictionary
# with open(config_path, 'r') as stream:
#     try:
#         config = yaml.safe_load(stream)
        
#     except yaml.YAMLError as exc:
#         print(exc)

# fps = config['fps']
# dlc_params = {
#     'xlims': config['xlims'],
#     'ylims': config['ylims'],
#     'fps': config['fps'],
#     'dlc_conf_thresh': config['dlc_conf_thresh'],
#     'dlc_interp_factor': config['dlc_interp_factor'],
#     'interp_fill_lim': config['interp_fill_lim'],
#     'dlc_smooth_ON': config['dlc_smooth_ON'],
#     'dlc_sm_sec': config['dlc_sm_sec'],
#     'no_abs_pose': config['no_abs_pose'],
#     'pose_to_exclude': config['pose_to_exclude'],
#     'head_body_norm': config['head_body_norm']
#     }
# kwargs = dlc_params

# # dlc_path = r"D:\Work\DATA\EleniB\SCN2A_raw_videos_labelsAndbouts_DLCfiles\EB15\EB15_D1_HS1DLC_resnet50_WT_vs_SCN2A_res1280x720Feb7shuffle1_1640000.csv"
# dlc_path = r"D:\Work\DATA\EleniB\acclim_train_data\EB14_D1_2immobcricksDLC_resnet50_WT_vs_SCN2A_res1280x720Feb7shuffle1_1640000.csv"

def extract_features(dlc_path, **kwargs):
    
    print('Running feature extraction...')
    # params used for automated pursuit detection and dlc pose extraction

    conf_thresh = kwargs.get('dlc_conf_thresh',0.7)
    smooth_ON = kwargs.get('dlc_smooth_ON',False)
    dlc_sm_sec = kwargs.get('dlc_sm_sec',0.5)
    no_abs_pose = kwargs.get('no_abs_pose',True)
    pose_to_exclude = kwargs.get('pose_to_exclude',[])
    fps = kwargs.get('fps',20)
    head_body_norm = kwargs.get('head_body_norm',[])
    
    interp_factor = int(kwargs.get('dlc_interp_factor',1))
    if interp_factor > 1:
        interp_ON = True
    else:
        interp_ON = False
    
    smooth_wind = int(fps*dlc_sm_sec*interp_factor)
    interp_fill_lim_sec = kwargs.get('interp_fill_lim',0.5)
    interp_fill_lim = int(interp_fill_lim_sec*fps) # was 4 for first round
    verbose_ON = False
    skip_file = False
    
    print(f'Params:')
    print(f'fps: {fps}')
    print(f'conf_thresh: {conf_thresh}')
    print(f'smooth_ON: {smooth_ON}')
    print(f'dlc_sm_sec: {dlc_sm_sec}')
    print(f'smooth_wind: {smooth_wind}')
    print(f'interp_factor: {interp_factor}')
    print(f'interp_fill_lim_sec: {interp_fill_lim_sec}')
    print(f'no_abs_pose: {no_abs_pose}') 
    print(f'verbose_ON: {verbose_ON}')    
    
    print(f'Loading DLC matrix: {dlc_path} ...')
    # load in dlc matrix using numpy laod txt, skip 3 rows that have header labeling
    dlc_mat = np.loadtxt(open(dlc_path, "rb"), delimiter=",", skiprows=3)

    print('extracting pose data...')
    # loop thru and extract out pose variables, thresholding for confidence and smoothing
    if len(dlc_mat.shape) > 1:
        num_poses = int((dlc_mat.shape[1]-1)/3)
        num_frames = dlc_mat.shape[0]
        
        if dlc_mat.shape[0] < 8:
            print('DLC file too short...')
            
            skip_file = True
    else:
        print('DLC mat only contains 1 entry...')
        num_poses = int((dlc_mat.shape[0]-1)/3)
        num_frames = 1
        skip_file = True
    
    if not skip_file:
            
        pose_to_proc = np.arange(num_poses)
        pose_to_proc = np.delete(pose_to_proc,pose_to_exclude,axis=0)
        num_poses = len(pose_to_proc)
        if interp_ON == True:
            pose_mat_x = np.zeros([num_poses, int(dlc_mat.shape[0])*interp_factor])
            pose_mat_y = np.zeros([num_poses, int(dlc_mat.shape[0])*interp_factor])
        else:
            pose_mat_x = np.zeros([num_poses, int(dlc_mat.shape[0])])
            pose_mat_y = np.zeros([num_poses, int(dlc_mat.shape[0])])    
            
        perc_nan_list = []
        for ind, pose_i in enumerate(pose_to_proc):
            pose_ind = pose_i*3
            pose_x = dlc_mat[:,pose_ind+1]
            pose_y = dlc_mat[:,pose_ind+2]
            pose_conf = dlc_mat[:,pose_ind+3]
        
            # interpolate pose data
            if interp_ON == True:
                t = np.arange(len(pose_x))
                interp_t = np.arange(len(pose_x),step=0.5)
                pose_x = np.interp(interp_t, t, pose_x)
                pose_y = np.interp(interp_t, t, pose_y)
                
                pose_conf = np.repeat(pose_conf,2)
        
            pose_x[pose_conf < conf_thresh] = np.nan
            pose_y[pose_conf < conf_thresh] = np.nan
            
            pose_x = pd.Series(pose_x)
            pose_y = pd.Series(pose_y)
            pose_x = pose_x.interpolate(limit=interp_fill_lim,limit_direction='both')
            pose_y = pose_y.interpolate(limit=interp_fill_lim,limit_direction='both')
            pose_x = np.array(pose_x)
            pose_y = np.array(pose_y)
            
            # replacing nans with medians might improve ML 
            pose_x[np.isnan(pose_x)] = np.median(pose_x)
            pose_y[np.isnan(pose_y)] = np.median(pose_y)
            
            perc_nan = len(np.where(np.isnan(pose_x))[0])/len(pose_x)
            perc_nan_list.append(perc_nan)
            
            if verbose_ON:
                print(f'Pose {pose_i+1} percent below confidence: {perc_nan*100:.2f}%')
                
            # TODO: remove out of bounds next?
            
            # fig = plt.figure('raw_v_smooth')
            # ax = fig.gca()
            # ax.plot(pose_x)
            if smooth_ON == True:
                pose_x = smooth(pose_x,window_len=smooth_wind,window='hanning')
                pose_y = smooth(pose_y,window_len=smooth_wind,window='hanning')
            # ax.plot(pose_x)
            # plt.draw() 
            # plt.show()
            
            pose_mat_x[ind,:] = pose_x
            pose_mat_y[ind,:] = pose_y
        
        # use itertools to get all 2pair combinations of poses
        combs = combinations(list(np.arange(num_poses)),2)
        combs = [x for x in combs]
        
        head_body_dist = np.array([])
        pose_dists = np.zeros(shape=(len(combs), pose_mat_x.shape[1]))
        for ind, comb in enumerate(combs):
            coords_1 = np.array([pose_mat_x[comb[0],:], pose_mat_y[comb[0],:]])
            coords_2 =  np.array([pose_mat_x[comb[1],:], pose_mat_y[comb[1],:]])
            pose_dists[ind,:] = np.linalg.norm(coords_1-coords_2,axis=0)
            
            if comb == tuple(head_body_norm):
                head_body_dist = pose_dists[ind,:]
        
        # fig = plt.figure()
        # plt.plot(head_body_dist)
        if head_body_dist.any():
            body_length = np.nanpercentile(head_body_dist,90)
            print(f'Using body length of {body_length} pixels for normalizing feats')
        else:
            body_length = 1
        
        # find angle between two pose points
        # https://stackoverflow.com/questions/31735499/calculate-angle-clockwise-between-two-points
        pose_angles = np.zeros(shape=(len(combs), pose_mat_x.shape[1]))
        for ind, comb in enumerate(combs):
            coords_1 = np.vstack([pose_mat_x[comb[0],:], pose_mat_y[comb[0],:]])
            coords_2 = np.vstack([pose_mat_x[comb[1],:], pose_mat_y[comb[1],:]])
            coords_1[np.isnan(coords_1)] = 0
            coords_2[np.isnan(coords_2)] = 0
            coords_ang = np.zeros(coords_1.shape[1])
            for i in range(coords_1.shape[1]):
                # coords_ang[i] = np.math.atan2(np.linalg.det([coords_1[:,i],coords_2[:,i]]),
                #                                      np.dot(coords_1[:,i],coords_2[:,i]))
                coords_ang[i] = angle_between_points((coords_1[0,i],coords_1[1,i]),
                                                     (coords_2[0,i],coords_2[1,i]))
            pose_angles[ind,:] = coords_ang
            
        # # testing angles to verify
        # ind = 500
        # fig = plt.figure(100)
        # ax = plt.gca()
        # plt.scatter(coords_1[0,ind],coords_1[1,ind],20,color=[0,0,0])
        # plt.scatter(coords_2[0,ind],coords_2[1,ind],20,color=[0,1,0])    
        # plt.xlim([0,250])
        # plt.ylim([0,250])
        # angle_between_points((coords_1[0,ind],coords_1[1,ind]),(coords_2[0,ind],coords_2[1,ind]))
        
        # find highest conf poses
        low_inds = np.argsort(perc_nan_list)
        conf_poses = low_inds[0:2]
        
        # calcualte deltas with different index offsets
        # diff_inds = [-6,-4,-2,-1,1,2]        
        diff_inds = [-1*fps,-0.25*fps,-0.15*fps,-1,1,0.15*fps]
        diff_inds = np.ceil(diff_inds).astype(int)        
        input_feats = np.vstack([pose_dists,pose_mat_x[conf_poses,:],pose_mat_y[conf_poses,:]])
        pose_deltas = np.zeros(shape=(input_feats.shape[0]*len(diff_inds),
                                      input_feats.shape[1]))
        for ind, i in enumerate(diff_inds):
            inds = np.arange(ind*input_feats.shape[0],
                             ind*input_feats.shape[0]+input_feats.shape[0])
            if i < 0:
                diff = input_feats[:,-i:] - input_feats[:,:i]
                pose_deltas[inds,:] = np.hstack([np.zeros(shape=(input_feats.shape[0],-i)), diff])
            else:
                diff = input_feats[:,i:] - input_feats[:,:-i]
                pose_deltas[inds,:] = np.hstack([diff, np.zeros(shape=(input_feats.shape[0],i))])
    
        
        # calculate speeds with different offsets    
        # diff_inds = [-6,-4,-2,-1,1,2]
        diff_inds = [-1*fps,-0.25*fps,-0.15*fps,-1,1,0.15*fps]
        diff_inds = np.ceil(diff_inds).astype(int)            
        pose_speeds = np.zeros(shape=(pose_mat_x.shape[0]*len(diff_inds),
                                      pose_mat_x.shape[1]))
        for ind, i in enumerate(diff_inds):
            inds = np.arange(ind*pose_mat_x.shape[0],
                             ind*pose_mat_x.shape[0]+pose_mat_x.shape[0])
            if i < 0:
                xdiff = pose_mat_x[:,-i:] - pose_mat_x[:,:i]
                ydiff = pose_mat_y[:,-i:] - pose_mat_y[:,:i]
                pose_speeds[inds,:] = np.hstack([np.zeros(shape=(pose_mat_x.shape[0],-i)),
                                                 np.sqrt(xdiff**2 + ydiff**2)])
            else:
                xdiff = pose_mat_x[:,i:] - pose_mat_x[:,:-i]
                ydiff = pose_mat_y[:,i:] - pose_mat_y[:,:-i]            
                pose_speeds[inds,:] = np.hstack([np.sqrt(xdiff**2 + ydiff**2),
                                                 np.zeros(shape=(pose_mat_x.shape[0],i))])
        
        # calculate pose angle deltas with diff offsets
        # diff_inds = [-4,-2,-1,1,2]
        diff_inds = [-0.25*fps,-0.15*fps,-1,1,0.15*fps]
        diff_inds = np.ceil(diff_inds).astype(int)              
        input_feats = pose_angles
        pose_ang_deltas = np.zeros(shape=(input_feats.shape[0]*len(diff_inds),
                                      input_feats.shape[1]))
        for ind, i in enumerate(diff_inds):    
            inds = np.arange(ind*input_feats.shape[0],
                             ind*input_feats.shape[0]+input_feats.shape[0])  
            if i < 0:
                diff = np.unwrap(input_feats[:,-i:],discont=180) - np.unwrap(input_feats[:,:i],discont=180)
                pose_ang_deltas[inds,:] = np.hstack([np.zeros(shape=(input_feats.shape[0],-i)), diff])
            else:
                diff = np.unwrap(input_feats[:,i:],discont=180) - np.unwrap(input_feats[:,:-i],discont=180)
                pose_ang_deltas[inds,:] = np.hstack([diff, np.zeros(shape=(input_feats.shape[0],i))])            
        
        print(f'pose delta shape: {pose_deltas.shape}')
        print(f'pose_dists shape: {pose_dists.shape}')
        print(f'pose_speeds shape: {pose_speeds.shape}')
        print(f'pose_angles shape: {pose_angles.shape}')
        print(f'pose_ang_deltas shape: {pose_ang_deltas.shape}')
        
        if no_abs_pose:
            full_feats = np.vstack([pose_deltas/body_length,
                                   pose_dists/body_length,
                                   pose_speeds/body_length,
                                   pose_ang_deltas])
        else:
            full_feats = np.vstack([pose_mat_x,
                                   pose_mat_y,
                                   pose_deltas/body_length,
                                   pose_dists/body_length,
                                   pose_speeds/body_length,
                                   pose_angles,
                                   pose_ang_deltas])            
            
    else:
        full_feats = np.array([])
    
    # TODO: include info on features used, e.g. diff indices etc.
    feat_meta = dict()
    feat_meta = {
        'fps': fps,
        'conf_thresh': conf_thresh,
        'smooth_ON': smooth_ON,
        'smooth_window': smooth_wind,
        'interp_ON': interp_ON,
        'interp_factor': interp_factor,
        'num_frames': num_frames,
        'num_poses': num_poses
    }
        
    return full_feats, feat_meta




















































