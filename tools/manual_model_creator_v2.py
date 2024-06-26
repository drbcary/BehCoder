# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 15:48:13 2023

@author: bcary

train RF using all labeledInds and DLC pose csvs in a folder

"""

import os
import glob
import pickle
import time
from datetime import datetime

import yaml

import numpy as np

import FeatureExtraction
import TrainModel_RF

from matplotlib import pyplot as plt

# for plotting in separate window
from IPython import get_ipython

if get_ipython():
    get_ipython().run_line_magic('matplotlib', 'qt')



#####################################
### SET PARAMS FOR MODEL CREATION ###
#####################################

# manually set paths for data, config, and model save
data_dir = r''

config_path = r""

model_savename = r''
model_save_dir = r''

#####################################


# parses yaml file into config dictionary
with open(config_path, 'r') as stream:
    try:
        config = yaml.safe_load(stream)
        
    except yaml.YAMLError as exc:
        print(exc)

fps = config['fps']
dlc_params = {
    'xlims': config['xlims'],
    'ylims': config['ylims'],
    'fps': config['fps'],
    'dlc_conf_thresh': config['dlc_conf_thresh'],
    'dlc_interp_factor': config['dlc_interp_factor'],
    'interp_fill_lim': config['interp_fill_lim'],
    'dlc_smooth_ON': config['dlc_smooth_ON'],
    'dlc_sm_sec': config['dlc_sm_sec'],
    'no_abs_pose': config['no_abs_pose'],
    'pose_to_exclude': config['pose_to_exclude'],
    'head_body_norm': config['head_body_norm']
    }

ML_params = {
    'max_model_len': config['max_model_len'],
    'model_nTrees': config['model_nTrees'],
    'ML_norm_ON': config['ML_norm_ON'],
    'ML_PCA_ON': config['ML_PCA_ON'],
    'double_train_ON': config['double_train_ON']
    }


dlc_params['no_abs_pose'] = True

labeled_str = '*LabeledInds*'
lab_seach_path = os.path.join(data_dir,labeled_str + '.csv')
dlc_search_path = os.path.join(data_dir,'*DLC*' + '.csv')

t0 = time.time()

lab_files = glob.glob(lab_seach_path)
dlc_files = glob.glob(dlc_search_path)

with open(lab_files[0], newline = '') as f:
    output = np.loadtxt(f, delimiter=',', skiprows=1, dtype=int)
    
[file_feats, meta] = FeatureExtraction.extract_features(dlc_files[0], **dlc_params)

row_ind = 0
feat_data = np.zeros(shape=(len(lab_files)*file_feats.shape[1],
                            file_feats.shape[0]))
labels = np.array([])
for file in lab_files:
    with open(file, newline = '') as f:
        output = np.loadtxt(f, delimiter=',', skiprows=1, dtype=int)
    
    file_labels = np.array(output[:,1:]) # get rid of first frame col and transpose
    
    file_labels = np.repeat(file_labels, dlc_params['dlc_interp_factor'], axis=0)
    lab_inds = np.where(file_labels[:,0] > -1)[0] # should work to look at one label col
    labels_toAdd = np.array(file_labels[lab_inds,:])
    
    if labels.size > 1:
        labels = np.vstack([labels, labels_toAdd])
    else:
        labels = np.array(labels_toAdd)
    
    path_sp = file.split(os.path.sep)
    filename = path_sp[-1]
    vid_name = filename.split('_'+labeled_str[1])[0]
    
    print(f'Found labeled file for: {vid_name}')
    dlc_name = vid_name + 'DLC'
    dlc_found = False
    for dlc_file in dlc_files:
        if dlc_name in dlc_file:
            print(f'Found corresponding dlc')
            [file_feats, meta] = FeatureExtraction.extract_features(dlc_file, **dlc_params)  
            
            feats_toAdd = np.array(file_feats[:,lab_inds])
            row_inds = np.arange(row_ind,row_ind+feats_toAdd.shape[1])
            feat_data[row_inds,:] = feats_toAdd.T
            
            row_ind += feats_toAdd.shape[1]
            
            dlc_found = True
            
            if feats_toAdd.shape[1] != labels_toAdd.shape[0]:
                print('not same size !!!!!')
                
            print(feats_toAdd.shape)
            print(labels_toAdd.shape)
            
    if dlc_found == False:
        print('Did not find corresponding DLC file')
        break
    
# get rid of extra rows that should just be zeros at the end if they exist
if feat_data.shape[0] > row_ind:
    rows_toDel = np.arange(row_ind,feat_data.shape[0])
    feat_data = np.delete(feat_data, rows_toDel, axis=0)

t1 = time.time()
print('-------------------------------------------')
print(f'feature extraction took {t1-t0} seconds')
print('-------------------------------------------')


# eat_ind = 6
# eat_rows = np.where(labels[:,eat_ind]==1)[0]
# if eat_rows.any():
#     print(f'Removing {len(eat_rows)} unlabeled frames from training...')
#     labels = np.delete(labels,eat_rows,axis=0)    
#     feat_data = np.delete(feat_data,eat_rows,axis=0)    


# create balanced weights based on number of samples of each label
lab_ct = np.sum(labels,axis=0).astype(float)
lab_ct[lab_ct==0] = np.NAN
num_nonzero_labs = np.sum(lab_ct > 0)
lab_weight_bal = np.nansum(lab_ct) / (lab_ct * num_nonzero_labs)
lab_weight_bal[np.isnan(lab_weight_bal)] = 0

# # alter balanced weights to bias label learning
# # lab_weight_bias = np.array([1.2,1,1,0.9,0.9,0.77,0.65,1,0.7,1,0.55,0.55,1.3,1.05])
# lab_weight_bias = np.array([1,0.9,0.93,1,0.9,1,0])
# lab_weights = lab_weight_bal * lab_weight_bias

# ML_params['label_train_weights'] = lab_weights
ML_params['double_train_ON'] = True
ML_params['ch_len'] = 100
ML_params['max_model_len'] = 10e10
verb_ON = True
[ML_model, PCA_model] = TrainModel_RF.train_model_RF(feat_data, labels, verb_ON, fps,
                                                     **ML_params)
PCA_ON = ML_params['ML_PCA_ON']
if PCA_ON == False:
    PCA_model = None

print('Saving ML model...')
if PCA_model:
    pkl_filename = os.path.join(model_save_dir,model_savename+'_Mdl_and_PCAMdl'+'.pkl')
    print(pkl_filename)

    with open(pkl_filename,'wb') as pkl_out:
     	pickle.dump([ML_model,PCA_model],pkl_out,-1)
else:
    pkl_filename = os.path.join(model_save_dir,model_savename+'_Mdl.pkl')
    print(pkl_filename)

    with open(pkl_filename,'wb') as pkl_out:
     	pickle.dump([ML_model,PCA_model],pkl_out,-1)



date_time_str = datetime.today().strftime('%Y-%m-%d_H%H-M%M')
                                 
filename = model_savename + '_feat_data.pkl'
pkl_filename = os.path.join(model_save_dir,filename)

print('Saving feat data...')
print(pkl_filename)
  
with open(pkl_filename,'wb') as pkl_out:
 	pickle.dump([feat_data.T , labels.T],pkl_out,-1)
     
print('Finished!')        





# labcount_filename = os.path.join(model_save_dir,model_savename+'_label_counts'+'.csv')
# np.savetxt(labcount_filename, label_counts)

# file = np.loadtxt(labcount_filename)

# # load feat data from pickle file
# feat_path = r"F:\Work\Anim_vids\MultiAnimal\models\MultiAnim_NoAbsDistAng_Double_sampWeight_v2_feat_data.pkl"
# with open(feat_path,'rb') as pkl_in:
# 	file_output = pickle.load(pkl_in)

# [ML_feats, ML_labels] = file_output

# feat_data = ML_feats.T
# labels = ML_labels.T

# # load model from pickle file
# model_path = r"F:\Work\Anim_vids\AT15_16\models\AT16_light_day1_dayER_testNoAbs_Mdl.pkl"
# with open(model_path,'rb') as pkl_in:
# 	file_output = pickle.load(pkl_in)
# loaded_model = file_output[0]

# loaded_model = ML_model
# feats_toplot = file_feats
# feats_toplot[np.isnan(feats_toplot)] = 0
# pred_lab_probs = loaded_model[0].predict_proba(feats_toplot.T)

# R = np.linspace(0, 1, 14)
# cmap = plt.cm.get_cmap("Spectral")(R)
# fig = plt.figure()
# ax = plt.gca()
# for i in range(len(pred_lab_probs)):
    
#     if pred_lab_probs[i].shape[1] > 1:
#         plt.plot(pred_lab_probs[i][:,1],color=cmap[i],linewidth=0.8)

# plt.title('predicted probs')



            