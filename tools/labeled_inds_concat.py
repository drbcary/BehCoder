# -*- coding: utf-8 -*-
"""
Created on Sun Jan 29 13:17:27 2023

@author: bcary


compile labeled inds together into one files


"""

import os
import csv
import glob
import time

import numpy as np

data_dir = r'F:\Work\Anim_vids\KH72_73'
fps = 2

labeled_str = '*LabeledInds_Pred*'
lab_seach_path = os.path.join(data_dir,labeled_str + '.csv')

dlc_search_path = os.path.join(data_dir,'*DLC*' + '.csv')
dlc_files = glob.glob(dlc_search_path)

# order the label files by vid num
label_files = glob.glob(lab_seach_path)
vid_nums = np.array([])
for i in label_files:
    sep_split = i.split(os.sep)
    filename = sep_split[-1]
    lab_split = filename.split('_Labeled')
    vid_name = lab_split[0]
    vid_num = vid_name.split('_')[-1]
    vid_nums = np.append(vid_nums, int(vid_num))

sort_ind = np.argsort(vid_nums)
label_files = [label_files[x] for x in sort_ind]

with open(label_files[500], newline = '') as f:
    output = np.loadtxt(f, delimiter=',', skiprows=1, dtype=int)
    
row_ind = 0
labels = np.zeros(shape=(len(label_files)*output.shape[0],
                            output.shape[1]-1))
for file in label_files:
    print(f'Loading file: {file}')
    with open(file, newline = '') as f:
        output = np.loadtxt(f, delimiter=',', skiprows=1, dtype=int)
        
    if len(output.shape) > 1:
        file_labels = np.array(output[:,1:]) # get rid of first frame col
    else:
        file_labels = np.array(output[1:])
    
    sep_split = file.split(os.sep)
    filename = sep_split[-1]
    lab_split = filename.split('_Labeled')
    vid_name = lab_split[0]
    dlc_file_found = False
    
    # compare shape to dlc file
    for f in dlc_files:
        if vid_name+'DLC' in f:
            dlc_file_found = True
            dlc_mat = np.loadtxt(open(f, "rb"), delimiter=",", skiprows=3)
            if len(dlc_mat.shape) > 1:
                dlc_frames = dlc_mat.shape[0]
            else:
                dlc_frames = 1
            
    if len(file_labels.shape) > 1:
        num_rows = file_labels.shape[0]
    else:
        num_rows = 1            
          
    if dlc_file_found:
        same_fr_num = dlc_frames == num_rows
        if not same_fr_num:
            print('DLC and labeled inds do not have same frame number!')
            break
    else:
        print('corresponding DLC file not found:')
    
    print(f'Number of rows: {num_rows}')
    row_inds = np.arange(row_ind,row_ind+num_rows)
    labels[row_inds,:] = file_labels
    
    row_ind += num_rows

# get rid of extra rows that should just be zeros at the end if they exist
if labels.shape[0] > row_ind:
    rows_toDel = np.arange(row_ind,labels.shape[0])
    labels = np.delete(labels, rows_toDel, axis=0)
    

label_names = ['none', 'toy play','cross walk','circle run','wall explore','bed invest.',
          'alert','human','social','nesting','eating','drink','groom','sleep']

save_dir = r'D:\Work\DATA\V1_BehMod\V1BehMod_Data\Processed_Structs\AutoBeh'
savename = r'KH72_MultiAnim_NoAbsDistAng_Double_learnWeights_v2_Mdl'

save_path_inds = os.path.join(save_dir, savename + '_PredLabels' + '.csv')
print(f'Saving labeled inds: {save_path_inds}')
with open(save_path_inds, 'w', newline='') as f:
    writer = csv.writer(f)
    header = label_names[:] # make new list variable
    header.insert(0, 'frame')
    writer.writerow(header)
    
    # write data
    for i, l in enumerate(labels):
        row = [int(x) for x in l]
        row.insert(0,i+1) # start the frame count at 1 not 0
                        
        writer.writerow(row)    




























