
# import pandas as pd
import numpy as np
import sklearn.ensemble as skl
from sklearn.preprocessing import LabelEncoder

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

from matplotlib import pyplot as plt

import pickle
import time
import os


def train_model_RF(feat_data, labels, verb_ON, fps, **kwargs):
    """
    inputs:
        -feat_data: matrix with feature data (feats x frames), numpy array
        -labels: beh labels (1 x frames), numpy array
        -nTrees, number of RF trees used for model, int
        -verb_ON/norm_ON: boolean values for additional settings
    """
    
    max_model_len = kwargs.get('max_model_len',10000)
    nTrees = kwargs.get('model_nTrees',400)
    norm_ON = kwargs.get('ML_norm_ON',10000)
    PCA_ON = kwargs.get('ML_PCA_ON',10000)
    double_train_ON = kwargs.get('double_train_ON',False)
    ch_len = kwargs.get('ch_len',10)
    label_samp_weights = kwargs.get('label_samp_weights',np.array([]))
    label_train_weights = kwargs.get('label_train_weights',np.array([]))
    
    print(f'double train: {double_train_ON}')
    
    # remove nans if exist
    feat_data[np.isnan(feat_data)] = 0
        
    label_counts = np.sum(labels,axis=0)
    num_nonzero_labs = np.sum(label_counts > 0)
    temp_labels = np.array(labels)
    ideal_prop = 1/num_nonzero_labs
    
    # experimenting with limiting size of input feat data (8/25/23)
    if temp_labels.shape[0] > max_model_len:    
        resamp_inds = []
        
        while len(resamp_inds) < max_model_len:
            for l in np.arange(temp_labels.shape[1]):
                l_inds = np.where(temp_labels[:,l] == 1)[0]
                
                if len(resamp_inds) > 0:
                    resamp_inds = resamp_inds.astype(int)
                    prop_l = np.sum(labels[resamp_inds,l] == 1)/len(resamp_inds)
                else:
                    prop_l = ideal_prop
                    
                if prop_l == 0:
                    prop_l = 0.01
                    
                if not label_samp_weights:
                    ch_adj = np.round(np.clip(ch_len*(ideal_prop/prop_l),0,ch_len*2)).astype(int)
                else:
                    ch_adj = np.round(np.clip(
                        ch_len*(ideal_prop/prop_l),0,ch_len*2)*label_samp_weights[l]).astype(int)
                
                # if ch_adj < ch_len*0.6:
                #     ch_adj = np.round(ch_len/10).astype(int)
                
                if len(l_inds) != 0:
                    samp_inds = np.random.choice(l_inds, ch_adj, replace=True)
                    resamp_inds = np.append(resamp_inds, samp_inds)
                    temp_labels[samp_inds,:] = 0
                    
        resamp_inds = resamp_inds.astype(int)
        labels = labels[resamp_inds,:]
        feat_data = feat_data[resamp_inds,:]
    
    if verb_ON:
        print('feature array shape: {}'.format(feat_data.shape))
        print('class array shape: {}'.format(labels.shape))
        
    # attempt to correct for input variables not aligned along axes
    if feat_data.shape[0] != labels.shape[0]:
        if feat_data.shape[1] == labels.shape[0]:
            feat_data = feat_data.T
        else:
            print('Conflicting variable shapes')
      
    # get counts of each label for potential use for training data balancing
    label_counts = np.sum(labels,axis=0)
    
    if verb_ON:
        label_props = label_counts/np.sum(label_counts)
        for ind, i in enumerate(label_props):
            print(f'Label {ind} proportion: {i}')
        
    if norm_ON:
        print('Normalizing feat data...')
        # normalize and pca data to aid in RF
        scaler = StandardScaler()
        feat_data = scaler.fit_transform(feat_data)
    
    if PCA_ON:
        print('Running PCA on features to reduce dimensions')
        print(f'PCA reduced dims from: {feat_data.shape}')
        
        t0 = time.time()
        # potentially add in ability to perform PCA on feats to improve performance
        pca = PCA(n_components = 0.90)
        pca.fit(feat_data)
        num_pca_dims = pca.n_components_
        feat_data = pca.transform(feat_data)
        
        t1 = time.time()
        print(f'to: {feat_data.shape}')
        print(f'PCA took {t1-t0} seconds')
    
    else:
        pca = None
    
    # # if there are -1s in labels turn into 0s
    # labels[labels<0] = 0
    
    # remove frame data with -1s
    neg_hits_arr = np.zeros(shape=labels.shape)
    neg_hits_arr[labels<0] = 1
    neg_rows = np.where(np.sum(neg_hits_arr,axis=1)>0)[0]
    print(f'Removing {len(neg_rows)} unlabeled frames from training...')
    
    if neg_rows.any():
        print(f'Removing {len(neg_rows)} unlabeled frames from training...')
        labels = np.delete(labels,neg_rows,axis=0)    
        feat_data = np.delete(feat_data,neg_rows,axis=0)    
    
    # Run RF training
    # OOB_score not working with multi label one hot encoding for some reason
    
    print('Starting Random Forest training...')
    if verb_ON:
        verb_lvl = 1
    min_sample_frac = 0.0005 # testing this new hyperparam
    cores_to_use = os.cpu_count() - 3
    
    if label_train_weights.any():
        
        # build class weight dict
        class_weights = []
        for ind, l in enumerate(label_train_weights):
            
            if label_counts[ind] > 0:
                class_weights.append({0: 1, 1: l})
            else:
                class_weights.append({0: 1})

    else:
        class_weights = 'balanced_subsample'
        
    RF1 = skl.RandomForestClassifier(nTrees,oob_score=False,bootstrap=True,
                                    verbose=verb_lvl,n_jobs=cores_to_use,
                                    min_samples_leaf=min_sample_frac,
                                    class_weight=class_weights)
    
    Mdl1 = RF1.fit(feat_data,labels)
    # oob_error = Mdl.oob_score_
    feature_importance1 = Mdl1.feature_importances_
        
    if double_train_ON:
    
        pred_lab_probs = Mdl1.predict_proba(feat_data)
        
        # convert predicted label probs to array
        pred_prob_array = np.zeros(shape=(labels.shape))
        for i in range(len(pred_lab_probs)):
            lab_probs = pred_lab_probs[i]

            if lab_probs.shape[1] > 1:
                pred_prob_array[:,i] = lab_probs[:,1]
            else:
                pred_prob_array[:,i] = np.zeros(shape=(lab_probs.shape[0]))
        
        print('creating label features')
        # select highest prob label if none chosen for a frame
        pred_labels = np.zeros(shape=pred_prob_array.shape)
        pred_labels[pred_prob_array > 0.5] = 1
        for i in range(pred_labels.shape[0]):
            no_lab = (pred_labels[i,:] == 1).any()
            
            if no_lab == False:
                max_lab = np.argmax(pred_prob_array[i,:])
                pred_labels[i,max_lab] = 1    
        
        # create pred_label_features
        # diff_inds = [-15,-8,-4,-3,-2,-1,1,2,3,10]
        diff_inds = [-10*fps,-5*fps,-1*fps,-0.25*fps,-0.15*fps,-0.1*fps,-1,
                     1,0.1*fps,0.15*fps,1*fps,5*fps]
        diff_inds = np.ceil(diff_inds).astype(int) 
        
        input_feats = pred_labels.T
        label_feats = np.zeros(shape=(input_feats.shape[0]*len(diff_inds),
                                      input_feats.shape[1]))
        for ind, i in enumerate(diff_inds):
            inds = np.arange(ind*input_feats.shape[0],
                             ind*input_feats.shape[0]+input_feats.shape[0])
            if i < 0:
                diff = input_feats[:,:i]
                label_feats[inds,:] = np.hstack([np.zeros(shape=(input_feats.shape[0],-i)), diff])
            else:
                diff = input_feats[:,i:]
                label_feats[inds,:] = np.hstack([diff, np.zeros(shape=(input_feats.shape[0],i))])   
                
        full_feats = np.hstack([label_feats.T,feat_data])
            
        print('Running second model training...')
        RF2 = skl.RandomForestClassifier(nTrees,oob_score=False,bootstrap=True,
                                        verbose=verb_lvl,n_jobs=cores_to_use,
                                        min_samples_leaf=min_sample_frac,
                                        class_weight='balanced_subsample')        
        Mdl2 = RF2.fit(full_feats,labels)
        # oob_error = Mdl.oob_score_
        feature_importance2 = Mdl2.feature_importances_    
        
        mdl_output = [Mdl1, Mdl2]
        
        fig = plt.figure(100)
        ax = plt.gca()
        ax.plot(feature_importance2)        
        plt.title('feat importance 2')
        
    else:
        mdl_output = Mdl1
        
        
    fig = plt.figure(101)
    ax = plt.gca()
    ax.plot(feature_importance1)
    plt.title('feat importance 1')
    
    
    
    
    
    # if verb_ON:
    #     print('RF OOB score is: {:.2f}%'.format(oob_error * 100))
    
    return mdl_output, pca
        
