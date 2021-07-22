#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 20 15:30:33 2021

@author: markre

Script to run GLMs used for analysis of RETROICOR vs AROMA variance explained.

Attempt at recreating FSL GLMs for easier access
"""

import glob
import numpy as np
import nibabel as nib
import pandas as pd
from nilearn import glm
from Subject_Class import Subject
from nilearn import glm
from nilearn import plotting
import matplotlib.pyplot as plt
from scipy.stats import norm
from nilearn.glm import threshold_stats_img

part_list = glob.glob('/project/3013068.03/RETROICOR/Example_Visualisation/sub-*')
part_list.sort() 

# Indicating subject having the 'stress' condition during their FIRST functional session
stress_list = ['sub-002', 'sub-003', 'sub-004', 'sub-007', 'sub-009', 'sub-013', 'sub-015', 'sub-017', 'sub-021', 'sub-023', 'sub-025', 'sub-027', 'sub-029']

for subs in part_list:
    
    #Invoke Subject_Class to allow access to all necessary data
    sub = Subject(subs[-7:])
    subject = subs[-7:]
    
    # Account for balancing in stress/control session order
    if subject in stress_list:
        ses_nr = 2
    elif subject not in stress_list:
        ses_nr = 1
    
    # Base scan settings for the used sequence
    t_r = 2.02
    n_scans = 240
    constant = [1.0]*240
    frame_times = np.arange(n_scans)*t_r
    
    # Standard MNI mask used for masking
    mni_mask = '/project/3013068.03/RETROICOR/MNI152lin_T1_2mm_brain_mask.nii.gz'
    
    #GLM settings
    melodic_GLM = glm.first_level.FirstLevelModel(t_r=2.02, slice_time_ref=0.5, smoothing_fwhm=6, \
                                                  drift_model=None, hrf_model=None, mask_img= mni_mask)
    
    #Loading respective functional run as NII-img-like nibabel object
    func_data = sub.get_func_data(session=ses_nr,run=2,task='RS', MNI=True)
    
    #All file-paths to the respective regressor files as created by 'Confound_file_creation_AROMA_model.py'
    cardiac_phase = glob.glob('/project/3013068.03/RETROICOR/Example_Visualisation/{0}/3C4R1M_vs_AROMA_corrected/cardiac*'.format(subject))
    respiratory_phase = glob.glob('/project/3013068.03/RETROICOR/Example_Visualisation/{0}/'\
                                  '3C4R1M_vs_AROMA_corrected/respiratory*'.format(subject))
    multiplication_phase = glob.glob('/project/3013068.03/RETROICOR/Example_Visualisation/{0}/'\
                                     '3C4R1M_vs_AROMA_corrected/multiplication*'.format(subject))
    multiplication_phase.sort()
    aroma_noise = glob.glob('/project/3013068.03/RETROICOR/Example_Visualisation/{0}/3C4R1M_vs_AROMA_corrected/aroma*'.format(subject))
    aroma_noise.sort()
    
    #Create lists of data contained in the regressor files
    cardiac_reg = [np.loadtxt(x) for x in cardiac_phase[:6]]
    respiratory_reg = [np.loadtxt(x) for x in respiratory_phase[:8]]
    multiplication_reg = [np.loadtxt(x) for x in multiplication_phase]
    aroma_noise_reg = [np.loadtxt(x) for x in aroma_noise]
    regressors_RETRO = cardiac_reg + respiratory_reg + multiplication_reg + aroma_noise_reg + [constant]
    regressors_AROMA = aroma_noise_reg + cardiac_reg + respiratory_reg + multiplication_reg + [constant]
    
    #Column names for 3C4R1M RETROICOR
    columns_RETRO = ['Car_sin_01', 'Car_cos_01', 'Car_sin_02', 'Car_cos_02', 'Car_sin_03', 'Car_cos_03', \
                     'Resp_sin_01', 'Resp_cos_01', 'Resp_sin_02', 'Resp_cos_02', 'Resp_sin_03', \
                     'Resp_cos_03', 'Resp_sin_04', 'Resp_cos_04', 'Mult_01', 'Mult_02', 'Mult_03', 'Mult_04']
    
    #Column names for melodic components
    columns_AROMA = ['AR_' + x[-2:] for x in aroma_noise]
    
    #Column name constant
    column_constant = ['Constant']
    
    #Joint column names
    column_names_RETRO = columns_RETRO + columns_AROMA + column_constant
    column_names_AROMA = columns_AROMA + columns_RETRO + column_constant
    
    #Create design matrix as used by nilearn(v0.8) for RETRO unique variance. This is done because I could not figure out
    #how to create a non-singular design matrix for the F-contrast
    design_RETRO = pd.DataFrame(regressors_RETRO)
    design_RETRO = design_RETRO.T
    design_RETRO.index =frame_times
    design_RETRO.columns = column_names_RETRO
    
    #Create design matrix as used by nilearn(v0.8) for AROMA unique variance.
    design_AROMA = pd.DataFrame(regressors_AROMA)
    design_AROMA = design_AROMA.T
    design_AROMA.index =frame_times
    design_AROMA.columns = column_names_AROMA

    #compute SEPERATE glm using the specified design matrix (yeah yeah I know..)
    glm_output_RETRO = melodic_GLM.fit(func_data, design_matrices=design_RETRO)
    glm_output_AROMA = melodic_GLM.fit(func_data, design_matrices=design_AROMA)
    
    #Create contrast matrix for F-tests
    contrast_length_retro = len(cardiac_reg + respiratory_reg + multiplication_reg)
    contrast_length_aroma = len(aroma_noise_reg)
    contrast_matrix = np.eye(design_RETRO.shape[1])
    F_contrast_RETRO_unique = contrast_matrix[:contrast_length_retro] 
    F_contrast_AROMA_unique = contrast_matrix[:contrast_length_aroma]
    F_contrast_shared = contrast_matrix[:contrast_length_retro+contrast_length_aroma]
    
    #Compute contrasts (for the shared contrast the choice of GLM is redundant)
    F_contrast_RETRO_output = glm_output_RETRO.compute_contrast([F_contrast_RETRO_unique], stat_type='F')
    F_contrast_AROMA_output = glm_output_AROMA.compute_contrast([F_contrast_AROMA_unique], stat_type='F')
    F_contrast_shared = glm_output_RETRO.compute_contrast([F_contrast_shared], stat_type='F')
    
    #Save resulting z-maps (unthresholded)
    nib.save(F_contrast_RETRO_output, '/project/3013068.03/RETROICOR/Example_Visualisation/{0}/'\
             'RETRO_vs_AROMA_revised/Unique_Variance_RETRO.nii.gz'.format(subject))
    nib.save(F_contrast_AROMA_output, '/project/3013068.03/RETROICOR/Example_Visualisation/{0}/'\
             'RETRO_vs_AROMA_revised/Unique_Variance_AROMA.nii.gz'.format(subject))
    nib.save(F_contrast_AROMA_output, '/project/3013068.03/RETROICOR/Example_Visualisation/{0}/'\
             'RETRO_vs_AROMA_revised/Shared_Variance_AROMA_RETRO.nii.gz'.format(subject))
                                                           
    #Threshold maps FDR
    thresholded_RETRO_FDR, threshold_RETRO_FDR = threshold_stats_img(F_contrast_RETRO_output, alpha=.05, height_control='fdr')
    thresholded_AROMA_FDR, threshold_AROMA_FDR = threshold_stats_img(F_contrast_AROMA_output, alpha=.05, height_control='fdr')   
    thresholded_shared_FDR, threshold_shared_FDR = threshold_stats_img(F_contrast_shared, alpha=.05, height_control='fdr') 

    #Save resulting z-maps (unthresholded)
    nib.save(thresholded_RETRO_FDR, '/project/3013068.03/RETROICOR/Example_Visualisation/{0}/'\
             'RETRO_vs_AROMA_revised/Unique_Variance_RETRO_fdr_corrected.nii.gz'.format(subject))
    nib.save(thresholded_AROMA_FDR, '/project/3013068.03/RETROICOR/Example_Visualisation/{0}/'\
             'RETRO_vs_AROMA_revised/Unique_Variance_AROMA_fdr_corrected.nii.gz'.format(subject))
    nib.save(thresholded_shared_FDR, '/project/3013068.03/RETROICOR/Example_Visualisation/{0}/'\
             'RETRO_vs_AROMA_revised/Shared_Variance_AROMA_RETRO_fdr_corrected.nii.gz'.format(subject))
                                                           
    #Thresholded maps FWE
    thresholded_RETRO_FWE, threshold_RETRO_FWE = threshold_stats_img(F_contrast_RETRO_output, alpha=.05, height_control='bonferroni')
    thresholded_AROMA_FWE, threshold_AROMA_FWE= threshold_stats_img(F_contrast_AROMA_output, alpha=.05, height_control='bonferroni')
    thresholded_shared_FWE, threshold_shared_FWE= threshold_stats_img(F_contrast_shared, alpha=.05, height_control='bonferroni')

    #Save resulting z-maps (unthresholded)
    nib.save(thresholded_RETRO_FWE, '/project/3013068.03/RETROICOR/Example_Visualisation/{0}/'\
             'RETRO_vs_AROMA_revised/Unique_Variance_RETRO_fwe_corrected.nii.gz'.format(subject))
    nib.save(thresholded_AROMA_FWE, '/project/3013068.03/RETROICOR/Example_Visualisation/{0}/'\
             'RETRO_vs_AROMA_revised/Unique_Variance_AROMA_fwe_corrected.nii.gz'.format(subject))
    nib.save(thresholded_shared_FWE, '/project/3013068.03/RETROICOR/Example_Visualisation/{0}/'\
             'RETRO_vs_AROMA_revised/Shared_Variance_AROMA_RETRO_fwe_corrected.nii.gz'.format(subject))
    
