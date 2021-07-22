#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 20 15:30:33 2021

@author: markre

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

part_list = glob.glob('/project/3013068.03/RETROICOR/Example_Visualisation/sub-*')
part_list.sort() 
stress_list = ['sub-002', 'sub-003', 'sub-004', 'sub-007', 'sub-009', 'sub-013', 'sub-015', 'sub-017', 'sub-021', 'sub-023', 'sub-025', 'sub-027', 'sub-029']

for subs in part_list:
    sub = Subject(subs[-7:])
    subject = subs[-7:]
    
    if subject in stress_list:
        ses_nr = 3
    elif subject not in stress_list:
        ses_nr = 2
    
    
    t_r = 2.02
    n_scans = 240
    constant = [1]*240
    
    frame_times = np.arange(n_scans)*t_r
    mni_mask = '/project/3013068.03/RETROICOR/MNI152lin_T1_2mm_brain_mask.nii.gz'
    
    #GLM settings
    melodic_GLM = glm.first_level.FirstLevelModel(t_r=2.02, slice_time_ref=0.5, smoothing_fwhm=6, drift_model=None, hrf_model=None, mask_img= mni_mask)
    func_data = sub.get_func_data(session=ses_nr-1,run=2,task='RS', MNI=True)
    
    
    cardiac_phase = glob.glob('/project/3013068.03/RETROICOR/Example_Visualisation/{0}/3C4R1M_vs_AROMA_corrected/cardiac*'.format(subject))
    respiratory_phase = glob.glob('/project/3013068.03/RETROICOR/Example_Visualisation/{0}/3C4R1M_vs_AROMA_corrected/respiratory*'.format(subject))
    multiplication_phase = glob.glob('/project/3013068.03/RETROICOR/Example_Visualisation/{0}/3C4R1M_vs_AROMA_corrected/multiplication*'.format(subject))
    multiplication_phase.sort()
    aroma_noise = glob.glob('/project/3013068.03/RETROICOR/Example_Visualisation/{0}/3C4R1M_vs_AROMA_corrected/aroma*'.format(subject))
    aroma_noise.sort()
    
    cardiac_reg = [np.loadtxt(x) for x in cardiac_phase[:6]]
    respiratory_reg = [np.loadtxt(x) for x in respiratory_phase[:8]]
    multiplication_reg = [np.loadtxt(x) for x in multiplication_phase]
    aroma_noise_reg = [np.loadtxt(x) for x in aroma_noise]
    regressors = cardiac_reg + respiratory_reg + multiplication_reg + aroma_noise_reg + [constant]
    
    columns_retro = ['Car_sin_01', 'Car_cos_01', 'Car_sin_02', 'Car_cos_02', 'Car_sin_03', 'Car_cos_03', \
                     'Resp_sin_01', 'Resp_cos_01', 'Resp_sin_02', 'Resp_cos_02', 'Resp_sin_03', 'Resp_cos_03', 'Resp_sin_04', 'Resp_cos_04', \
                     'Mult_01', 'Mult_02', 'Mult_03', 'Mult_04']
    columns_AROMA = ['AR_' + x[-2:] for x in aroma_noise]
    column_constant = ['Constant']
    
    column_names = columns_retro + columns_AROMA + column_constant
    
    design = pd.DataFrame(regressors)
    design = design.T
    design.index=frame_times
    design.columns = column_names
    
    
    glm_output = melodic_GLM.fit(func_data, design_matrices=design)
    contrast_length_retro = len(cardiac_reg + respiratory_reg + multiplication_reg)
    contrast_length_aroma = len(aroma_noise_reg + column_constant)
    contrast_matrix = np.eye(design.shape[1])
    F_contrast_retro_unique = contrast_matrix[:contrast_length_retro] 
    F_contrast_retro_output = glm_output.compute_contrast([F_contrast_retro_unique], stat_type='F')
    nib.save(F_contrast_retro_output, '/project/3013068.03/RETROICOR/Example_Visualisation/{0}/RETRO_vs_AROMA_revised/Unique_Variance_RETRO_over_AROMA.nii.gz'.format(subject))
