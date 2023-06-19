#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 20 15:30:33 2021

@author: markre

"""

import glob
import numpy as np
import nibabel as nib
import pandas as pd
from Subject_Class_new import Subject
from nilearn import glm
from nilearn.glm import threshold_stats_img
from nilearn.datasets import load_mni152_brain_mask


BASEPATH = '/project/3013068.03/physio_revision/GLM_approach/'
part_list = glob.glob(BASEPATH + 'sub-*')
part_list.sort() 

# Indicating subject having the 'stress' condition during their FIRsT functional session
stress_list = ['sub-002', 'sub-003', 'sub-004', 'sub-007', 'sub-009', 'sub-013', 'sub-015',
               'sub-017', 'sub-021', 'sub-023', 'sub-025', 'sub-027', 'sub-029']

for subs in part_list:

    # Invoke subject_Class to allow access to all necessary data
    sub_id = subs[-7:]
    glm_path = '/project/3013068.03/physio_revision/GLM_approach/{0}/glm_output/'.format(sub_id)
    sub = Subject(sub_id)

    # Account for balancing in stress/control session order
    ses_nr = 2 if sub_id in stress_list else 1
    
    # Base scan settings for the used sequence
    t_r = 2.02
    n_scans = 240
    constant = [1.0]*240
    frame_times = np.arange(n_scans)*t_r
    
    # standard MNI mask used for masking
    mni_mask = load_mni152_brain_mask()
    
    # GLM settings
    melodic_GLM = glm.first_level.FirstLevelModel(t_r=2.02, 
                                                  slice_time_ref=None,
                                                  smoothing_fwhm=6,
                                                  drift_model=None, 
                                                  hrf_model=None, 
                                                  mask_img=mni_mask,
                                                  verbose=1)
    
    # Loading respective functional run as NII-img-like nibabel object
    func_data = sub.get_func_data(session=ses_nr,
                                  run=2,
                                  task='RS',
                                  MNI=True)
    
    # All file-paths to the respective regressor files as created by 'Confound_file_creation_aroma_model.py'

    retro_noise = sub.get_retroicor_confounds(session=ses_nr, run=2, task='RS')
    aroma_noise = sub.get_aroma_confounds(session=ses_nr, run=2, task='RS')
    acompcor_noise = sub.get_acompcor_confounds(session=ses_nr, run=2, task='RS', number_regressors=5)

    # Create lists of data contained in the regressor files
    regressors_glm6 = pd.concat([retro_noise, aroma_noise, acompcor_noise], axis=1)
    regressors_glm6['constant'] = constant

    contrast_length_retro = len(retro_noise.columns)
    contrast_length_aroma = len(aroma_noise.columns)
    contrast_length_acompcor = len(acompcor_noise.columns)
    
    # Create design matrix as used by nilearn(v0.8) for retro unique variance. This is done because
    # I could not figure out
    # how to create a non-singular design matrix for the F-contrast

    # GLM retroICOR
    design_glm1 = retro_noise.copy()
    design_glm1['constant'] = constant
    design_glm1.index = frame_times
    glm_output = melodic_GLM.fit(func_data, design_matrices=design_glm1)
    contrast_matrix = np.eye(design_glm1.shape[1])
    F_contrast_retro = contrast_matrix[:contrast_length_retro]
    F_contrast_retro_output = glm_output.compute_contrast([F_contrast_retro], stat_type='F')

    # save resulting z-maps (unthresholded)
    nib.save(F_contrast_retro_output, glm_path + 'glm1_retro/variance_retro.nii.gz')

    # Threshold maps FDR
    thresholded_retro_FDR, threshold_retro_FDR = threshold_stats_img(F_contrast_retro_output,
                                                                     alpha=.05,
                                                                     height_control='fdr')
    nib.save(thresholded_retro_FDR, glm_path + 'glm1_retro/variance_retro_fdr_corrected.nii.gz')
    
    # Thresholded maps FWE
    thresholded_retro_FWE, threshold_retro_FWE = threshold_stats_img(F_contrast_retro_output,
                                                                     alpha=.05,
                                                                     height_control='bonferroni')
    nib.save(thresholded_retro_FWE, glm_path + 'glm1_retro/variance_retro_fwe_corrected.nii.gz')
    
    # GLM aroma
    design_glm2 = aroma_noise.copy()
    design_glm2['constant'] = constant
    design_glm2.index = frame_times
    glm_output = melodic_GLM.fit(func_data, design_matrices=design_glm2)
    contrast_matrix = np.eye(design_glm2.shape[1])
    F_contrast_aroma = contrast_matrix[:contrast_length_aroma]
    F_contrast_aroma_output = glm_output.compute_contrast([F_contrast_aroma], stat_type='F')

    # save resulting z-maps (unthresholded)
    nib.save(F_contrast_aroma_output, glm_path + 'glm2_aroma/variance_aroma.nii.gz')

    # Threshold maps FDR
    thresholded_aroma_FDR, threshold_aroma_FDR = threshold_stats_img(F_contrast_aroma_output,
                                                                     alpha=.05,
                                                                     height_control='fdr')
    nib.save(thresholded_aroma_FDR, glm_path + 'glm2_aroma/variance_aroma_fdr_corrected.nii.gz')
    
    # Thresholded maps FWE
    thresholded_aroma_FWE, threshold_aroma_FWE = threshold_stats_img(F_contrast_aroma_output,
                                                                     alpha=.05,
                                                                     height_control='bonferroni')
    nib.save(thresholded_aroma_FWE, glm_path + 'glm2_aroma/variance_aroma_fwe_corrected.nii.gz')
    
    # GLM acompcor

    design_glm3 = acompcor_noise.copy()
    design_glm3['constant'] = constant
    design_glm3.index = frame_times
    glm_output = melodic_GLM.fit(func_data, design_matrices=design_glm3)
    contrast_matrix = np.eye(design_glm3.shape[1])
    F_contrast_acompcor = contrast_matrix[:contrast_length_acompcor]
    F_contrast_acompcor_output = glm_output.compute_contrast([F_contrast_acompcor], stat_type='F')

    nib.save(F_contrast_acompcor_output, glm_path + 'glm3_acompcor/variance_acompcor.nii.gz')

    # Threshold maps FDR
    thresholded_acompcor_FDR, threshold_acompcor_FDR = threshold_stats_img(F_contrast_acompcor_output,
                                                                           alpha=.05,
                                                                           height_control='fdr')
    nib.save(thresholded_acompcor_FDR, glm_path + 'glm3_acompcor/variance_acompcor_fdr_corrected.nii.gz')
    
    # Thresholded maps FWE
    thresholded_acompcor_FWE, threshold_acompcor_FWE = threshold_stats_img(F_contrast_acompcor_output,
                                                                           alpha=.05,
                                                                           height_control='bonferroni')
    nib.save(thresholded_acompcor_FWE, glm_path + 'glm3_acompcor/variance_acompcor_fwe_corrected.nii.gz')
    
    # GLM acompcor + aroma
    design_glm4 = pd.concat([acompcor_noise.copy(), aroma_noise.copy()], axis=1)
    design_glm4['constant'] = constant
    design_glm4.index = frame_times
    contrast_matrix = np.eye(design_glm4.shape[1])
    F_contrast_aroma_unique = contrast_matrix[:contrast_length_aroma]
    F_contrast_acompcor_unique = contrast_matrix[contrast_length_aroma:-1]
    F_contrast_shared = contrast_matrix[:contrast_length_aroma+contrast_length_acompcor]

    glm_output = melodic_GLM.fit(func_data, design_matrices=design_glm4)

    # Create contrast matrix for F-tests

    # Compute contrasts (for the shared contrast the choice of GLM is redundant)
    F_contrast_aroma_output = glm_output.compute_contrast([F_contrast_aroma_unique], stat_type='F')
    F_contrast_acompcor_output = glm_output.compute_contrast([F_contrast_acompcor_unique], stat_type='F')
    F_contrast_shared = glm_output.compute_contrast([F_contrast_shared], stat_type='F')

    # save resulting z-maps (unthresholded)
    nib.save(F_contrast_aroma_output, glm_path + 'glm4_aroma_acompcor/unique_variance_aroma.nii.gz')
    nib.save(F_contrast_acompcor_output, glm_path + 'glm4_aroma_acompcor/unique_variance_acompcor.nii.gz')
    nib.save(F_contrast_shared, glm_path + 'glm4_aroma_acompcor/shared_variance_aroma_acompcor.nii.gz')

    # Threshold maps FDR
    thresholded_aroma_FDR, threshold_aroma_FDR = threshold_stats_img(F_contrast_aroma_output,
                                                                     alpha=.05,
                                                                     height_control='fdr')
    thresholded_acompcor_FDR, threshold_acompcor_FDR = threshold_stats_img(F_contrast_acompcor_output,
                                                                           alpha=.05,
                                                                           height_control='fdr')
    thresholded_shared_FDR, threshold_shared_FDR = threshold_stats_img(F_contrast_shared,
                                                                       alpha=.05,
                                                                       height_control='fdr')
        
    # save resulting z-maps (unthresholded)
    nib.save(thresholded_aroma_FDR, glm_path + 'glm4_aroma_acompcor/unique_variance_aroma_fdr_corrected.nii.gz')
    nib.save(thresholded_acompcor_FDR, glm_path + 'glm4_aroma_acompcor/unique_variance_acompcor_fdr_corrected.nii.gz')
    nib.save(thresholded_shared_FDR, glm_path + 'glm4_aroma_acompcor/shared_variance_aroma_acompcor_fdr_'
                                                'corrected.nii.gz')
        
    # Thresholded maps FWE
    thresholded_aroma_FWE, threshold_aroma_FWE = threshold_stats_img(F_contrast_aroma_output,
                                                                     alpha=.05,
                                                                     height_control='bonferroni')
    thresholded_acompcor_FWE, threshold_acompcor_FWE = threshold_stats_img(F_contrast_acompcor_output,
                                                                           alpha=.05,
                                                                           height_control='bonferroni')
    thresholded_shared_FWE, threshold_shared_FWE = threshold_stats_img(F_contrast_shared,
                                                                       alpha=.05,
                                                                       height_control='bonferroni')

    # save resulting z-maps (unthresholded)
    nib.save(thresholded_aroma_FWE, glm_path + 'glm4_aroma_acompcor/unique_variance_aroma_fwe_corrected.nii.gz')
    nib.save(thresholded_acompcor_FWE, glm_path + 'glm4_aroma_acompcor/unique_variance_acompcor_fwe_corrected.nii.gz')
    nib.save(thresholded_shared_FWE, glm_path + 'glm4_aroma_acompcor/shared_variance_aroma_acompcor_'
                                                'fwe_corrected.nii.gz')

    # GLM retroICOR on top of aroma
    design_glm5 = pd.concat([retro_noise.copy(), aroma_noise.copy()], axis=1)
    design_glm5['constant'] = constant
    design_glm5.index = frame_times
    contrast_matrix = np.eye(design_glm5.shape[1])
    F_contrast_retro_unique = contrast_matrix[:contrast_length_retro] 
    F_contrast_aroma_unique = contrast_matrix[contrast_length_retro:-1]
    F_contrast_shared = contrast_matrix[:contrast_length_retro+contrast_length_aroma]
    
    glm_output = melodic_GLM.fit(func_data, design_matrices=design_glm5)
    

    # Compute contrasts (for the shared contrast the choice of GLM is redundant)
    F_contrast_retro_output = glm_output.compute_contrast([F_contrast_retro_unique], stat_type='F')
    F_contrast_aroma_output = glm_output.compute_contrast([F_contrast_aroma_unique], stat_type='F')
    F_contrast_shared = glm_output.compute_contrast([F_contrast_shared], stat_type='F')

    nib.save(F_contrast_retro_output, glm_path + 'glm5_retro_aroma/unique_variance_retro.nii.gz')
    nib.save(F_contrast_aroma_output, glm_path + 'glm5_retro_aroma/unique_variance_aroma.nii.gz')
    nib.save(F_contrast_shared, glm_path + 'glm5_retro_aroma/shared_variance_aroma_retro.nii.gz')
                                                           
    #Threshold maps FDR
    thresholded_retro_FDR, threshold_retro_FDR = threshold_stats_img(F_contrast_retro_output,
                                                                     alpha=.05,
                                                                     height_control='fdr')
    thresholded_aroma_FDR, threshold_aroma_FDR = threshold_stats_img(F_contrast_aroma_output,
                                                                     alpha=.05,
                                                                     height_control='fdr')
    thresholded_shared_FDR, threshold_shared_FDR = threshold_stats_img(F_contrast_shared,
                                                                       alpha=.05,
                                                                       height_control='fdr')

    nib.save(thresholded_retro_FDR, glm_path + 'glm5_retro_aroma/unique_variance_retro_fdr_corrected.nii.gz')
    nib.save(thresholded_aroma_FDR, glm_path + 'glm5_retro_aroma/unique_variance_aroma_fdr_corrected.nii.gz')
    nib.save(thresholded_shared_FDR, glm_path + 'glm5_retro_aroma/shared_variance_aroma_retro_fdr_corrected.nii.gz')
                                                           
    # Thresholded maps FWE
    thresholded_retro_FWE, threshold_retro_FWE = threshold_stats_img(F_contrast_retro_output,
                                                                     alpha=.05,
                                                                     height_control='bonferroni')
    thresholded_aroma_FWE, threshold_aroma_FWE = threshold_stats_img(F_contrast_aroma_output,
                                                                     alpha=.05,
                                                                     height_control='bonferroni')
    thresholded_shared_FWE, threshold_shared_FWE = threshold_stats_img(F_contrast_shared,
                                                                       alpha=.05,
                                                                       height_control='bonferroni')

    # save resulting z-maps (unthresholded)
    nib.save(thresholded_retro_FWE, glm_path + 'glm5_retro_aroma/unique_variance_retro_fwe_corrected.nii.gz')
    nib.save(thresholded_aroma_FWE, glm_path + 'glm5_retro_aroma/unique_variance_aroma_fwe_corrected.nii.gz')
    nib.save(thresholded_shared_FWE, glm_path + 'glm5_retro_aroma/shared_variance_aroma_retro_fwe_corrected.nii.gz')

    # GLM retroICOR on top of aroma and acompcor
    design_glm6 = regressors_glm6.copy()
    design_glm6.index = frame_times
    glm_output = melodic_GLM.fit(func_data, design_matrices=design_glm6)
    contrast_matrix = np.eye(design_glm6.shape[1])
    F_contrast_retro_unique = contrast_matrix[:contrast_length_retro] 
    F_contrast_aroma_unique = contrast_matrix[contrast_length_retro:contrast_length_retro+contrast_length_aroma]
    F_contrast_acompcor_unique = contrast_matrix[contrast_length_retro+contrast_length_aroma:-1]
    F_contrast_shared = contrast_matrix[:contrast_length_retro+contrast_length_aroma+contrast_length_acompcor]

    # Create contrast matrix for F-tests

    # Compute contrasts (for the shared contrast the choice of GLM is redundant)
    F_contrast_retro_output = glm_output.compute_contrast([F_contrast_retro_unique], stat_type='F')
    F_contrast_aroma_output = glm_output.compute_contrast([F_contrast_aroma_unique], stat_type='F')
    F_contrast_acompcor_output = glm_output.compute_contrast([F_contrast_acompcor_unique], stat_type='F')
    F_contrast_shared = glm_output.compute_contrast([F_contrast_shared], stat_type='F')
    
    # save resulting z-maps (unthresholded)
    nib.save(F_contrast_retro_output, glm_path + 'glm6_retro_aroma_acompcor/unique_variance_retro.nii.gz')
    nib.save(F_contrast_aroma_output, glm_path + 'glm6_retro_aroma_acompcor/unique_variance_aroma.nii.gz')
    nib.save(F_contrast_acompcor_output, glm_path + 'glm6_retro_aroma_acompcor/unique_variance_acompcor.nii.gz')
    nib.save(F_contrast_shared, glm_path + 'glm6_retro_aroma_acompcor/shared_variance_aroma_retro_acompcor.nii.gz')
    
                                                       
    # Threshold maps FDR
    thresholded_retro_FDR, threshold_retro_FDR = threshold_stats_img(F_contrast_retro_output,
                                                                     alpha=.05,
                                                                     height_control='fdr')
    thresholded_aroma_FDR, threshold_aroma_FDR = threshold_stats_img(F_contrast_aroma_output,
                                                                     alpha=.05,
                                                                     height_control='fdr')
    thresholded_acompcor_FDR, threshold_acompcor_FDR = threshold_stats_img(F_contrast_acompcor_output,
                                                                           alpha=.05,
                                                                           height_control='fdr')
    thresholded_shared_FDR, threshold_shared_FDR = threshold_stats_img(F_contrast_shared,
                                                                       alpha=.05,
                                                                       height_control='fdr')

    # save resulting z-maps (unthresholded)
    nib.save(thresholded_retro_FDR, glm_path + 'glm6_retro_aroma_acompcor/unique_variance_retro_fdr_corrected.nii.gz')
    nib.save(thresholded_aroma_FDR, glm_path + 'glm6_retro_aroma_acompcor/unique_variance_aroma_fdr_corrected.nii.gz')
    nib.save(thresholded_acompcor_FDR, glm_path + 'glm6_retro_aroma_acompcor/'
                                                  'unique_variance_acompcor_fdr_corrected.nii.gz')
    nib.save(thresholded_shared_FDR, glm_path + 'glm6_retro_aroma_acompcor/shared_variance_aroma_retro_'
                                                'acompcor_fdr_corrected.nii.gz')
                                                           
    # Thresholded maps FWE
    thresholded_retro_FWE, threshold_retro_FWE = threshold_stats_img(F_contrast_retro_output,
                                                                     alpha=.05,
                                                                     height_control='bonferroni')
    thresholded_aroma_FWE, threshold_aroma_FWE = threshold_stats_img(F_contrast_aroma_output,
                                                                     alpha=.05,
                                                                     height_control='bonferroni')
    thresholded_acompcor_FWE, threshold_aroma_FWE = threshold_stats_img(F_contrast_acompcor_output,
                                                                        alpha=.05,
                                                                        height_control='bonferroni')
    thresholded_shared_FWE, threshold_shared_FWE = threshold_stats_img(F_contrast_shared,
                                                                       alpha=.05,
                                                                       height_control='bonferroni')

    # save resulting z-maps (unthresholded)
    nib.save(thresholded_retro_FWE, glm_path + 'glm6_retro_aroma_acompcor/unique_variance_retro_fwe_corrected.nii.gz')
    nib.save(thresholded_aroma_FWE, glm_path + 'glm6_retro_aroma_acompcor/unique_variance_aroma_fwe_corrected.nii.gz')
    nib.save(thresholded_acompcor_FWE, glm_path + 'glm6_retro_aroma_acompcor/'
                                                  'unique_variance_acompcor_fwe_corrected.nii.gz')
    nib.save(thresholded_shared_FWE, glm_path + 'glm6_retro_aroma_acompcor/'
                                                'shared_variance_aroma_retro_fwe_corrected.nii.gz')
