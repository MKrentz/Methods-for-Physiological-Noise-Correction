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
from Subject_Class import Subject
from nilearn import glm
from nilearn.glm import threshold_stats_img
from nilearn.datasets import load_mni152_brain_mask


BASEPATH = '/project/3013068.03/test/GLM_approach/'
part_list = glob.glob(BASEPATH + 'sub-*')
part_list.sort() 

#part_list = [part_list[0]]

# Indicating subject having the 'stress' condition during their FIRsT functional session
stress_list = ['sub-002', 'sub-003', 'sub-004', 'sub-007', 'sub-009', 'sub-013', 'sub-015', 'sub-017', 'sub-021', 'sub-023', 'sub-025', 'sub-027', 'sub-029']

for subs in part_list:
    

    #Invoke subject_Class to allow access to all necessary data
    sub_id = subs[-7:]
    glm_path = '/project/3013068.03/test/GLM_approach/{0}/glm_output/'.format(sub_id)
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
    
    #GLM settings
    melodic_GLM = glm.first_level.FirstLevelModel(t_r=2.02, 
                                                  slice_time_ref=0.5, 
                                                  smoothing_fwhm=6,
                                                  drift_model=None, 
                                                  hrf_model=None, 
                                                  mask_img= mni_mask, 
                                                  verbose=1)
    
    #Loading respective functional run as NII-img-like nibabel object
    func_data = sub.get_func_data(session = ses_nr, run = 2, task = 'RS', MNI = True)
    
    #All file-paths to the respective regressor files as created by 'Confound_file_creation_aroma_model.py'
    cardiac_phase = glob.glob(BASEPATH + '{0}/confounds/cardiac*'.format(sub_id))
    respiratory_phase = glob.glob(BASEPATH + '{0}/'\
                                  'confounds/respiratory*'.format(sub_id))
    multiplication_phase = glob.glob(BASEPATH + '{0}/'\
                                     'confounds/multiplication*'.format(sub_id))
    multiplication_phase.sort()
    aroma_noise = glob.glob(BASEPATH + '{0}/confounds/aroma*'.format(sub_id))
    aroma_noise.sort()
    
    acompcor_noise = glob.glob((BASEPATH + '{0}/confounds/a_comp_cor*'.format(sub_id)))
    acompcor_noise.sort()
    
    #Create lists of data contained in the regressor files
    cardiac_reg = [np.loadtxt(x) for x in cardiac_phase[:6]]
    respiratory_reg = [np.loadtxt(x) for x in respiratory_phase[:8]]
    multiplication_reg = [np.loadtxt(x) for x in multiplication_phase]
    aroma_noise_reg = [np.loadtxt(x) for x in aroma_noise]
    acompcor_noise_reg = [np.loadtxt(x) for x in acompcor_noise]
    regressors_glm6 = cardiac_reg + respiratory_reg + multiplication_reg + aroma_noise_reg + acompcor_noise_reg + [constant]
    
    #Column names for 3C4R1M retroICOR
    columns_retro = ['Car_sin_01', 'Car_cos_01', 'Car_sin_02', 'Car_cos_02', 'Car_sin_03', 'Car_cos_03', \
                     'Resp_sin_01', 'Resp_cos_01', 'Resp_sin_02', 'Resp_cos_02', 'Resp_sin_03', \
                     'Resp_cos_03', 'Resp_sin_04', 'Resp_cos_04', 'Mult_01', 'Mult_02', 'Mult_03', 'Mult_04']
    
    #Column names for melodic components
    columns_aroma = ['AR_' + x[-2:] for x in aroma_noise]
    
    #Column names for acompcor compoents
    columns_acompcor = ['ACC_' + x[-2:] for x in acompcor_noise]
    
    #Column name constant
    column_constant = ['Constant']
  
    #Joint column names
    column_names = columns_retro + columns_aroma + columns_acompcor + column_constant
    
    
    contrast_length_retro = len(cardiac_reg + respiratory_reg + multiplication_reg)
    contrast_length_aroma = len(aroma_noise_reg)
    contrast_length_acompcor = len(acompcor_noise_reg)
    
    #Create design matrix as used by nilearn(v0.8) for retro unique variance. This is done because I could not figure out
    #how to create a non-singular design matrix for the F-contrast
    

    #GLM retroICOR
    column_names_glm1 = columns_retro + column_constant
    regressors_glm1 = cardiac_reg + respiratory_reg + multiplication_reg + [constant]
    design_glm1 = pd.DataFrame(regressors_glm1)
    design_glm1 = design_glm1.T
    design_glm1.index =frame_times
    design_glm1.columns = column_names_glm1
    glm_output = melodic_GLM.fit(func_data, design_matrices=design_glm1)
    contrast_matrix = np.eye(design_glm1.shape[1])
    F_contrast_retro= contrast_matrix[:contrast_length_retro]
    F_contrast_retro_output = glm_output.compute_contrast([F_contrast_retro], stat_type= 'F')
    #save resulting z-maps (unthresholded)
    nib.save(F_contrast_retro_output, glm_path + \
             'glm1_retro/variance_retro.nii.gz')
    #Threshold maps FDR
    thresholded_retro_FDR, threshold_retro_FDR = threshold_stats_img(F_contrast_retro_output, alpha=.05, height_control= 'fdr' )
    #save resulting z-maps (unthresholded)
    nib.save(thresholded_retro_FDR, glm_path + \
             'glm1_retro/variance_retro_fdr_corrected.nii.gz')
    
    #Thresholded maps FWE
    thresholded_retro_FWE, threshold_retro_FWE = threshold_stats_img(F_contrast_retro_output, alpha = .05, height_control = 'bonferroni')   
    nib.save(thresholded_retro_FWE, glm_path + \
             'glm1_retro/variance_retro_fwe_corrected.nii.gz')
    
    #GLM aroma
    column_names_glm2 = columns_aroma + column_constant
    regressors_glm2 = aroma_noise_reg + [constant]
    design_glm2 = pd.DataFrame(regressors_glm2)
    design_glm2 = design_glm2.T
    design_glm2.index =frame_times
    design_glm2.columns = column_names_glm2
    glm_output = melodic_GLM.fit(func_data, design_matrices=design_glm2)
    contrast_matrix = np.eye(design_glm2.shape[1])
    F_contrast_aroma= contrast_matrix[:contrast_length_aroma]
    F_contrast_aroma_output = glm_output.compute_contrast([F_contrast_aroma], stat_type= 'F')
    #save resulting z-maps (unthresholded)
    nib.save(F_contrast_aroma_output, glm_path + \
             'glm2_aroma/variance_aroma.nii.gz')
    #Threshold maps FDR
    thresholded_aroma_FDR, threshold_aroma_FDR = threshold_stats_img(F_contrast_aroma_output, alpha=.05, height_control= 'fdr' )
    #save resulting z-maps (unthresholded)
    nib.save(thresholded_aroma_FDR, glm_path + \
             'glm2_aroma/variance_aroma_fdr_corrected.nii.gz')
    
    #Thresholded maps FWE
    thresholded_aroma_FWE, threshold_aroma_FWE = threshold_stats_img(F_contrast_aroma_output, alpha = .05, height_control = 'bonferroni')   
    nib.save(thresholded_aroma_FWE, glm_path + \
             'glm2_aroma/variance_aroma_fwe_corrected.nii.gz')
    
    #GLM acompcor
    column_names_glm3 = columns_acompcor + column_constant
    regressors_glm3 = acompcor_noise_reg + [constant]
    design_glm3 = pd.DataFrame(regressors_glm3)
    design_glm3 = design_glm3.T
    design_glm3.index =frame_times
    design_glm3.columns = column_names_glm3
    contrast_matrix = np.eye(design_glm3.shape[1])
    F_contrast_acompcor= contrast_matrix[:contrast_length_acompcor]
    
    glm_output = melodic_GLM.fit(func_data, design_matrices=design_glm3)
    F_contrast_acompcor_output = glm_output.compute_contrast([F_contrast_acompcor], stat_type= 'F')
    #save resulting z-maps (unthresholded)
    nib.save(F_contrast_acompcor_output, glm_path + \
             'glm3_acompcor/variance_acompcor.nii.gz')
    #Threshold maps FDR
    thresholded_acompcor_FDR, threshold_acompcor_FDR = threshold_stats_img(F_contrast_acompcor_output, alpha=.05, height_control= 'fdr' )
    #save resulting z-maps (unthresholded)
    nib.save(thresholded_acompcor_FDR, glm_path + \
             'glm3_acompcor/variance_acompcor_fdr_corrected.nii.gz')
    
    #Thresholded maps FWE
    thresholded_acompcor_FWE, threshold_acompcor_FWE = threshold_stats_img(F_contrast_acompcor_output, alpha = .05, height_control = 'bonferroni')   
    nib.save(thresholded_acompcor_FWE, glm_path + \
             'glm3_acompcor/variance_acompcor_fwe_corrected.nii.gz')
    
    #GLM acompcor + aroma
    column_names_glm4 = columns_aroma + columns_acompcor + column_constant
    regressors_glm4 = aroma_noise_reg + acompcor_noise_reg + [constant]
    design_glm4 = pd.DataFrame(regressors_glm4)
    design_glm4 = design_glm4.T
    design_glm4.index =frame_times
    design_glm4.columns = column_names_glm4
    
    contrast_matrix = np.eye(design_glm4.shape[1])
    F_contrast_aroma_unique = contrast_matrix[:contrast_length_aroma]
    F_contrast_acompcor_unique = contrast_matrix[contrast_length_aroma:-1]
    F_contrast_shared = contrast_matrix[:contrast_length_aroma+contrast_length_acompcor]

    glm_output = melodic_GLM.fit(func_data, design_matrices=design_glm4)

    #Create contrast matrix for F-tests


    #Compute contrasts (for the shared contrast the choice of GLM is redundant)
    F_contrast_aroma_output = glm_output.compute_contrast([F_contrast_aroma_unique], stat_type= 'F')
    F_contrast_acompcor_output = glm_output.compute_contrast([F_contrast_acompcor_unique], stat_type= 'F')
    F_contrast_shared = glm_output.compute_contrast([F_contrast_shared], stat_type= 'F')

    #save resulting z-maps (unthresholded)
    nib.save(F_contrast_aroma_output, glm_path + \
             'glm4_aroma_acompcor/unique_variance_aroma.nii.gz')
    nib.save(F_contrast_acompcor_output, glm_path + \
             'glm4_aroma_acompcor/unique_variance_acompcor.nii.gz')
    nib.save(F_contrast_shared, glm_path + \
             'glm4_aroma_acompcor/shared_variance_aroma_acompcor.nii.gz')

    #Threshold maps FDR
    thresholded_aroma_FDR, threshold_aroma_FDR = threshold_stats_img(F_contrast_aroma_output, alpha=.05, height_control= 'fdr' )
    thresholded_acompcor_FDR, threshold_acompcor_FDR = threshold_stats_img(F_contrast_acompcor_output, alpha=.05, height_control= 'fdr' )
    thresholded_shared_FDR, threshold_shared_FDR = threshold_stats_img(F_contrast_shared, alpha=.05, height_control= 'fdr' )
        
    #save resulting z-maps (unthresholded)
    nib.save(thresholded_aroma_FDR, glm_path + \
             'glm4_aroma_acompcor/unique_variance_aroma_fdr_corrected.nii.gz')
    nib.save(thresholded_acompcor_FDR, glm_path + \
             'glm4_aroma_acompcor/unique_variance_acompcor_fdr_corrected.nii.gz')
    nib.save(thresholded_shared_FDR, glm_path + \
             'glm4_aroma_acompcor/shared_variance_aroma_acompcor_fdr_corrected.nii.gz')
        
    #Thresholded maps FWE
    thresholded_aroma_FWE, threshold_aroma_FWE = threshold_stats_img(F_contrast_aroma_output, alpha = .05, height_control = 'bonferroni')
    thresholded_acompcor_FWE, threshold_acompcor_FWE = threshold_stats_img(F_contrast_acompcor_output, alpha = .05, height_control = 'bonferroni')
    thresholded_shared_FWE, threshold_shared_FWE = threshold_stats_img(F_contrast_shared, alpha = .05, height_control = 'bonferroni')

    #save resulting z-maps (unthresholded)
    nib.save(thresholded_aroma_FWE, glm_path + \
             'glm4_aroma_acompcor/unique_variance_aroma_fwe_corrected.nii.gz')
    nib.save(thresholded_acompcor_FWE, glm_path + \
             'glm4_aroma_acompcor/unique_variance_acompcor_fwe_corrected.nii.gz')
    nib.save(thresholded_shared_FWE, glm_path + \
             'glm4_aroma_acompcor/shared_variance_aroma_acompcor_fwe_corrected.nii.gz')
            
        
    #GLM retroICOR on top of aroma
    column_names_glm5 = columns_retro + columns_aroma + column_constant
    regressors_glm5 = cardiac_reg + respiratory_reg + multiplication_reg + aroma_noise_reg + [constant]
    design_glm5 = pd.DataFrame(regressors_glm5)
    design_glm5 = design_glm5.T
    design_glm5.index =frame_times
    design_glm5.columns = column_names_glm5
    contrast_matrix = np.eye(design_glm5.shape[1])
    F_contrast_retro_unique = contrast_matrix[:contrast_length_retro] 
    F_contrast_aroma_unique = contrast_matrix[contrast_length_retro:-1]
    F_contrast_shared = contrast_matrix[:contrast_length_retro+contrast_length_aroma]
    
    glm_output = melodic_GLM.fit(func_data, design_matrices=design_glm5)
    

    #Compute contrasts (for the shared contrast the choice of GLM is redundant)
    F_contrast_retro_output = glm_output.compute_contrast([F_contrast_retro_unique], stat_type= 'F')
    F_contrast_aroma_output = glm_output.compute_contrast([F_contrast_aroma_unique], stat_type= 'F')
    F_contrast_shared = glm_output.compute_contrast([F_contrast_shared], stat_type= 'F')
    
    #save resulting z-maps (unthresholded)
    nib.save(F_contrast_retro_output, glm_path + \
             'glm5_retro_aroma/unique_variance_retro.nii.gz')
    nib.save(F_contrast_aroma_output, glm_path + \
             'glm5_retro_aroma/unique_variance_aroma.nii.gz')
    nib.save(F_contrast_shared, glm_path + \
             'glm5_retro_aroma/shared_variance_aroma_retro.nii.gz')
                                                           
    #Threshold maps FDR
    thresholded_retro_FDR, threshold_retro_FDR = threshold_stats_img(F_contrast_retro_output, alpha=.05, height_control= 'fdr' )
    thresholded_aroma_FDR, threshold_aroma_FDR = threshold_stats_img(F_contrast_aroma_output, alpha=.05, height_control= 'fdr' )
    thresholded_shared_FDR, threshold_shared_FDR = threshold_stats_img(F_contrast_shared, alpha=.05, height_control= 'fdr' )

    #save resulting z-maps (unthresholded)
    nib.save(thresholded_retro_FDR, glm_path + \
             'glm5_retro_aroma/unique_variance_retro_fdr_corrected.nii.gz')
    nib.save(thresholded_aroma_FDR, glm_path + \
             'glm5_retro_aroma/unique_variance_aroma_fdr_corrected.nii.gz')
    nib.save(thresholded_shared_FDR, glm_path + \
             'glm5_retro_aroma/shared_variance_aroma_retro_fdr_corrected.nii.gz')
                                                           
    #Thresholded maps FWE
    thresholded_retro_FWE, threshold_retro_FWE = threshold_stats_img(F_contrast_retro_output, alpha = .05, height_control = 'bonferroni')
    thresholded_aroma_FWE, threshold_aroma_FWE = threshold_stats_img(F_contrast_aroma_output, alpha = .05, height_control = 'bonferroni')
    thresholded_shared_FWE, threshold_shared_FWE = threshold_stats_img(F_contrast_shared, alpha = .05, height_control = 'bonferroni')

    #save resulting z-maps (unthresholded)
    nib.save(thresholded_retro_FWE, glm_path + \
             'glm5_retro_aroma/unique_variance_retro_fwe_corrected.nii.gz')
    nib.save(thresholded_aroma_FWE, glm_path + \
             'glm5_retro_aroma/unique_variance_aroma_fwe_corrected.nii.gz')
    nib.save(thresholded_shared_FWE, glm_path + \
             'glm5_retro_aroma/shared_variance_aroma_retro_fwe_corrected.nii.gz')
        
        
        
    #GLM retroICOR on top of aroma and acompcor
    regressors_glm6 = cardiac_reg + respiratory_reg + multiplication_reg + aroma_noise_reg + acompcor_noise_reg + [constant]
    column_names_glm6 = columns_retro + columns_aroma + columns_acompcor + column_constant
    design_glm6 = pd.DataFrame(regressors_glm6)
    design_glm6 = design_glm6.T
    design_glm6.index =frame_times
    design_glm6.columns = column_names_glm6
    
    contrast_matrix = np.eye(design_glm6.shape[1])
    F_contrast_retro_unique = contrast_matrix[:contrast_length_retro] 
    F_contrast_aroma_unique = contrast_matrix[contrast_length_retro:-1]
    F_contrast_acompcor_unique = contrast_matrix[contrast_length_retro+contrast_length_aroma:-1]
    F_contrast_shared = contrast_matrix[:contrast_length_retro+contrast_length_aroma+contrast_length_acompcor]
    
    
    glm_output = melodic_GLM.fit(func_data, design_matrices=design_glm6)
    
    #Create contrast matrix for F-tests

    #Compute contrasts (for the shared contrast the choice of GLM is redundant)
    F_contrast_retro_output = glm_output.compute_contrast([F_contrast_retro_unique], stat_type= 'F')
    F_contrast_aroma_output = glm_output.compute_contrast([F_contrast_aroma_unique], stat_type= 'F')
    F_contrast_acompcor_output = glm_output.compute_contrast([F_contrast_acompcor_unique], stat_type= 'F')
    F_contrast_shared = glm_output.compute_contrast([F_contrast_shared], stat_type= 'F')
    
    #save resulting z-maps (unthresholded)
    nib.save(F_contrast_retro_output, glm_path + \
             'glm6_retro_aroma_acompcor/unique_variance_retro.nii.gz')
    nib.save(F_contrast_aroma_output, glm_path + \
             'glm6_retro_aroma_acompcor/unique_variance_aroma.nii.gz')
    nib.save(F_contrast_acompcor_output, glm_path + \
             'glm6_retro_aroma_acompcor/unique_variance_acompcor.nii.gz')
    nib.save(F_contrast_shared, glm_path + \
             'glm6_retro_aroma_acompcor/shared_variance_aroma_retro_acompcor.nii.gz')
    
                                                       
    #Threshold maps FDR
    thresholded_retro_FDR, threshold_retro_FDR = threshold_stats_img(F_contrast_retro_output, alpha=.05, height_control= 'fdr' )
    thresholded_aroma_FDR, threshold_aroma_FDR = threshold_stats_img(F_contrast_aroma_output, alpha=.05, height_control= 'fdr' )
    thresholded_acompcor_FDR, threshold_acompcor_FDR = threshold_stats_img(F_contrast_acompcor_output, alpha=.05, height_control= 'fdr' )
    thresholded_shared_FDR, threshold_shared_FDR = threshold_stats_img(F_contrast_shared, alpha=.05, height_control= 'fdr' )

    #save resulting z-maps (unthresholded)
    nib.save(thresholded_retro_FDR, glm_path + \
             'glm6_retro_aroma_acompcor/unique_variance_retro_fdr_corrected.nii.gz')
    nib.save(thresholded_aroma_FDR, glm_path + \
             'glm6_retro_aroma_acompcor/unique_variance_aroma_fdr_corrected.nii.gz')
    nib.save(thresholded_acompcor_FDR, glm_path + \
             'glm6_retro_aroma_acompcor/unique_variance_acompcor_fdr_corrected.nii.gz')
    nib.save(thresholded_shared_FDR, glm_path + \
             'glm6_retro_aroma_acompcor/shared_variance_aroma_retro_acompcor_fdr_corrected.nii.gz')
                                                           
    #Thresholded maps FWE
    thresholded_retro_FWE, threshold_retro_FWE = threshold_stats_img(F_contrast_retro_output, alpha = .05, height_control = 'bonferroni')
    thresholded_aroma_FWE, threshold_aroma_FWE = threshold_stats_img(F_contrast_aroma_output, alpha = .05, height_control = 'bonferroni')
    thresholded_acompcor_FWE, threshold_aroma_FWE = threshold_stats_img(F_contrast_acompcor_output, alpha = .05, height_control = 'bonferroni')
    thresholded_shared_FWE, threshold_shared_FWE = threshold_stats_img(F_contrast_shared, alpha = .05, height_control = 'bonferroni')

    #save resulting z-maps (unthresholded)
    nib.save(thresholded_retro_FWE, glm_path + \
             'glm6_retro_aroma_acompcor/unique_variance_retro_fwe_corrected.nii.gz')
    nib.save(thresholded_aroma_FWE, glm_path + \
             'glm6_retro_aroma_acompcor/unique_variance_aroma_fwe_corrected.nii.gz')    
    nib.save(thresholded_acompcor_FWE, glm_path + \
             'glm6_retro_aroma_acompcor/unique_variance_acompcor_fwe_corrected.nii.gz')
    nib.save(thresholded_shared_FWE, glm_path + \
             'glm6_retro_aroma_acompcor/shared_variance_aroma_retro_fwe_corrected.nii.gz')
    
