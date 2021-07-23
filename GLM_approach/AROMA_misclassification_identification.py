"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

The function of this script is to compile and sort the MELODIC component matching results.
For every melodic component a spatial overlap was calculated with a binarised z-map of an F-contrast
exploring unique variance of RETROICOR with AROMA components in the model (created with Comparison_FMAP_MELODIC.py).
Resulting overlap can be taken as a quantitative approach to identify misclassification of AROMA-signal components
and will be used to identify the potential for model additions after identification.

"""

import glob
import numpy as np
import nibabel as nib
import pandas as pd
from Subject_Class import Subject
from nilearn import glm
from nilearn import plotting
from nilearn.glm import threshold_stats_img
import matplotlib.pyplot as plt
from scipy.stats import norm

summary_files = glob.glob('/project/3013068.03/RETROICOR/Example_Visualisation/sub-*/Melodic_Matching_corrected/*summary.txt')
summary_files.sort()
summary_files = summary_files[:2]
summary_list = []

# Indicating subject having the 'stress' condition during their FIRST functional session
stress_list = ['sub-002', 'sub-003', 'sub-004', 'sub-007', 'sub-009', 'sub-013', 'sub-015', 'sub-017', 'sub-021', 'sub-023', 'sub-025', 'sub-027', 'sub-029']

#Compiling of all summary files
for summary_counter, summary in enumerate(summary_files):
    fit_matrix = pd.read_csv(summary, index_col=0)
    fit_matrix.sort_values('Goodness of Fit', ascending=False, kind='stable', inplace=True)
    summary_list.append(fit_matrix)


for summary_counter, summary in enumerate(summary_files):
    
    #Invoke Subject_Class to allow access to all necessary data
    subject_id = summary[summary.find('sub-'): summary.find('sub-')+7]
    subject_obj = Subject(subject_id)
        
    # Account for balancing in stress/control session order
    if subject_id in stress_list:
        ses_nr = 2
    elif subject_id not in stress_list:
        ses_nr = 1
    
    #Load melodic mixing matrix
    melodic_mixing_matrix = pd.read_csv(glob.glob('/project/3013068.03/derivate/fmriprep/{0}/ses-mri0{1}/'\
                                                  'func/{0}_ses-mri0{1}_task'\
                                                  '-15isomb3TR2020TE28RS*run-2_echo-1_desc-'\
                                                  'MELODIC_mixing.tsv'.format(subject_id,ses_nr+1))[0],\
                                        header=None, sep='\t')
    potential_misclass_id = []
    potential_misclass_matrix = []
    #Identification of potential misclassification
    for row in summary_list[0].iterrows():
        if row[1]['Goodness of Fit'] > 0.75 and row[1]['Component Classification'] == 'Signal':
            potential_misclass_id.append(row[0])
            potential_misclass_matrix.append(melodic_mixing_matrix[row[0]])
    
    # Base scan settings for the used sequence
    t_r = 2.02
    n_scans = 240
    constant = [1.0]*240
    frame_times = np.arange(n_scans)*t_r

    # Standard MNI mask used for masking
    mni_mask = '/project/3013068.03/RETROICOR/MNI152lin_T1_2mm_brain_mask.nii.gz'

    #GLM settings
    melodic_GLM = glm.first_level.FirstLevelModel(t_r=2.02, slice_time_ref=0.5, smoothing_fwhm=6, \
                                                drift_model=None, hrf_model=None, mask_img= mni_mask, verbose=1)


    #Loading respective functional run as NII-img-like nibabel object
    func_data = subject_obj.get_func_data(session=ses_nr,run=2,task='RS', MNI=True)

    #All file-paths to the respective regressor files as created by 'Confound_file_creation_AROMA_model.py'
    cardiac_phase = glob.glob('/project/3013068.03/RETROICOR/Example_Visualisation/{0}/'\
                              '3C4R1M_vs_AROMA_corrected/cardiac*'.format(subject_id))
    
    respiratory_phase = glob.glob('/project/3013068.03/RETROICOR/Example_Visualisation/{0}/'\
                                  '3C4R1M_vs_AROMA_corrected/respiratory*'.format(subject_id))
    
    multiplication_phase = glob.glob('/project/3013068.03/RETROICOR/Example_Visualisation/{0}/'\
                                     '3C4R1M_vs_AROMA_corrected/multiplication*'.format(subject_id))
    multiplication_phase.sort()
    
    aroma_noise = glob.glob('/project/3013068.03/RETROICOR/Example_Visualisation/{0}/'\
                            '3C4R1M_vs_AROMA_corrected/aroma*'.format(subject_id))
    aroma_noise.sort()
    
    #Column names for 3C4R1M RETROICOR
    columns_RETRO = ['Car_sin_01', 'Car_cos_01', 'Car_sin_02', 'Car_cos_02', 'Car_sin_03', 'Car_cos_03', \
                     'Resp_sin_01', 'Resp_cos_01', 'Resp_sin_02', 'Resp_cos_02', 'Resp_sin_03', \
                     'Resp_cos_03', 'Resp_sin_04', 'Resp_cos_04', 'Mult_01', 'Mult_02', 'Mult_03', 'Mult_04']
    
    #Column names for melodic components
    columns_AROMA = ['AR_' + x[-2:] for x in aroma_noise]
    
    #Column name constant
    column_constant = ['Constant']
    
    
    for component_id, component in enumerate(potential_misclass_matrix):
        
        #Regressor list
        regressors = [np.array(component)] + \
        [np.loadtxt(x) for x in cardiac_phase[:6]] + \
        [np.loadtxt(x) for x in respiratory_phase[:8]] + \
        [np.loadtxt(x) for x in multiplication_phase] + \
        [np.loadtxt(x) for x in aroma_noise] + \
        [constant]

        #Joint column names
        column_names = ['AddComp_' + str(potential_misclass_id[component_id]+1)] + columns_RETRO + columns_AROMA + column_constant

        #Create design matrix as used by nilearn(v0.8).
        design = pd.DataFrame(regressors)
        design = design.T
        design.index =frame_times
        design.columns = column_names
        
        #compute SEPERATE glm using the specified design matrix (yeah yeah I know..)
        glm_output = melodic_GLM.fit(func_data, design_matrices=design)
    
        #Create contrast matrix for F-tests
        contrast_length_retro = 18
        contrast_matrix = np.eye(design.shape[1])
        F_RETRO_unique_added_comp = contrast_matrix[1:contrast_length_retro+1] 

        #Compute contrasts (for the shared contrast the choice of GLM is redundant)
        F_RETRO_unique_added_comp_output = glm_output.compute_contrast([F_RETRO_unique_added_comp], stat_type='F')
        
        #Save resulting z-maps (unthresholded)
        nib.save(F_RETRO_unique_added_comp_output, '/project/3013068.03/RETROICOR/Example_Visualisation/{0}/'\
                 'Melodic_Matching_corrected/potential_misclassfications/{1}.nii.gz'.format(subject_id,'AddComp_' + str(potential_misclass_id[component_id]+1)))
        
        #Thresholded maps FWE
        thresholded_RETRO_FWE, threshold_RETRO_FWE = threshold_stats_img(F_RETRO_unique_added_comp_output, alpha=.05, height_control='bonferroni')
        
        #Save resulting z-maps FWE-thresholded 0.05
        nib.save(thresholded_RETRO_FWE, '/project/3013068.03/RETROICOR/Example_Visualisation/{0}/'\
             'Melodic_Matching_corrected/potential_misclassfications/{1}_fwe_corrected.nii.gz'.format(subject_id,'AddComp_' + str(potential_misclass_id[component_id]+1)))

        #Plot thresholded results
        plotting.plot_glass_brain(thresholded_RETRO_FWE, colorbar=True, threshold=None, \
                                  title=subject_id + 'AddComp: ' + str(potential_misclass_id[component_id]+1),\
                                  output_file = '/project/3013068.03/RETROICOR/Example_Visualisation/{0}/'\
                                  'Melodic_Matching_corrected/potential_misclassfications/{1}_fwe_corrected.png'\
                                  .format(subject_id,'AddComp_' + str(potential_misclass_id[component_id]+1)), \
                                  plot_abs=False)
        plt.close()
