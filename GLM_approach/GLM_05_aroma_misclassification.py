"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

@author: markre
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

BASEPATH = '/project/3013068.03/RETROICOR/Example_Visualisation/'

# Pools all available summary files
summary_files = glob.glob(BASEPATH + 'sub-*/Melodic_Matching_corrected/*summary.txt')
summary_files.sort()
summary_list = []

# Indicating subject having the 'stress' condition during their FIRST functional session
stress_list = ['sub-002', 'sub-003', 'sub-004', 'sub-007', 'sub-009', 'sub-013', 'sub-015', 'sub-017', 'sub-021', 'sub-023', 'sub-025', 'sub-027', 'sub-029']

# Compiling of all summary files
for summary_counter, summary in enumerate(summary_files):
    fit_matrix = pd.read_csv(summary, index_col=0)
    fit_matrix.sort_values('Goodness of Fit', ascending=False, kind='stable', inplace=True)

    # Invoke Subject_Class to allow access to all necessary data
    sub_id = summary[summary.find('sub-'): summary.find('sub-')+7]
    sub_obj = Subject(sub_id)
        
    # Account for balancing in stress/control session order
    ses_nr = 2 if sub_id in stress_list else 1
    
    # Load melodic mixing matrix
    melodic_mixing_matrix = pd.read_csv(glob.glob('/project/3013068.03/derivate/fmriprep/{0}/ses-mri0{1}/'\
                                                  'func/{0}_ses-mri0{1}_task'\
                                                  '-15isomb3TR2020TE28RS*run-2_echo-1_desc-'\
                                                  'MELODIC_mixing.tsv'.format(sub_id, ses_nr+1))[0],
                                        header = None, sep = '\t')
    potential_misclass_id = []
    potential_misclass_matrix = []

    # Identification of potential misclassification
    for row in fit_matrix.iterrows():
        if row[1]['Goodness of Fit'] > 0.75 and row[1]['Component Classification'] == 'Signal':
            print(row[0])
            potential_misclass_id.append(row[0])
            potential_misclass_matrix.append(melodic_mixing_matrix[row[0]])
    
    # Base scan settings for the used sequence
    t_r = 2.02
    n_scans = 240
    constant = [1.0] * 240
    frame_times = np.arange(n_scans) * t_r

    # Standard MNI mask used for masking
    mni_mask = '/project/3013068.03/RETROICOR/MNI152lin_T1_2mm_brain_mask.nii.gz'

    # GLM settings
    melodic_GLM = glm.first_level.FirstLevelModel(t_r = 2.02,
                                                  slice_time_ref = 0.5,
                                                  smoothing_fwhm = 6,
                                                  drift_model = None,
                                                  hrf_model = None,
                                                  mask_img = mni_mask,
                                                  verbose = 1)


    # Loading respective functional run as NII-img-like nibabel object
    func_data = sub_obj.get_func_data(session = ses_nr, run =2 , task = 'RS', MNI = True)

    # All file-paths to the respective regressor files as created by 'Confound_file_creation_AROMA_model.py'
    cardiac_phase = glob.glob(BASEPATH + '{0}/'\
                              '3C4R1M_vs_AROMA_corrected/cardiac*'.format(sub_id))
    
    respiratory_phase = glob.glob(BASEPATH + '{0}/'\
                                  '3C4R1M_vs_AROMA_corrected/respiratory*'.format(sub_id))
    
    multiplication_phase = glob.glob(BASEPATH + '{0}/'\
                                     '3C4R1M_vs_AROMA_corrected/multiplication*'.format(sub_id))
    multiplication_phase.sort()
    
    aroma_noise = glob.glob(BASEPATH + '{0}/'\
                            '3C4R1M_vs_AROMA_corrected/aroma*'.format(sub_id))
    aroma_noise.sort()
    
    # Column names for 3C4R1M RETROICOR
    columns_RETRO = ['Car_sin_01', 'Car_cos_01', 'Car_sin_02', 'Car_cos_02', 'Car_sin_03', 'Car_cos_03',
                     'Resp_sin_01', 'Resp_cos_01', 'Resp_sin_02', 'Resp_cos_02', 'Resp_sin_03',
                     'Resp_cos_03', 'Resp_sin_04', 'Resp_cos_04', 'Mult_01', 'Mult_02', 'Mult_03', 'Mult_04']
    
    # Column names for melodic components
    columns_AROMA = ['AR_' + x[-2:] for x in aroma_noise]
    
    # Column name constant
    column_constant = ['Constant']

    # Account for subject without potential misclassification
    if potential_misclass_matrix:
        for component_id, component in enumerate(potential_misclass_matrix):

            # Regressor list
            regressors = [np.array(component)] + \
            [np.loadtxt(x) for x in cardiac_phase[:6]] + \
            [np.loadtxt(x) for x in respiratory_phase[:8]] + \
            [np.loadtxt(x) for x in multiplication_phase] + \
            [np.loadtxt(x) for x in aroma_noise] + \
            [constant]

            # Joint column names
            column_names = ['AddComp_' + str(potential_misclass_id[component_id] + 1)] + columns_RETRO + columns_AROMA + column_constant
            print('AddComp_' + str(potential_misclass_id[component_id] + 1))
            # Create design matrix as used by nilearn(v0.8).
            design = pd.DataFrame(regressors)
            design = design.T
            design.index =frame_times
            design.columns = column_names

            # Compute SEPERATE glm using the specified design matrix (yeah yeah I know..)
            glm_output = melodic_GLM.fit(func_data, design_matrices=design)

            # Create contrast matrix for F-tests
            contrast_length_retro = 18
            contrast_matrix = np.eye(design.shape[1])
            F_RETRO_unique_added_comp = contrast_matrix[1:contrast_length_retro + 1]

            # Compute contrasts (for the shared contrast the choice of GLM is redundant)
            F_RETRO_unique_added_comp_output = glm_output.compute_contrast([F_RETRO_unique_added_comp], stat_type = 'F')

            # Save resulting z-maps (unthresholded)
            nib.save(F_RETRO_unique_added_comp_output, BASEPATH + '{0}/'\
                     'Melodic_Matching_corrected/potential_misclassifications/{1}_uncorrected.nii.gz'.format(sub_id, 'AddComp_' + str(potential_misclass_id[component_id] + 1)))

            # Thresholded maps FWE
            thresholded_RETRO_FWE, threshold_RETRO_FWE = threshold_stats_img(F_RETRO_unique_added_comp_output,
                                                                             alpha = .05,
                                                                             height_control = 'bonferroni')

            # Save resulting z-maps FWE-thresholded 0.05
            nib.save(thresholded_RETRO_FWE, BASEPATH + '{0}/'\
                 'Melodic_Matching_corrected/potential_misclassifications/{1}_fwe_corrected.nii.gz'.format(sub_id,
                                                                                                           'AddComp_' + str(potential_misclass_id[component_id]+1)))
            # Thresholded maps FDR
            thresholded_RETRO_FWE, threshold_RETRO_FWE = threshold_stats_img(F_RETRO_unique_added_comp_output,
                                                                             alpha = .05,
                                                                             height_control = 'fdr')

            # Save resulting z-maps FDR-thresholded 0.05
            nib.save(thresholded_RETRO_FWE, BASEPATH + '{0}/'\
                 'Melodic_Matching_corrected/potential_misclassifications/{1}_fdr_corrected.nii.gz'.format(sub_id,
                                                                                                           'AddComp_' + str(potential_misclass_id[component_id]+1)))

            # Plot thresholded results
            plotting.plot_glass_brain(thresholded_RETRO_FWE,
                                      colorbar = True,
                                      threshold = None,
                                      title = sub_id + 'AddComp: ' + str(potential_misclass_id[component_id] + 1),
                                      output_file = BASEPATH + '{0}/'\
                                      'Melodic_Matching_corrected/potential_misclassifications/{1}_fwe_corrected.png'\
                                      .format(sub_id, 'AddComp_' + str(potential_misclass_id[component_id] + 1)),
                                      plot_abs = False)
            plt.close()
