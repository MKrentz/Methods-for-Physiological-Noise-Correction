#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 15:46:30 2021

@author: markre
"""

import numpy as np
import glob
from Subject_Class import Subject
import pandas as pd

part_list = glob.glob('/project/3013068.03/RETROICOR/Example_Visualisation/sub-*')
part_list.sort()
#participant_alt_list = ['sub_001','sub_002', 'sub_003', 'sub_004', 'sub_005', 'sub_006', 'sub_007', 'sub_008', 'sub_009',  'sub_012','sub_013', 'sub_014', 'sub_015', 'sub_016', 'sub_017', 'sub_018', 'sub_020', 'sub_021','sub_022', 'sub_023', 'sub_024', 'sub_025', 'sub_026', 'sub_027', 'sub_028', 'sub_029', 'sub_030']

for subject_long in part_list:    
    subject = subject_long[-7:]
    sub = Subject(subject)
    
    fmriprep_confounds = sub.get_confounds(session=2,run=2,task='RS')
    
    #Selection of nuisance regressors from fmriprep-confound file
    fmriprep_confound_selection = fmriprep_confounds[['cosine00', 'cosine01', 'cosine02', 'cosine03', 'cosine04', 'cosine05']]

    fmriprep_confounds_trimmed = fmriprep_confound_selection

    output_confound_file = '/project/3013068.03/RETROICOR/Example_Visualisation/{0}/3C4R1M_vs_AROMA/Resting_State_confounds_all_{0}_ses-03_run-02.txt'.format(subject)
    fmriprep_confounds_trimmed.to_csv(output_confound_file, sep=' ', header=None, index=False)

    #Addition of non-aggressive RETORICOR signal components to nuisance file for in-model non-aggressive AROMA
    aroma_noise_ics = glob.glob('/project/3013068.03/derivate/fmriprep/{0}/ses-mri03/func/*RS*run-2*AROMAnoiseICs*'.format(subject))[0]
    aroma_noise_ics_text  = np.loadtxt(aroma_noise_ics, delimiter=',', dtype=int).tolist()
    aroma_noise_ics_text  = [ic-1 for ic in aroma_noise_ics_text]
    mixing = pd.read_csv(glob.glob('/project/3013068.03/derivate/fmriprep/{0}/ses-mri03/func/*RS_*run-2*MELODIC_mixing.tsv'.format(subject))[0], sep='\t', header=None)
    mixing = mixing.drop(aroma_noise_ics_text, 1)
    mixing_output_file = '/project/3013068.03/RETROICOR/Example_Visualisation/{0}/3C4R1M_vs_AROMA/Resting_State_confounds_all_{0}_ses-03_run-02_nonaggrAROMA.txt'.format(subject)
    mixing.to_csv(mixing_output_file, sep=' ', header=None, index=False)
    
    combination_confounds_mixing = pd.concat([fmriprep_confounds_trimmed,mixing], axis=1)
    combination_confounds_mixing.to_csv()
    
    output_confound_file = '/project/3013068.03/RETROICOR/Example_Visualisation/{0}/3C4R1M_vs_AROMA/Resting_State_confounds_all_{0}_ses-03_run-02_combined+nonaggrAROMA.txt'.format(subject)
    combination_confounds_mixing.to_csv(output_confound_file, sep=' ', header=None, index=False)
    
    #Retroicor Part
    sub_physio = sub.get_physio(session=2,run=2,task='RS')
    sub_physio_np = np.array(sub_physio)
    
    sub_physio_columns = sub_physio.columns.tolist()
    for regressor in sub_physio_columns:
        single_reg = sub_physio[regressor]
        np.savetxt('/project/3013068.03/RETROICOR/Example_Visualisation/{0}/3C4R1M_vs_AROMA/{1}'.format(subject, regressor), single_reg)
    
    
    # Multiplication Term
    multi_list = []
    multi_01 = sub_physio_np[:,0] * sub_physio_np[:,10]
    multi_list.append(multi_01)
    multi_02 = sub_physio_np[:,0] * sub_physio_np[:,11]
    multi_list.append(multi_02)
    multi_03 = sub_physio_np[:,1] * sub_physio_np[:,10]
    multi_list.append(multi_03)
    multi_04 = sub_physio_np[:,1] * sub_physio_np[:,11]
    multi_list.append(multi_04)
    
    for mult_counter, multiplications in enumerate(multi_list):
                np.savetxt('/project/3013068.03/RETROICOR/Example_Visualisation/{0}/3C4R1M_vs_AROMA/multiplication_term_CPRP_0{1}.txt'.format(subject, mult_counter+1), multiplications)
    
    #Aroma Part
    sub_confounds = sub.get_confounds(session=2, run=2, task='RS')
    confounds_column_index = sub_confounds.columns.tolist()
    aroma_sum = sum((itm.count("aroma_motion") for itm in confounds_column_index))
    aroma_variables = confounds_column_index[-aroma_sum:]
    for aroma_confounds in aroma_variables:
        confound_selection = sub_confounds[aroma_confounds]
        np.savetxt('/project/3013068.03/RETROICOR/Example_Visualisation/{0}/3C4R1M_vs_AROMA/{1}'.format(subject, aroma_confounds), confound_selection)
