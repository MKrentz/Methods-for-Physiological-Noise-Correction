#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 15:46:30 2021

@author: markre
"""

import numpy as np
import glob
from Subject_Class import Subject
import pandas as pd

FMRIPREP_PATH = "/project/3013068.03/derivate/fmriprep/"
BASEPATH = '/project/3013068.03/test/GLM_approach/'

part_list = glob.glob(BASEPATH + 'sub-*')
part_list.sort()

# Indicating subject having the 'stress' condition during their FIRST functional session
stress_list = ['sub-002', 'sub-003', 'sub-004', 'sub-007', 'sub-009', 'sub-013', 'sub-015', 'sub-017', 'sub-021',
               'sub-023', 'sub-025', 'sub-027', 'sub-029']

for subject_long in part_list:
    sub_id = subject_long[-7:]
    sub = Subject(sub_id)

    ses_nr = 3 if sub_id in stress_list else 2

    fmriprep_confounds = sub.get_confounds(session = ses_nr-1, run = 2, task = 'RS')

    # Selection of nuisance regressors from fmriprep-confound file
    fmriprep_confound_selection = fmriprep_confounds[['cosine00', 'cosine01', 'cosine02', 'cosine03', 'cosine04',
                                                      'cosine05']]

    fmriprep_confounds_trimmed = fmriprep_confound_selection

    output_confound_file = BASEPATH + '{0}/confounds/Resting_State_confounds_all_{0}_ses-0{1}_run-02.txt'.format(sub_id, str(ses_nr))
    fmriprep_confounds_trimmed.to_csv(output_confound_file, sep = ' ', header = None, index = False)

    #Selection of aCompCor components
    aCompCor_selection = fmriprep_confounds[['a_comp_cor_01', 'a_comp_cor_02', 'a_comp_cor_03', 'a_comp_cor_04', 'a_comp_cor_05']]
    aCompCor_confounds_trimmed = aCompCor_selection
    output_aCompCor_file = BASEPATH + '{0}/confounds/aCompCor_all_{0}_ses-0{1}_run-02.txt'.format(sub_id, str(ses_nr))
    aCompCor_confounds_trimmed.to_csv(output_aCompCor_file, sep = ' ', header = None, index = False)

    # Addition of non-aggressive AROMA signal components to nuisance file for in-model non-aggressive AROMA
    aroma_noise_ics = glob.glob(FMRIPREP_PATH + '{0}/ses-mri0{1}/func/*RS*run-2*AROMAnoiseICs*'.format(sub_id, str(ses_nr)))[0]
    aroma_noise_ics_text = np.loadtxt(aroma_noise_ics, delimiter = ',', dtype = int).tolist()
    aroma_noise_ics_text = [ic - 1 for ic in aroma_noise_ics_text]
    mixing = pd.read_csv(glob.glob(FMRIPREP_PATH + '{0}/ses-mri0{1}/func/*RS_*run-2*MELODIC_mixing.tsv'.format(sub_id, str(ses_nr)))[0], sep = '\t', header = None)
    mixing = mixing.drop(aroma_noise_ics_text, 1)
    mixing_output_file = BASEPATH + \
                         '{0}/confounds/Resting_State_confounds_all_{0}_ses-0{1}_run-02_nonaggrAROMA.txt'.format(sub_id, str(ses_nr))
    mixing.to_csv(mixing_output_file, sep = ' ', header = None, index = False)
    
    combination_confounds_mixing = pd.concat([fmriprep_confounds_trimmed,mixing], axis = 1)
    output_confound_file = BASEPATH + \
                           '{0}/confounds/Resting_State_confounds_all_{0}_ses-0{1}_run-02_combined+nonaggrAROMA.txt'.format(sub_id, str(ses_nr))
    combination_confounds_mixing.to_csv(output_confound_file, sep = ' ', header = None, index = False)
    
    # Addition of aCompCor
    combination_confounds_aCompCor = pd.concat([fmriprep_confounds_trimmed, aCompCor_confounds_trimmed], axis = 1)
    output_aCompCor_file_2 = BASEPATH + \
       '{0}/confounds/Resting_State_confounds_all_{0}_ses-0{1}_run-02_combined+aCompCor.txt'.format(sub_id, str(ses_nr))
    combination_confounds_aCompCor.to_csv(output_aCompCor_file_2, sep = ' ', header = None, index = False)
    
    # Retroicor Part
    sub_physio = sub.get_physio(session = ses_nr-1, run = 2, task = 'RS')
    sub_physio_np = np.array(sub_physio)
    
    sub_physio_columns = sub_physio.columns.tolist()
    for regressor in sub_physio_columns:
        single_reg = sub_physio[regressor]
        np.savetxt(BASEPATH + '{0}/confounds/{1}'.format(sub_id, regressor), single_reg)

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
        np.savetxt(BASEPATH + '{0}/confounds/multiplication_term_CPRP_0{1}.txt'.format(sub_id, mult_counter + 1), multiplications)
    
    # Aroma Part
    sub_confounds = sub.get_confounds(session = ses_nr - 1, run = 2, task = 'RS')
    confounds_column_index = sub_confounds.columns.tolist()
    aroma_sum = sum((itm.count("aroma_motion") for itm in confounds_column_index))
    aroma_variables = confounds_column_index[-aroma_sum:]
    for aroma_confounds in aroma_variables:
        confound_selection = sub_confounds[aroma_confounds]
        np.savetxt(BASEPATH + '{0}/confounds/{1}'.format(sub_id , aroma_confounds), confound_selection)
        
    #aCompCor Part
    for aCompCor_confounds in aCompCor_confounds_trimmed:
        aCompCor_c = aCompCor_confounds_trimmed[aCompCor_confounds]
        np.savetxt(BASEPATH + '{0}/confounds/{1}'.format(sub_id , aCompCor_confounds), aCompCor_c)