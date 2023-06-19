#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 15:46:30 2021

@author: markre
"""

import glob
from Subject_Class_new import Subject
import pandas as pd

FMRIPREP_PATH = "/project/3013068.03/fmriprep_test/"
BASEPATH = '/project/3013068.03/physio_revision/GLM_approach/'

part_list = glob.glob(BASEPATH + 'sub-*')
part_list.sort()

# Indicating subject having the 'stress' condition during their FIRST functional session
stress_list = ['sub-002', 'sub-003', 'sub-004', 'sub-007', 'sub-009', 'sub-013', 'sub-015', 'sub-017', 'sub-021',
               'sub-023', 'sub-025', 'sub-027', 'sub-029']

for subject_long in part_list:
    sub_id = subject_long[-7:]
    sub = Subject(sub_id)

    ses_nr = 2 if sub_id in stress_list else 1

    # LOAD ALL FMRIPREP CONFOUNDS
    fmriprep_confounds = sub.get_confounds(session=ses_nr, run=2, task='RS')

    # SELECT COSINES FOR TEMPORAL DETRENDING
    fmriprep_confound_selection = fmriprep_confounds[['cosine00', 'cosine01', 'cosine02', 'cosine03', 'cosine04',
                                                      'cosine05']].copy()
    output_confound_file = BASEPATH + f'{sub_id}/confounds/rs_confounds_{sub_id}_ses-0{ses_nr}_run-02_cosines.csv'
    fmriprep_confound_selection.to_csv(output_confound_file, sep=' ', header=False, index=False)

    # SELECTION OF RELEVANT ACOMPCOR COMPONENTS
    acompcor_confounds_trimmed = sub.get_acompcor_confounds(session=ses_nr, run=2, task='RS', number_regressors=5)
    output_acompcor_file = BASEPATH + f'{sub_id}/confounds/rs_confounds_{sub_id}_ses-0{ses_nr}_run-02_acompcor.csv'
    acompcor_confounds_trimmed.to_csv(output_acompcor_file,
                                      sep=' ',
                                      header=False,
                                      index=False)

    for acompcor_confound in acompcor_confounds_trimmed:
        acompcor_confounds_trimmed[acompcor_confound].to_csv(BASEPATH + f'{sub_id}/confounds/{acompcor_confound}')

    # SELECTION OF AROMA COMPONENTS
    mixing = sub.get_aroma_confounds(session=ses_nr, run=2, task='RS')
    mixing_output_file = BASEPATH + f'{sub_id}/confounds/rs_confounds_{sub_id}_ses-0{ses_nr}_run-02_aroma.csv'
    mixing.to_csv(mixing_output_file,
                  sep=' ',
                  header=False,
                  index=False)

    for aroma_confound in mixing:
        mixing[aroma_confound].to_csv(BASEPATH + f'{sub_id}/confounds/{aroma_confound}',
                                      sep=' ',
                                      header=False,
                                      index=False)

    # COMBINATION OF ACOMPCOR AND AROMA
    combination_confounds_mixing = pd.concat([fmriprep_confound_selection, mixing], axis=1)
    output_confound_file = BASEPATH + f'{sub_id}/confounds/rs_confounds_{sub_id}_ses-0{ses_nr}_run-02_cosines_aroma.csv'

    combination_confounds_mixing.to_csv(output_confound_file, sep=' ', header=False, index=False)
    # COMBINATION OF COSINES AND ACOMPCOR
    combination_confounds_aCompCor = pd.concat([fmriprep_confound_selection, acompcor_confounds_trimmed], axis=1)
    output_aCompCor_file_2 = BASEPATH + f'{sub_id}/confounds/rs_confounds_{sub_id}_ses-0{ses_nr}_' \
                                        f'run-02_cosines_aCompCor.csv'

    combination_confounds_aCompCor.to_csv(output_aCompCor_file_2, sep=' ', header=False, index=False)
    
    # RETROICOR SELECTION
    sub_physio = sub.get_physio(session=ses_nr, run=2, task='RS')

    for regressor in sub_physio:
        sub_physio[regressor].to_csv(BASEPATH + f'{sub_id}/confounds/{regressor}.csv')

    sub_mult = sub.get_retroicor_confounds(session=ses_nr, run=2, task='RS')
    for multiplication in sub_mult[sub_mult.columns[-4:]]:
        sub_mult[sub_mult.columns[-4:]][multiplication].to_csv(BASEPATH + f'{sub_id}/confounds/{multiplication}.csv',
                                                               sep=' ',
                                                               header=False,
                                                               index=False)
