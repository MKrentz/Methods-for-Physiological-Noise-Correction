#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 17:55:52 2021

This script implements the calculation of TSNR for different noise-cleaning procedures.

Creating different TSNR map for:
    Uncleaned data
    aroma cleaned data
    retroICOR cleaned data
    aroma AND retroICOR cleaned data

Additionally maps are created visualising the unique contributions of a method OVER another.
    Unique TSNR improvement of aroma (TSNR of aroma+retro - TSNR of retro)
    Unique TSNR improvement of retroICOR (TSNR of aroma+retro - TSNR of aroma)
    Difference in TSNR improvement between retroICOR and aroma (TSNR of aroma - TSNR of retro)
    TSNR improvement of uncleaned data for aroma (TSNR of aroma - TSNR of uncleaned data)
    TSNR improvement of uncleaned data for retroICOR (TSNR of retroICOR - TSNR of uncleaned data)

nilearn 0.10.1
@author: MKrentz
"""

import numpy as np
import nibabel as nib
import pandas as pd
from Subject_Class_new import Subject
import nilearn
from nilearn import image
import numpy.ma as ma
from nilearn.plotting import plot_design_matrix
import sys

BASEPATH = '/project/3013068.03/physio_revision_fixed/TSNR_approach/'
SAVEPATH = '/project/3013068.03/physio_revision_fixed/TSNR_approach/mean_TSNR/'
# Load MNI mask to used masked data matrices and switch 0 to 1 and 1 to 0

print(sys.argv[1])
sub_id = sys.argv[1]

# Load all available participants

# Indicating subject having the 'stress' condition during their FIRST functional session
stress_list = ['sub-002', 'sub-003', 'sub-004', 'sub-007', 'sub-009', 'sub-013', 'sub-015', 'sub-017',
               'sub-021', 'sub-023', 'sub-025', 'sub-027', 'sub-029']

# Fix the random seed for padding regressor (comment out when using an existing seed dictionary)


sub = Subject(sub_id)
ses_nr = 2 if sub_id in stress_list else 1

mni_mat = sub.get_brainmask(MNI=True, session=ses_nr, run=2).get_fdata()
mni_mat = np.where((mni_mat == 0) | (mni_mat == 1), 1 - mni_mat, mni_mat)
# Account for balancing in stress/control session order

# Loading respective functional data into memory and online-smooth with 6mm FWHM
func_data_mni = sub.get_func_data(session=ses_nr,
                                      run=2,
                                      task='RS',
                                      MNI=True)
func_data_mni = image.smooth_img(func_data_mni,
                                 fwhm=6)
func_data_native = sub.get_func_data(session=ses_nr,
                                         run=2,
                                         task='RS',
                                         MNI=False)
func_data_native = image.smooth_img(func_data_native,
                                    fwhm=6)

# Load fmriprep confound files for respective runs, T1-brainmask and retroICOR regressors
sub_confounds = sub.get_confounds(session=ses_nr, run=2, task='RS')
sub_brainmask = sub.get_brainmask(session=ses_nr, run=2, MNI=False).get_fdata()
sub_brainmask = np.where((sub_brainmask == 0) | (sub_brainmask == 1), 1 - sub_brainmask, sub_brainmask)
sub_phys = sub.get_physio(session=ses_nr, run=2, task='RS')
full_physio = sub.get_physio(session=ses_nr, run=2, task='RS')
hr_regressors = full_physio[full_physio.columns[-5:-2]]
rvt_regressors = full_physio[full_physio.columns[-2:]]

# confound creation
retro_regressors = sub.get_retroicor_confounds(session=ses_nr, run=2, task='RS')
aroma_regressors = sub.get_aroma_confounds(session=ses_nr, run=2, task='RS')
acompcor_regressors = sub.get_acompcor_confounds(session=ses_nr, run=2, task='RS', number_regressors=5)

# Account for processing in MNI space (for MNI-mask and Brainstem Mask) as well as native space (LC mask and GM mask)
func_data_list = [func_data_mni, func_data_native]
for func_data_counter, func_data in enumerate(func_data_list):

    if func_data_counter == 0:
        mask = mni_mat
        space_identifier = 'MNI'

    elif func_data_counter == 1:
        mask = sub_brainmask
        space_identifier = 'native'


    # Full brain uncleaned TSNR map
    func_data_uncleaned_dummy = func_data
    tsnr_matrix_uncleaned = np.divide(np.mean(func_data_uncleaned_dummy.get_fdata(),
                                              axis=3),
                                      np.std(func_data_uncleaned_dummy.get_fdata(),
                                             axis=3))
    tsnr_matrix_noinf_uncleaned = np.nan_to_num(tsnr_matrix_uncleaned,
                                                neginf=0,
                                                posinf=0)
    del func_data_uncleaned_dummy
    masked_tsnr_uncleaned = ma.array(tsnr_matrix_noinf_uncleaned,
                                     mask=mask).filled(0)
    masked_tsnr_uncleaned[masked_tsnr_uncleaned > 500],\
        masked_tsnr_uncleaned[masked_tsnr_uncleaned < -100] = 500, -100
    nib.save(nib.Nifti2Image(masked_tsnr_uncleaned,
                             affine=func_data.affine,
                             header=func_data.header),
             BASEPATH + '{0}/glms/tsnr_noclean_{1}.nii.gz'.format(sub_id, space_identifier))

    tsnr_dic = {'tsnr_retro': [retro_regressors.copy(), None],
                'tsnr_aroma': [aroma_regressors.copy(), None],
                'tsnr_acompcor': [acompcor_regressors.copy(), None],
                'tsnr_hr': [hr_regressors.copy(), None],
                'tsnr_rvt': [rvt_regressors.copy(), None],
                'tsnr_hr_rvt': [pd.concat([hr_regressors, rvt_regressors], axis=1), None],
                'tsnr_aroma_acompcor': [pd.concat([aroma_regressors, acompcor_regressors], axis=1), None],
                'tsnr_aroma_retro': [pd.concat([retro_regressors, aroma_regressors], axis=1), None],
                'tsnr_retro_hr_rvt': [pd.concat([retro_regressors, hr_regressors, rvt_regressors], axis=1), None],
                'tsnr_aroma_retro_hr_rvt': [pd.concat([retro_regressors, hr_regressors, rvt_regressors,
                                                      aroma_regressors], axis=1), None],
                'tsnr_aroma_retro_acompcor': [pd.concat([retro_regressors, aroma_regressors, acompcor_regressors],
                                                       axis=1), None],
                'tsnr_aroma_retro_acompcor_hr_rvt': [pd.concat([retro_regressors, hr_regressors, rvt_regressors,
                                                               aroma_regressors, acompcor_regressors], axis=1),
                                                     None]}

    for tsnr_map in tsnr_dic.keys():
        dummy_reg = tsnr_dic[tsnr_map][0]
        fig = plot_design_matrix(dummy_reg,
                                 output_file=BASEPATH + f'{sub_id}/design/confounds_cleaning_{tsnr_map}.png')
        func_data_phys_cleaned = nilearn.image.clean_img(func_data,
                                                         standardize=False,
                                                         detrend=False,
                                                         confounds=dummy_reg,
                                                         t_r=2.02)
        tsnr_matrix = np.divide(np.mean(func_data_phys_cleaned.get_fdata(),
                                        axis=3),
                                np.std(func_data_phys_cleaned.get_fdata(),
                                        axis=3))

        masked_tsnr = ma.array(np.nan_to_num(tsnr_matrix,
                                             neginf=0,
                                             posinf=0),
                               mask=mask).filled(0)

        masked_tsnr[masked_tsnr > 500], masked_tsnr[masked_tsnr < -100] = 500, -100
        nib.save(nib.Nifti2Image(masked_tsnr,
                                 affine=func_data.affine,
                                 header=func_data.header),
                 BASEPATH + f'{sub_id}/glms/{tsnr_map}_{space_identifier}.nii.gz')
        tsnr_dic[tsnr_map][1] = masked_tsnr

    contrast_dic = {'difference_aroma_to_uncleaned': tsnr_dic['tsnr_aroma'][1] - masked_tsnr_uncleaned,
                    'difference_retro_to_uncleaned': tsnr_dic['tsnr_retro'][1] - masked_tsnr_uncleaned,
                    'difference_acompcor_to_uncleaned': tsnr_dic['tsnr_acompcor'][1] - masked_tsnr_uncleaned,
                    'difference_aroma_acompcor_to_uncleaned':
                        tsnr_dic['tsnr_aroma_acompcor'][1] - masked_tsnr_uncleaned,
                    'difference_aroma_retro_to_uncleaned': tsnr_dic['tsnr_aroma_retro'][1] - masked_tsnr_uncleaned,
                    'difference_unique_aroma_to_retro': tsnr_dic['tsnr_aroma_retro'][1] - tsnr_dic['tsnr_retro'][1],
                    'difference_unique_retro_to_aroma': tsnr_dic['tsnr_aroma_retro'][1] - tsnr_dic['tsnr_aroma'][1],
                    'difference_unique_acompcor_to_aroma': tsnr_dic['tsnr_aroma_acompcor'][1]
                                                           - tsnr_dic['tsnr_aroma'][1],
                    'difference_unique_retro_to_aroma_acompcor':
                        tsnr_dic['tsnr_aroma_retro_acompcor'][1] - tsnr_dic['tsnr_aroma_acompcor'][1],
                    'difference_percent_unique_retro_to_aroma':
                        ((tsnr_dic['tsnr_aroma_retro'][1] / tsnr_dic['tsnr_aroma'][1]) - 1) * 100,
                    'difference_percent_unique_aroma_to_retro':
                        ((tsnr_dic['tsnr_aroma_retro'][1] / tsnr_dic['tsnr_retro'][1]) - 1) * 100,
                    'difference_percent_unique_retro_to_aroma_acompcor':
                        ((tsnr_dic['tsnr_aroma_retro_acompcor'][1] / tsnr_dic['tsnr_aroma_acompcor'][1]) - 1) * 100,
                    'difference_percent_unique_acompcor_to_aroma':
                        ((tsnr_dic['tsnr_aroma_acompcor'][1] / tsnr_dic['tsnr_aroma'][1]) - 1) * 100,
                    'difference_percent_unique_retro_to_aroma_vs_uncleaned':
                        ((((tsnr_dic['tsnr_aroma_retro'][1] /masked_tsnr_uncleaned) - 1) * 100) -
                         (((tsnr_dic['tsnr_aroma'][1] / masked_tsnr_uncleaned) - 1) * 100)),
                    'difference_percent_unique_aroma_to_retro_vs_uncleaned':
                        ((((tsnr_dic['tsnr_aroma_retro'][1] / masked_tsnr_uncleaned) - 1) * 100) -
                         (((tsnr_dic['tsnr_retro'][1] / masked_tsnr_uncleaned) - 1) * 100)),
                    'difference_percent_unique_acompcor_to_aroma_vs_uncleaned':
                        ((((tsnr_dic['tsnr_aroma_acompcor'][1] / masked_tsnr_uncleaned) - 1) * 100) - (
                                ((tsnr_dic['tsnr_aroma'][1] / masked_tsnr_uncleaned) - 1) * 100)),
                    'difference_percent_unique_retro_to_aroma_acompcor_vs_uncleaned':
                        ((((tsnr_dic['tsnr_aroma_retro_acompcor'][1] / masked_tsnr_uncleaned) - 1) * 100) - (
                                (((tsnr_dic['tsnr_aroma_acompcor'][1]) / masked_tsnr_uncleaned) - 1) * 100)),
                    'difference_percent_retro_to_uncleaned':
                        ((tsnr_dic['tsnr_retro'][1] / masked_tsnr_uncleaned) - 1) * 100,
                    'difference_percent_aroma_to_uncleaned':
                        ((tsnr_dic['tsnr_aroma'][1] / masked_tsnr_uncleaned) - 1) * 100,
                    'difference_percent_acompcor_to_uncleaned':
                        ((tsnr_dic['tsnr_acompcor'][1] / masked_tsnr_uncleaned) - 1) * 100,
                    'difference_percent_hr_to_uncleaned': ((tsnr_dic['tsnr_hr'][1] / masked_tsnr_uncleaned) - 1) * 100,
                    'difference_percent_rvt_to_uncleaned': ((tsnr_dic['tsnr_rvt'][
                                                                1] / masked_tsnr_uncleaned) - 1) * 100,
                    'difference_percent_hr_rvt_to_uncleaned':
                        ((tsnr_dic['tsnr_hr_rvt'][1] / masked_tsnr_uncleaned) - 1) * 100,
                    'difference_hr_rvt_to_uncleaned': tsnr_dic['tsnr_hr_rvt'][1] - masked_tsnr_uncleaned,
                    'difference_retro_hr_rvt_to_uncleaned': tsnr_dic['tsnr_retro_hr_rvt'][1] - masked_tsnr_uncleaned,
                    'difference_unique_retro_hr_rvt_to_aroma': tsnr_dic['tsnr_aroma_retro_hr_rvt'][1]
                                                               - tsnr_dic['tsnr_aroma'][1],
                    'difference_percent_unique_retro_hr_rvt_to_aroma':
                        ((tsnr_dic['tsnr_aroma_retro_hr_rvt'][1] / tsnr_dic['tsnr_aroma'][1]) - 1) * 100,
                    'difference_percent_unique_retro_hr_rvt_to_aroma_acompcor':
                        ((tsnr_dic['tsnr_aroma_retro_acompcor_hr_rvt'][1] / tsnr_dic['tsnr_aroma_acompcor'][1]) - 1)
                        * 100,
                    'difference_percent_unique_retro_hr_rvt_to_aroma_vs_uncleaned':
                        ((((tsnr_dic['tsnr_aroma_retro_hr_rvt'][1] / masked_tsnr_uncleaned) - 1) * 100) -
                         (((tsnr_dic['tsnr_aroma'][1] / masked_tsnr_uncleaned) - 1) * 100)),
                    'difference_percent_unique_retro_hr_rvt_to_aroma_acompcor_vs_uncleaned':
                        ((((tsnr_dic['tsnr_aroma_retro_acompcor_hr_rvt'][1] / masked_tsnr_uncleaned) - 1) * 100) - (
                                (((tsnr_dic['tsnr_aroma_acompcor'][1]) / masked_tsnr_uncleaned) - 1) * 100)),
                    'difference_percent_retro_hr_rvt_to_uncleaned': ((tsnr_dic['tsnr_retro_hr_rvt'][1]
                                                                      / masked_tsnr_uncleaned) - 1) * 100,
                    'difference_percent_aroma_acompcor_to_uncleaned': ((tsnr_dic['tsnr_aroma_acompcor'][1]
                                                                        / masked_tsnr_uncleaned) - 1) * 100}

    for contrast in contrast_dic.keys():
        nib.save(nib.Nifti2Image(contrast_dic[contrast],
                                 affine=func_data.affine,
                                 header=func_data.header),
                 BASEPATH + f'{sub_id}/glms/tsnr_{contrast}_{space_identifier}.nii.gz')

"""
    mask_list = [i for i in contrast_dic.keys()]

    #Create Average TSNR images in MNI space for all comparisons
    if sub_id == part_list[0][-7:] and func_data_counter == 0:
        mask_list_MNI = [i + '_MNI' for i in contrast_dic.keys()]
        contrast_list = [i[:, :, :, np.newaxis] for i in contrast_dic.values()]
        mni_list = dict(zip(mask_list_MNI, contrast_list))

    elif sub_id != part_list[0][-7:] and func_data_counter == 0:
        for output_counter, output in enumerate(mni_list.keys()):
            mni_list[output] = np.concatenate((mni_list[output], contrast_dic[mask_list[output_counter]][:, :, :, np.newaxis]),
                                              axis=3)

for output_counter, output in enumerate(mni_list.keys()):
nib.save(nib.Nifti2Image(np.mean(mni_list[output],
                                 axis=3),
                         affine=func_data_mni.affine,
                         header=func_data_mni.header),
         SAVEPATH + f'Overall_tsnr_{output}.nii.gz')
"""