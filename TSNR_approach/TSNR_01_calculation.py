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
import glob
from Subject_Class_new import Subject
import nilearn
from nilearn import image
import numpy.ma as ma
import random
from nilearn.plotting import plot_design_matrix
import matplotlib.pyplot as plt
import json
import gc


BASEPATH = '/project/3013068.03/physio_revision/TSNR_approach/'
SAVEPATH = '/project/3013068.03/physio_revision/TSNR_approach/mean_TSNR/'
# Load MNI mask to used masked data matrices and switch 0 to 1 and 1 to 0


# Load all available participants
part_list = glob.glob(BASEPATH + 'sub-*')
part_list.sort()
part_list = part_list[:-2]

# Indicating subject having the 'stress' condition during their FIRST functional session
stress_list = ['sub-002', 'sub-003', 'sub-004', 'sub-007', 'sub-009', 'sub-013', 'sub-015', 'sub-017',
               'sub-021', 'sub-023', 'sub-025', 'sub-027', 'sub-029']

# Fix the random seed for padding regressor (comment out when using an existing seed dictionary)

# Subject loop
for subject in part_list:

    # Subject space_identifier
    sub_id = subject[-7:]
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
    hr_rvt_regressors = full_physio[full_physio.columns[-5:]]
    
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
            del func_data_mni

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

        # retroICOR TSNR map
        retro_dummy = retro_regressors.copy()
        fig = plot_design_matrix(retro_dummy,
                                 output_file= BASEPATH + f'{sub_id}/design/confounds_cleaning_retroicor.png')
        plt.savefig(BASEPATH + '{0}/design/confounds_cleaning_retroicor.png'.format(sub_id))
        plt.close()
        func_data_phys_cleaned = nilearn.image.clean_img(func_data,
                                                         standardize=False,
                                                         detrend=False,
                                                         confounds=retro_dummy,
                                                         t_r=2.02)

        tsnr_matrix_retro = np.divide(np.mean(func_data_phys_cleaned.get_fdata(),
                                              axis=3),
                                      np.std(func_data_phys_cleaned.get_fdata(),
                                             axis=3))
        del func_data_phys_cleaned
        masked_tsnr_retro = ma.array(np.nan_to_num(tsnr_matrix_retro,
                                                   neginf=0,
                                                   posinf=0),
                                     mask=mask).filled(0)
        masked_tsnr_retro[masked_tsnr_retro > 500], masked_tsnr_retro[masked_tsnr_retro < -100] = 500, -100
        nib.save(nib.Nifti2Image(masked_tsnr_retro,
                                 affine=func_data.affine,
                                 header=func_data.header),
                 BASEPATH + '{0}/glms/tsnr_retro_{1}.nii.gz'.format(sub_id, space_identifier))
        
        gc.collect()
        # AROMA TSNR map
        aroma_dummy = aroma_regressors.copy()
        fig = plot_design_matrix(aroma_dummy,
                                 output_file=BASEPATH + f'{sub_id}/design/confounds_cleaning_AROMA.png')

        func_data_aroma_cleaned = nilearn.image.clean_img(func_data,
                                                          standardize=False,
                                                          detrend=False,
                                                          confounds=aroma_dummy,
                                                          t_r=2.02)
        tsnr_matrix_aroma = np.divide(np.mean(func_data_aroma_cleaned.get_fdata(),
                                              axis=3),
                                      np.std(func_data_aroma_cleaned.get_fdata(),
                                             axis=3))
        del func_data_aroma_cleaned
        masked_tsnr_aroma = ma.array(np.nan_to_num(tsnr_matrix_aroma,
                                                   neginf=0,
                                                   posinf=0),
                                     mask=mask).filled(0)
        masked_tsnr_aroma[masked_tsnr_aroma>500], masked_tsnr_aroma[masked_tsnr_aroma<-100] = 500, -100
        nib.save(nib.Nifti2Image(masked_tsnr_aroma,
                                 affine=func_data.affine,
                                 header=func_data.header),
                 BASEPATH + '{0}/glms/tsnr_aroma_{1}.nii.gz'.format(sub_id, space_identifier))

        gc.collect()
        
        # aCompCor TSNR map
        acompcor_dummy = acompcor_regressors.copy()
        fig = plot_design_matrix(aroma_dummy,
                                 output_file=BASEPATH + f'{sub_id}/design/confounds_cleaning_aCompCor.png')

        func_data_acompcor_cleaned = nilearn.image.clean_img(func_data,
                                                             standardize=False,
                                                             detrend=False,
                                                             confounds=acompcor_dummy,
                                                             t_r=2.02)
        tsnr_matrix_acompcor = np.divide(np.mean(func_data_acompcor_cleaned.get_fdata(),
                                                 axis=3),
                                         np.std(func_data_acompcor_cleaned.get_fdata(),
                                                axis=3))
        del func_data_acompcor_cleaned
        masked_tsnr_acompcor = ma.array(np.nan_to_num(tsnr_matrix_acompcor,
                                                      neginf=0,
                                                      posinf=0),
                                        mask=mask).filled(0)
        masked_tsnr_acompcor[masked_tsnr_acompcor>500], masked_tsnr_acompcor[masked_tsnr_acompcor<-100] = 500, -100
        nib.save(nib.Nifti2Image(masked_tsnr_acompcor,
                                 affine=func_data.affine,
                                 header=func_data.header),
                 BASEPATH + '{0}/glms/tsnr_acompcor_{1}.nii.gz'.format(sub_id,space_identifier))

        gc.collect()
        
        #RVT_HR ADDITION
        hr_rvt_regressors_dummy = hr_rvt_regressors.copy()
        fig = plot_design_matrix(hr_rvt_regressors_dummy,
                                 output_file=BASEPATH + f'{sub_id}/design/confounds_cleaning_RVTHR.png')

        rvt_hr_cleaned = nilearn.image.clean_img(func_data,
                                                 standardize=False,
                                                 detrend=False,
                                                 confounds=hr_rvt_regressors_dummy,
                                                 t_r=2.02)
        tsnr_matrix_rvt_hr = np.divide(np.mean(rvt_hr_cleaned.get_fdata(),
                                               axis=3),
                                       np.std(rvt_hr_cleaned.get_fdata(),
                                              axis=3))

        masked_tsnr_hr_rvt = ma.array(np.nan_to_num(tsnr_matrix_acompcor,
                                                    neginf=0,
                                                    posinf=0),
                                      mask=mask).filled(0)
        masked_tsnr_hr_rvt[masked_tsnr_hr_rvt > 500], masked_tsnr_hr_rvt[masked_tsnr_hr_rvt < -100] = 500, -100
        nib.save(nib.Nifti2Image(masked_tsnr_hr_rvt,
                                 affine=func_data.affine,
                                 header=func_data.header),
                 BASEPATH + '{0}/glms/tsnr_rvt_hr_{1}.nii.gz'.format(sub_id, space_identifier))

        gc.collect()

        #Combined AROMA and aCompCor TSNR map
        combined_aroma_acompcor = pd.concat([aroma_regressors, acompcor_regressors],
                                            axis=1)
        fig = plot_design_matrix(combined_aroma_acompcor,
                                 output_file=BASEPATH + f'{sub_id}/design/confounds_cleaning_AROMA+aCompCor.png')

        func_data_aroma_acompcor_cleaned = nilearn.image.clean_img(func_data,
                                                                   standardize=False,
                                                                   detrend=False,
                                                                   confounds=combined_aroma_acompcor,
                                                                   t_r=2.02)
        tsnr_matrix_aroma_acompcor = np.divide(np.mean(func_data_aroma_acompcor_cleaned.get_fdata(),
                                                       axis=3),
                                               np.std(func_data_aroma_acompcor_cleaned.get_fdata(),
                                                      axis=3))
        del func_data_aroma_acompcor_cleaned
        masked_tsnr_aroma_acompcor = ma.array(np.nan_to_num(tsnr_matrix_aroma_acompcor,
                                                            neginf=0,
                                                            posinf=0),
                                              mask=mask).filled(0)
        masked_tsnr_aroma_acompcor[masked_tsnr_aroma_acompcor > 500],\
            masked_tsnr_aroma_acompcor[masked_tsnr_aroma_acompcor < -100] = 500, -100
        nib.save(nib.Nifti2Image(masked_tsnr_aroma_acompcor,
                                 affine=func_data.affine,
                                 header=func_data.header),
                 BASEPATH + '{0}/glms/tsnr_aroma_acompcor_{1}.nii.gz'.format(sub_id, space_identifier))

        
        #Combined AROMA and RETROICOR TSNR map
        combined_aroma_retro = pd.concat([retro_regressors, aroma_regressors],
                                         axis=1)
        fig = plot_design_matrix(combined_aroma_retro,
                                 output_file=BASEPATH + f'{sub_id}/design/confounds_cleaning_AROMA+retro.png')

        func_data_aroma_retro_cleaned = nilearn.image.clean_img(func_data,
                                                                standardize=False,
                                                                detrend=False,
                                                                confounds=combined_aroma_retro,
                                                                t_r=2.02)
        tsnr_matrix_aroma_retro = np.divide(np.mean(func_data_aroma_retro_cleaned.get_fdata(),
                                                    axis=3),
                                            np.std(func_data_aroma_retro_cleaned.get_fdata(),
                                                   axis=3))
        del func_data_aroma_retro_cleaned
        masked_tsnr_aroma_retro = ma.array(np.nan_to_num(tsnr_matrix_aroma_retro,
                                                         neginf=0,
                                                         posinf=0),
                                           mask=mask).filled(0)
        masked_tsnr_aroma_retro[masked_tsnr_aroma_retro > 500], \
            masked_tsnr_aroma_retro[masked_tsnr_aroma_retro < -100] = 500, -100
        nib.save(nib.Nifti2Image(masked_tsnr_aroma_retro,
                                 affine=func_data.affine,
                                 header=func_data.header),
                 BASEPATH + '{0}/glms/tsnr_aroma_retro_{1}.nii.gz'.format(sub_id, space_identifier))
        
        # Combined RETRO and HR/RVT TSNR map
        combined_retro_hr_rvt = pd.concat([retro_regressors, hr_rvt_regressors],
                                         axis=1)
        fig = plot_design_matrix(combined_retro_hr_rvt,
                                 output_file=BASEPATH + f'{sub_id}/design/confounds_cleaning_retro+HRRVT.png')

        func_data_retro_hr_rvt_cleaned = nilearn.image.clean_img(func_data,
                                                                standardize=False,
                                                                detrend=False,
                                                                confounds=combined_retro_hr_rvt,
                                                                t_r=2.02)
        tsnr_matrix_retro_hr_rvt = np.divide(np.mean(func_data_retro_hr_rvt_cleaned.get_fdata(),
                                                    axis=3),
                                            np.std(func_data_retro_hr_rvt_cleaned.get_fdata(),
                                                   axis=3))
        del func_data_retro_hr_rvt_cleaned
        masked_tsnr_retro_hr_rvt = ma.array(np.nan_to_num(tsnr_matrix_retro_hr_rvt,
                                                         neginf=0,
                                                         posinf=0),
                                           mask=mask).filled(0)
        masked_tsnr_retro_hr_rvt[masked_tsnr_retro_hr_rvt > 500], \
            masked_tsnr_retro_hr_rvt[masked_tsnr_retro_hr_rvt < -100] = 500, -100
        nib.save(nib.Nifti2Image(masked_tsnr_retro_hr_rvt,
                                 affine=func_data.affine,
                                 header=func_data.header),
                 BASEPATH + '{0}/glms/tsnr_retro_hr_rvt_{1}.nii.gz'.format(sub_id, space_identifier))
        
        #Combined AROMA, RETRO, RVT/HR TSNR map
        combined_aroma_retro_hr_rvt = pd.concat([retro_regressors, hr_rvt_regressors, aroma_regressors],
                                         axis=1)
        fig = plot_design_matrix(combined_aroma_retro_hr_rvt,
                                 output_file=BASEPATH + f'{sub_id}/design/confounds_cleaning_AROMA+retro+HRRVT.png')

        func_data_aroma_retro_hr_rvt_cleaned = nilearn.image.clean_img(func_data,
                                                                standardize=False,
                                                                detrend=False,
                                                                confounds= combined_aroma_retro_hr_rvt,
                                                                t_r=2.02)
        tsnr_matrix_aroma_retro_hr_rvt = np.divide(np.mean(func_data_aroma_retro_hr_rvt_cleaned.get_fdata(),
                                                    axis=3),
                                            np.std(func_data_aroma_retro_hr_rvt_cleaned.get_fdata(),
                                                   axis=3))
        del func_data_aroma_retro_hr_rvt_cleaned
        masked_tsnr_aroma_retro_hr_rvt = ma.array(np.nan_to_num(tsnr_matrix_aroma_retro_hr_rvt,
                                                         neginf=0,
                                                         posinf=0),
                                           mask=mask).filled(0)
        masked_tsnr_aroma_retro_hr_rvt[masked_tsnr_aroma_retro_hr_rvt > 500], \
            masked_tsnr_aroma_retro_hr_rvt[masked_tsnr_aroma_retro_hr_rvt < -100] = 500, -100
        nib.save(nib.Nifti2Image(masked_tsnr_aroma_retro_hr_rvt,
                                 affine=func_data.affine,
                                 header=func_data.header),
                 BASEPATH + '{0}/glms/tsnr_aroma_retro_hr_rvt_{1}.nii.gz'.format(sub_id, space_identifier))
        
        # Combined AROMA, retroICOR and aCompCor TSNR map
        combined_regressors = pd.concat([retro_regressors, aroma_regressors, acompcor_regressors],
                                        axis=1)
        fig = plot_design_matrix(combined_regressors,
                                 output_file=BASEPATH + f'{sub_id}/design/confounds_cleaning_AROMA+retro+aCompCor.png')
    
        func_data_aroma_retro_acompcor_cleaned = nilearn.image.clean_img(func_data,
                                                                         standardize=False,
                                                                         detrend=False,
                                                                         confounds=combined_regressors,
                                                                         t_r=2.02)
        
        tsnr_matrix_aroma_retro_acompcor = np.divide(np.mean(func_data_aroma_retro_acompcor_cleaned.get_fdata(),
                                                             axis=3),
                                                     np.std(func_data_aroma_retro_acompcor_cleaned.get_fdata(),
                                                            axis=3))
        del func_data_aroma_retro_acompcor_cleaned
        masked_tsnr_aroma_retro_acompcor = ma.array(np.nan_to_num(tsnr_matrix_aroma_retro_acompcor,
                                                                  neginf=0,
                                                                  posinf=0),
                                                    mask=mask).filled(0)
        masked_tsnr_aroma_retro_acompcor[masked_tsnr_aroma_retro_acompcor > 500],\
            masked_tsnr_aroma_retro_acompcor[masked_tsnr_aroma_retro_acompcor < -100] = 500, -100
        nib.save(nib.Nifti2Image(masked_tsnr_aroma_retro_acompcor,
                                 affine=func_data.affine,
                                 header=func_data.header),
                 BASEPATH + '{0}/glms/tsnr_aroma_retro_acompcor_{1}.nii.gz'.format(sub_id, space_identifier))
        
        # Combined AROMA, RETROICOR, HR/RVT, aCompCor
        combined_regressors = pd.concat([retro_regressors, hr_rvt_regressors, aroma_regressors, acompcor_regressors],
                                        axis=1)
        fig = plot_design_matrix(combined_regressors,
                                 output_file=BASEPATH + f'{sub_id}/design/confounds_cleaning_AROMA+retro+aCompCor'
                                                        f'+HRRVT.png')

        func_data_aroma_retro_acompcor_hr_rvt_cleaned = nilearn.image.clean_img(func_data,
                                                                         standardize=False,
                                                                         detrend=False,
                                                                         confounds=combined_regressors,
                                                                         t_r=2.02)

        tsnr_matrix_aroma_retro_acompcor_hr_rvt = np.divide(np.mean(
            func_data_aroma_retro_acompcor_hr_rvt_cleaned.get_fdata(),
                                                             axis=3),
                                                     np.std(func_data_aroma_retro_acompcor_hr_rvt_cleaned.get_fdata(),
                                                            axis=3))
        del func_data_aroma_retro_acompcor_hr_rvt_cleaned
        masked_tsnr_aroma_retro_acompcor_hr_rvt = ma.array(np.nan_to_num(tsnr_matrix_aroma_retro_acompcor_hr_rvt,
                                                                  neginf=0,
                                                                  posinf=0),
                                                    mask=mask).filled(0)
        masked_tsnr_aroma_retro_acompcor_hr_rvt[masked_tsnr_aroma_retro_acompcor_hr_rvt > 500], \
            masked_tsnr_aroma_retro_acompcor_hr_rvt[masked_tsnr_aroma_retro_acompcor_hr_rvt < -100] = 500, -100
        nib.save(nib.Nifti2Image(masked_tsnr_aroma_retro_acompcor_hr_rvt,
                                 affine=func_data.affine,
                                 header=func_data.header),
                 BASEPATH + '{0}/glms/tsnr_aroma_retro_acompcor_hr_rvt_{1}.nii.gz'.format(sub_id, space_identifier))
        
        contrast_dic = {'difference_aroma_to_uncleaned': masked_tsnr_aroma - masked_tsnr_uncleaned,
                        'difference_retro_to_uncleaned': masked_tsnr_retro - masked_tsnr_uncleaned,
                        'difference_acompcor_to_uncleaned': masked_tsnr_acompcor - masked_tsnr_uncleaned,
                        'difference_aroma_acompcor_to_uncleaned':
                            masked_tsnr_aroma_acompcor - masked_tsnr_uncleaned,
                        'difference_aroma_retro_to_uncleaned': masked_tsnr_aroma_retro - masked_tsnr_uncleaned,
                        'difference_unique_aroma_to_retro': masked_tsnr_aroma_retro - masked_tsnr_retro,
                        'difference_unique_retro_to_aroma': masked_tsnr_aroma_retro - masked_tsnr_aroma,
                        'difference_unique_acompcor_to_aroma': masked_tsnr_aroma_acompcor - masked_tsnr_aroma,
                        'difference_unique_retro_to_aroma_acompcor':
                            masked_tsnr_aroma_retro_acompcor - masked_tsnr_aroma_acompcor,
                        'difference_percent_unique_retro_to_aroma':
                            ((masked_tsnr_aroma_retro / masked_tsnr_aroma) - 1) * 100,
                        'difference_percent_unique_aroma_to_retro':
                            ((masked_tsnr_aroma_retro / masked_tsnr_retro) - 1) * 100,
                        'difference_percent_unique_retro_to_aroma_acompcor':
                            ((masked_tsnr_aroma_retro_acompcor / masked_tsnr_aroma_acompcor) - 1) * 100,
                        'difference_percent_unique_acompcor_to_aroma':
                            ((masked_tsnr_aroma_acompcor / masked_tsnr_aroma) - 1) * 100,
                        'difference_percent_unique_retro_to_aroma_vs_uncleaned':
                            ((((masked_tsnr_aroma_retro /masked_tsnr_uncleaned) - 1) * 100) -
                             (((masked_tsnr_aroma / masked_tsnr_uncleaned) - 1) * 100)),
                        'difference_percent_unique_aroma_to_retro_vs_uncleaned':
                            ((((masked_tsnr_aroma_retro / masked_tsnr_uncleaned) - 1) * 100) -
                             (((masked_tsnr_retro / masked_tsnr_uncleaned) - 1) * 100)),
                        'difference_percent_unique_acompcor_to_aroma_vs_uncleaned':
                            ((((masked_tsnr_aroma_acompcor / masked_tsnr_uncleaned) - 1) * 100) - (
                                    ((masked_tsnr_aroma / masked_tsnr_uncleaned) - 1) * 100)),
                        'difference_percent_unique_retro_to_aroma_acompcor_vs_uncleaned':
                            ((((masked_tsnr_aroma_retro_acompcor / masked_tsnr_uncleaned) - 1) * 100) - (
                                    (((masked_tsnr_aroma_acompcor) / masked_tsnr_uncleaned) - 1) * 100)),
                        'difference_percent_retro_to_uncleaned':
                            ((masked_tsnr_retro / masked_tsnr_uncleaned) - 1) * 100,
                        'difference_percent_aroma_to_uncleaned':
                            ((masked_tsnr_aroma / masked_tsnr_uncleaned) - 1) * 100,
                        'difference_percent_acompcor_to_uncleaned':
                            ((masked_tsnr_acompcor / masked_tsnr_uncleaned) - 1) * 100,
                        'difference_percent_hr_rvt_to_uncleaned':
                            ((masked_tsnr_hr_rvt / masked_tsnr_uncleaned) - 1) * 100,
                        'difference_hr_rvt_to_uncleaned': masked_tsnr_hr_rvt - masked_tsnr_uncleaned,
                        'difference_retro_hr_rvt_to_uncleaned': masked_tsnr_retro_hr_rvt - masked_tsnr_uncleaned,
                        'difference_unique_retro_hr_rvt_to_aroma': masked_tsnr_aroma_retro_hr_rvt - masked_tsnr_aroma,
                        'difference_percent_unique_retro_hr_rvt_to_aroma':
                            ((masked_tsnr_aroma_retro_hr_rvt / masked_tsnr_aroma) - 1) * 100,
                        'difference_percent_unique_retro_hr_rvt_to_aroma_acompcor':
                            ((masked_tsnr_aroma_retro_acompcor_hr_rvt / masked_tsnr_aroma_acompcor) - 1) * 100,
                        'difference_percent_unique_retro_hr_rvt_to_aroma_vs_uncleaned':
                            ((((masked_tsnr_aroma_retro_hr_rvt / masked_tsnr_uncleaned) - 1) * 100) -
                             (((masked_tsnr_aroma / masked_tsnr_uncleaned) - 1) * 100)),
                        'difference_percent_unique_retro_hr_rvt_to_aroma_acompcor_vs_uncleaned':
                            ((((masked_tsnr_aroma_retro_acompcor_hr_rvt / masked_tsnr_uncleaned) - 1) * 100) - (
                                    (((masked_tsnr_aroma_acompcor) / masked_tsnr_uncleaned) - 1) * 100)),
                        'difference_percent_retro_hr_rvt_to_uncleaned': ((masked_tsnr_retro_hr_rvt / masked_tsnr_uncleaned) - 1) * 100,
                        'difference_percent_aroma_acompcor_to_uncleaned': ((masked_tsnr_aroma_acompcor / masked_tsnr_uncleaned) - 1) * 100}

        for contrast in contrast_dic.keys():
            nib.save(nib.Nifti2Image(contrast_dic[contrast],
                                     affine=func_data.affine,
                                     header=func_data.header),
                     BASEPATH + f'{sub_id}/glms/tsnr_{contrast}_{space_identifier}.nii.gz')

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
