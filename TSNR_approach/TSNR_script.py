#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 17:55:52 2021

This script implements the calculation of TSNR for different noise-cleaning procedures.

Creating different TSNR map for:
    Uncleaned data
    aggrAROMA cleaned data
    RETROICOR cleaned data
    aggrAROMA AND RETROICOR cleaned data

Additionally maps are created visualising the unique contributions of a method OVER another.
    Unique TSNR improvement of aggrAROMA (TSNR of aggrAROMA+RETRO - TSNR of RETRO)
    Unique TSNR improvement of RETROICOR (TSNR of aggrAROMA+RETRO - TSNR of aggrAROMA)
    Difference in TSNR improvement between RETROICOR and aggrAROMA (TSNR of aggrAROMA - TSNR of RETRO)
    TSNR improvement of uncleaned data for aggrAROMA (TSNR of aggrAROMA - TSNR of uncleaned data)
    TSNR improvement of uncleaned data for RETROICOR (TSNR of RETROICOR - TSNR of uncleaned data)

@author: MKrentz
"""

import numpy as np
import nibabel as nib
import pandas as pd
import glob
from Subject_Class import Subject
import nilearn
from nilearn import image
import numpy.ma as ma
from nilearn.datasets import load_mni152_brain_mask


MNI_MASK = load_mni152_brain_mask()
BASEPATH = '/project/3013068.03/RETROICOR/TSNR/'

# Load MNI mask to used masked data matrices and switch 0 to 1 and 1 to 0
mni_mat = MNI_MASK.get_fdata()
mni_mat = np.where((mni_mat == 0)|(mni_mat == 1), 1-mni_mat, mni_mat)

# Load all available participants
part_list = glob.glob(BASEPATH + 'sub-*')
part_list.sort()

# Indicating subject having the 'stress' condition during their FIRST functional session
stress_list = ['sub-002', 'sub-003', 'sub-004', 'sub-007', 'sub-009', 'sub-013', 'sub-015', 'sub-017', 'sub-021', 'sub-023', 'sub-025', 'sub-027', 'sub-029']

# Chosen RETROICOR components
RETRO_model = ['cardiac_phase_sin_1', 'cardiac_phase_cos_1', 'cardiac_phase_sin_2', 'cardiac_phase_cos_2', 'cardiac_phase_sin_3', 'cardiac_phase_cos_3', \
                                  'respiratory_phase_sin_1', 'respiratory_phase_cos_1', 'respiratory_phase_sin_2', 'respiratory_phase_cos_2', 'respiratory_phase_sin_3', 'respiratory_phase_cos_3', \
                                  'respiratory_phase_sin_4', 'respiratory_phase_cos_4']

# Used for variable naming
def object_name(obj, name_pos):
    return [name for name in name_pos if name_pos[name] is obj]

# Subject loop
for subject_long in part_list:    
    # Subject space_identifier
    sub_id = subject_long[-7:]
    sub_obj = Subject(sub_id)
    
    # Account for balancing in stress/control session order
    ses_nr = 2 if sub_id in stress_list else 1
    
    # Loading respective functional data into memory and online-smooth with 6mm FWHM 
    func_data_mni = sub_obj.get_func_data(session=ses_nr,run=2,task='RS', MNI=True)
    func_data_mni = image.smooth_img(func_data_mni, fwhm=6)
    func_data_native = sub_obj.get_func_data(session=ses_nr,run=2,task='RS', MNI=False)
    func_data_native = image.smooth_img(func_data_native, fwhm=6)
    
    # Load fmriprep confound files for respective runs, T1-brainmask and RETROICOR regressors
    sub_confounds = sub_obj.get_confounds(session=ses_nr, run=2, task='RS')
    sub_brainmask = sub_obj.get_brainmask(session=ses_nr, run=2, MNI=False).get_fdata()
    sub_brainmask = np.where((sub_brainmask == 0) | (sub_brainmask == 1), 1 - sub_brainmask, sub_brainmask)
    sub_phys = sub_obj.get_physio(session=ses_nr, run=2, task='RS')
    
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
        func_data_matrix = func_data.get_fdata()
        tsnr_matrix_uncleaned = np.divide(np.mean(func_data_matrix,axis = 3), np.std(func_data_matrix,axis = 3))
        tsnr_matrix_noinf_uncleaned = np.nan_to_num(tsnr_matrix_uncleaned, neginf = 0, posinf = 0)
        masked_tsnr_uncleaned = ma.array(tsnr_matrix_noinf_uncleaned, mask = mask).filled(0)
        masked_tsnr_uncleaned[masked_tsnr_uncleaned > 500], masked_tsnr_uncleaned[masked_tsnr_uncleaned < -100] = 500, -100
        nib.save(nib.Nifti2Image(masked_tsnr_uncleaned, affine = func_data.affine, header = func_data.header), BASEPATH + '{0}/tsnr_noclean_{1}.nii.gz'.format(sub_id, space_identifier))

        # RETROICOR TSNR map
        sub_phys_3C4R = sub_phys[RETRO_model]
        
        # Expansion of RETROICOR model with multiplication terms
        multi_01 = sub_phys_3C4R['cardiac_phase_sin_1'] * sub_phys_3C4R['respiratory_phase_sin_1']
        multi_02 = sub_phys_3C4R['cardiac_phase_sin_1'] * sub_phys_3C4R['respiratory_phase_cos_1']
        multi_03 = sub_phys_3C4R['cardiac_phase_cos_1'] * sub_phys_3C4R['respiratory_phase_sin_1']
        multi_04 = sub_phys_3C4R['cardiac_phase_cos_1'] * sub_phys_3C4R['respiratory_phase_sin_1']

        sub_phys_3C4R1M = pd.concat([sub_phys_3C4R, multi_01, multi_02, multi_03, multi_04],axis=1)
        func_data_phys_cleaned = nilearn.image.clean_img(func_data, standardize = False, detrend = False, confounds = sub_phys_3C4R1M, t_r = 2.02)
        tsnr_matrix_RETRO = np.divide(np.mean(func_data_phys_cleaned.get_fdata(), axis = 3), np.std(func_data_phys_cleaned.get_fdata(), axis = 3))
        del func_data_phys_cleaned
        masked_tsnr_RETRO = ma.array(np.nan_to_num(tsnr_matrix_RETRO, neginf = 0, posinf = 0), mask = mask).filled(0)
        masked_tsnr_RETRO[masked_tsnr_RETRO > 500], masked_tsnr_RETRO[masked_tsnr_RETRO < -100] = 500, -100
        nib.save(nib.Nifti2Image(masked_tsnr_RETRO, affine = func_data.affine, header = func_data.header), BASEPATH + '{0}/tsnr_RETRO_{1}.nii.gz'.format(sub_id, space_identifier))

        # AROMA TSNR map
        
        #Grab regressors from confound file
        confounds_column_index = sub_confounds.columns.tolist()
        aroma_sum = sum((itm.count("aroma_motion") for itm in confounds_column_index))
        aroma_variables = confounds_column_index[-aroma_sum:]
        
        func_data_aroma_cleaned = nilearn.image.clean_img(func_data, standardize = False, detrend = False, confounds = sub_confounds[aroma_variables], t_r = 2.02)
        tsnr_matrix_aggrAROMA = np.divide(np.mean(func_data_aroma_cleaned.get_fdata(), axis = 3), np.std(func_data_aroma_cleaned.get_fdata(),axis = 3))
        del func_data_aroma_cleaned
        masked_tsnr_aggrAROMA = ma.array(np.nan_to_num(tsnr_matrix_aggrAROMA, neginf = 0, posinf = 0), mask = mask).filled(0)
        masked_tsnr_aggrAROMA[masked_tsnr_aggrAROMA>500], masked_tsnr_aggrAROMA[masked_tsnr_aggrAROMA<-100] = 500, -100
        nib.save(nib.Nifti2Image(masked_tsnr_aggrAROMA, affine = func_data.affine, header = func_data.header), BASEPATH + '{0}/tsnr_aggrAROMA_{1}.nii.gz'.format(sub_id,space_identifier))

        # Combined AROMA and RETROICOR TSNR map

        combined_aroma_retro = pd.concat([sub_phys_3C4R1M, sub_confounds[aroma_variables]], axis = 1)
        func_data_aroma_retro_cleaned = nilearn.image.clean_img(func_data, standardize = False, detrend = False, confounds = combined_aroma_retro, t_r = 2.02)
        tsnr_matrix_aggrAROMA_RETRO = np.divide(np.mean(func_data_aroma_retro_cleaned.get_fdata(), axis = 3), np.std(func_data_aroma_retro_cleaned.get_fdata(),axis = 3))
        del func_data_aroma_retro_cleaned
        masked_tsnr_aggrAROMA_RETRO = ma.array(np.nan_to_num(tsnr_matrix_aggrAROMA_RETRO, neginf = 0, posinf = 0), mask = mask).filled(0)
        masked_tsnr_aggrAROMA_RETRO[masked_tsnr_aggrAROMA_RETRO > 500], masked_tsnr_aggrAROMA_RETRO[masked_tsnr_aggrAROMA_RETRO < -100]  = 500, -100
        nib.save(nib.Nifti2Image(masked_tsnr_aggrAROMA_RETRO, affine = func_data.affine, header = func_data.header), BASEPATH + '{0}/tsnr_aggrAROMA_RETRO_{1}.nii.gz'.format(sub_id, space_identifier))

        # Difference
        unique_tsnr_aggrAROMA = masked_tsnr_aggrAROMA_RETRO - masked_tsnr_RETRO
        nib.save(nib.Nifti2Image(unique_tsnr_aggrAROMA, affine = func_data.affine, header = func_data.header), BASEPATH + '{0}/tsnr_difference_aggrAROMARETRO_RETRO_{1}.nii.gz'.format(sub_id, space_identifier))

        # Difference AROMA
        unique_tsnr_RETRO = masked_tsnr_aggrAROMA_RETRO - masked_tsnr_aggrAROMA
        nib.save(nib.Nifti2Image(unique_tsnr_RETRO, affine = func_data.affine, header = func_data.header), BASEPATH + '{0}/tsnr_difference_aggrAROMARETRO_aggrAROMA_{1}.nii.gz'.format(sub_id, space_identifier))

        # Difference aggreAroma to uncleaned
        difference_aggrAROMA_uncleaned =  masked_tsnr_aggrAROMA - masked_tsnr_uncleaned
        nib.save(nib.Nifti1Image(difference_aggrAROMA_uncleaned, affine = func_data.affine, header = func_data.header), BASEPATH + '{0}/tsnr_difference_aggrAROMA_uncleaned_{1}.nii.gz'.format(sub_id, space_identifier))

        # Difference uncleaned vs aggrAROMA
        difference_uncleaned_aggrAROMA =  masked_tsnr_uncleaned - masked_tsnr_aggrAROMA
        nib.save(nib.Nifti2Image(difference_uncleaned_aggrAROMA, affine = func_data.affine, header = func_data.header), BASEPATH + '{0}/tsnr_difference_uncleaned_aggrAROMA_{1}.nii.gz'.format(sub_id, space_identifier))

        # Difference Combination to uncleaned
        difference_aggrAROMARETRO_uncleaned = masked_tsnr_aggrAROMA_RETRO - masked_tsnr_uncleaned
        nib.save(nib.Nifti2Image(difference_aggrAROMARETRO_uncleaned, affine = func_data.affine, header = func_data.header), BASEPATH + '{0}/tsnr_difference_aggrAROMARETRO_uncleaned_{1}.nii.gz'.format(sub_id, space_identifier))

        # Difference RETRO to uncleaned
        difference_RETRO_uncleaned =  masked_tsnr_RETRO - masked_tsnr_uncleaned
        nib.save(nib.Nifti2Image(difference_RETRO_uncleaned, affine = func_data.affine, header = func_data.header), BASEPATH + '{0}/tsnr_difference_RETRO_uncleaned_{1}.nii.gz'.format(sub_id, space_identifier))

        # Difference RETRO to uncleaned
        difference_RETRO_aggrAROMA =  masked_tsnr_RETRO - masked_tsnr_aggrAROMA
        nib.save(nib.Nifti2Image(difference_RETRO_aggrAROMA, affine = func_data.affine, header = func_data.header), BASEPATH + '{0}/tsnr_difference_RETRO_aggrAROMA_{1}.nii.gz'.format(sub_id, space_identifier))


        mask_list = [masked_tsnr_uncleaned, masked_tsnr_RETRO, masked_tsnr_aggrAROMA, masked_tsnr_aggrAROMA_RETRO,
                     unique_tsnr_aggrAROMA, unique_tsnr_RETRO, difference_aggrAROMA_uncleaned, difference_uncleaned_aggrAROMA,
                     difference_aggrAROMARETRO_uncleaned, difference_RETRO_uncleaned, difference_RETRO_aggrAROMA]


        #Create Average TSNR images in MNI space for all comparisons
        if sub_id == part_list[0][-7:] and func_data_counter == 0:
            tsnr_noclean_MNI = masked_tsnr_uncleaned[:,:,:,np.newaxis]
            tsnr_RETRO_MNI = masked_tsnr_RETRO[:,:,:,np.newaxis]
            tsnr_aggrAROMA_MNI = masked_tsnr_aggrAROMA[:,:,:,np.newaxis]
            tsnr_aggrAROMARETRO_MNI = masked_tsnr_aggrAROMA_RETRO[:,:,:,np.newaxis]
            tsnr_difference_aggrAROMARETRO_RETRO_MNI = unique_tsnr_aggrAROMA[:,:,:,np.newaxis]
            tsnr_difference_aggrAROMARETRO_aggrAROMA_MNI = unique_tsnr_RETRO[:,:,:,np.newaxis]
            tsnr_difference_aggrAROMA_uncleaned_MNI = difference_aggrAROMA_uncleaned[:,:,:,np.newaxis]
            tsnr_difference_uncleaned_aggrAROMA_MNI = difference_uncleaned_aggrAROMA[:,:,:,np.newaxis]
            tsnr_difference_aggrAROMARETRO_uncleaned_MNI = difference_aggrAROMARETRO_uncleaned[:,:,:,np.newaxis]
            tsnr_difference_RETRO_uncleaned_MNI = difference_RETRO_uncleaned[:,:,:,np.newaxis]
            tsnr_difference_RETRO_aggrAROMA_MNI = difference_RETRO_aggrAROMA[:,:,:,np.newaxis]

            output_list = [tsnr_noclean_MNI, tsnr_RETRO_MNI, tsnr_aggrAROMA_MNI, tsnr_aggrAROMARETRO_MNI,
                           tsnr_difference_aggrAROMARETRO_RETRO_MNI, tsnr_difference_aggrAROMARETRO_aggrAROMA_MNI,
                           tsnr_difference_aggrAROMA_uncleaned_MNI, tsnr_difference_uncleaned_aggrAROMA_MNI,
                           tsnr_difference_aggrAROMARETRO_uncleaned_MNI, tsnr_difference_RETRO_uncleaned_MNI,
                           tsnr_difference_RETRO_aggrAROMA_MNI]

        elif sub_id != part_list[0][-7:] and func_data_counter == 0:
            for output_counter, output in enumerate(output_list):
                output_list[output_counter] = np.concatenate((output, mask_list[output_counter][:,:,:,np.newaxis]), axis=3)

nib.save(nib.Nifti2Image(np.mean(output, axis = 3), affine = func_data.affine, header = func_data.header), BASEPATH + 'Overall_{0}).nii.gz'.format(object_name(output, globals())))



