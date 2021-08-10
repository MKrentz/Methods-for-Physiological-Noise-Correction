#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 17:55:52 2021

tSNR calculation for RETROICOR comparisson first attempt

@author: markre
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

MNI_mask = load_mni152_brain_mask()

# Load MNI mask to used masked data matrices
mni_mat = MNI_mask.get_fdata()
mni_mat[mni_mat == 1] = 2
mni_mat[mni_mat == 0] = 1
mni_mat[mni_mat == 2] = 0

# Load all available participants
part_list = glob.glob('/project/3013068.03/RETROICOR/TSNR/sub-*')
part_list.sort()

# Indicating subject having the 'stress' condition during their FIRST functional session
stress_list = ['sub-002', 'sub-003', 'sub-004', 'sub-007', 'sub-009', 'sub-013', 'sub-015', 'sub-017', 'sub-021', 'sub-023', 'sub-025', 'sub-027', 'sub-029']

# Chosen RETROICOR components
RETRO_model = ['cardiac_phase_sin_1', 'cardiac_phase_cos_1', 'cardiac_phase_sin_2', 'cardiac_phase_cos_2', 'cardiac_phase_sin_3', 'cardiac_phase_cos_3', \
                                  'respiratory_phase_sin_1', 'respiratory_phase_cos_1', 'respiratory_phase_sin_2', 'respiratory_phase_cos_2', 'respiratory_phase_sin_3', 'respiratory_phase_cos_3', \
                                  'respiratory_phase_sin_4', 'respiratory_phase_cos_4']
#Path
basepath = '/project/3013068.03/RETROICOR/TSNR/'

#Used for variable naming
def object_name(obj, name_pos):
    return [name for name in name_pos if name_pos[name] is obj]

for subject_long in part_list:    

    # Subject identifier
    sub_id = subject_long[-7:]
    sub_obj = Subject(sub_id)
    
    # Account for balancing in stress/control session order
    ses_nr = 2 if sub_id in stress_list else 1
    
    # Loading respective functional data into memory
    func_data_mni = sub_obj.get_func_data(session=ses_nr,run=2,task='RS', MNI=True)
    func_data_mni = image.smooth_img(func_data_mni, fwhm=6)
    func_data_native = sub_obj.get_func_data(session=ses_nr,run=2,task='RS', MNI=False)
    func_data_native = image.smooth_img(func_data_native, fwhm=6)
    sub_confounds = sub_obj.get_confounds(session=ses_nr, run=2, task='RS')
    sub_brainmask = sub_obj.get_brainmask(session=ses_nr, run=2, MNI=False).get_fdata()
    sub_brainmask[sub_brainmask == 1] = 2
    sub_brainmask[sub_brainmask == 0] = 1
    sub_brainmask[sub_brainmask == 2] = 0
    sub_phys = sub_obj.get_physio(session=ses_nr, run=2, task='RS')
    func_data_list = [func_data_mni, func_data_native]

    for func_data_counter, func_data in enumerate(func_data_list):

        if func_data_counter == 0:
            mask = mni_mat
            identifier = 'MNI'
        
        elif func_data_counter == 1:
            mask = sub_brainmask
            identifier = 'native'
            del func_data_mni

        # MNI NORMAL
        func_data_matrix = func_data.get_fdata()
        dif_matrix_normal = np.divide(np.mean(func_data_matrix,axis=3), np.std(func_data_matrix,axis=3))
        dif_matrix_noinf_normal = np.nan_to_num(dif_matrix_normal, neginf=0, posinf=0)
        masked_TSNR_normal = ma.array(dif_matrix_noinf_normal, mask=mask).filled(0)
        masked_TSNR_normal[masked_TSNR_normal>500], masked_TSNR_normal[masked_TSNR_normal<-100] = 500, -100
        nib.save(nib.Nifti2Image(masked_TSNR_normal, affine=func_data.affine, header=func_data.header), basepath + '{0}/TSNR_noclean_{1}.nii.gz'.format(sub_id, identifier))

        # RETROICOR
        sub_phys_3C4R = sub_phys[RETRO_model]

        multi_01 = sub_phys_3C4R['cardiac_phase_sin_1'] * sub_phys_3C4R['respiratory_phase_sin_1']
        multi_02 = sub_phys_3C4R['cardiac_phase_sin_1'] * sub_phys_3C4R['respiratory_phase_cos_1']
        multi_03 = sub_phys_3C4R['cardiac_phase_cos_1'] * sub_phys_3C4R['respiratory_phase_sin_1']
        multi_04 = sub_phys_3C4R['cardiac_phase_cos_1'] * sub_phys_3C4R['respiratory_phase_sin_1']

        sub_phys_3C4R1M = pd.concat([sub_phys_3C4R, multi_01, multi_02, multi_03, multi_04],axis=1)
        func_data_phys_cleaned = nilearn.image.clean_img(func_data, standardize=False, detrend=False, confounds=sub_phys_3C4R1M, t_r=2.02)
        dif_matrix_RETRO = np.divide(np.mean(func_data_phys_cleaned.get_fdata(),axis=3), np.std(func_data_phys_cleaned.get_fdata(),axis=3))
        del func_data_phys_cleaned
        masked_TSNR_RETRO = ma.array(np.nan_to_num(dif_matrix_RETRO, neginf=0, posinf=0), mask=mask).filled(0)
        masked_TSNR_RETRO[masked_TSNR_RETRO>500], masked_TSNR_RETRO[masked_TSNR_RETRO<-100] = 500, -100
        nib.save(nib.Nifti2Image(masked_TSNR_RETRO, affine=func_data.affine, header=func_data.header), basepath + '{0}/TSNR_RETRO_{1}.nii.gz'.format(sub_id, identifier))

        # Aggressive AROMA

        confounds_column_index = sub_confounds.columns.tolist()
        aroma_sum = sum((itm.count("aroma_motion") for itm in confounds_column_index))
        aroma_variables = confounds_column_index[-aroma_sum:]
        func_data_aroma_cleaned = nilearn.image.clean_img(func_data, standardize=False, detrend=False, confounds=sub_confounds[aroma_variables], t_r=2.02)
        dif_matrix_aggrAROMA = np.divide(np.mean(func_data_aroma_cleaned.get_fdata(),axis=3), np.std(func_data_aroma_cleaned.get_fdata(),axis=3))
        del func_data_aroma_cleaned
        masked_TSNR_aggrAROMA = ma.array(np.nan_to_num(dif_matrix_aggrAROMA, neginf=0, posinf=0), mask=mask).filled(0)
        masked_TSNR_aggrAROMA[masked_TSNR_aggrAROMA>500], masked_TSNR_aggrAROMA[masked_TSNR_aggrAROMA<-100] = 500, -100
        nib.save(nib.Nifti2Image(masked_TSNR_aggrAROMA, affine=func_data.affine, header=func_data.header), basepath + '{0}/TSNR_aggrAROMA_{1}.nii.gz'.format(sub_id,identifier))

        # AROMA _ RETRO

        combined_aroma_retro = pd.concat([sub_phys_3C4R1M, sub_confounds[aroma_variables]], axis=1)
        func_data_aroma_retro_cleaned = nilearn.image.clean_img(func_data, standardize=False, detrend=False, confounds=combined_aroma_retro, t_r=2.02)
        dif_matrix_aggrAROMA_RETRO = np.divide(np.mean(func_data_aroma_retro_cleaned.get_fdata(),axis=3), np.std(func_data_aroma_retro_cleaned.get_fdata(),axis=3))
        del func_data_aroma_retro_cleaned
        masked_TSNR_aggrAROMA_RETRO = ma.array(np.nan_to_num(dif_matrix_aggrAROMA_RETRO, neginf=0, posinf=0), mask=mask).filled(0)
        masked_TSNR_aggrAROMA_RETRO[masked_TSNR_aggrAROMA_RETRO>500], masked_TSNR_aggrAROMA_RETRO[masked_TSNR_aggrAROMA_RETRO<-100]  = 500, -100
        nib.save(nib.Nifti2Image(masked_TSNR_aggrAROMA_RETRO, affine=func_data.affine, header=func_data.header), basepath + '{0}/TSNR_aggrAROMA_RETRO_{1}.nii.gz'.format(sub_id, identifier))

        # Difference
        difference_approaches = masked_TSNR_aggrAROMA_RETRO - masked_TSNR_RETRO
        nib.save(nib.Nifti2Image(difference_approaches, affine=func_data.affine, header=func_data.header), basepath + '{0}/TSNR_difference_aggrAROMARETRO_RETRO_{1}.nii.gz'.format(sub_id, identifier))

        # Difference AROMA
        difference_approaches_aroma = masked_TSNR_aggrAROMA_RETRO - masked_TSNR_aggrAROMA
        nib.save(nib.Nifti2Image(difference_approaches_aroma, affine=func_data.affine, header=func_data.header), basepath + '{0}/TSNR_difference_aggrAROMARETRO_aggrAROMA_{1}.nii.gz'.format(sub_id, identifier))

        # Difference aggreAroma to Normal
        difference_aggrAROMA_normal =  masked_TSNR_aggrAROMA - masked_TSNR_normal
        nib.save(nib.Nifti1Image(difference_aggrAROMA_normal, affine=func_data.affine, header=func_data.header), basepath + '{0}/TSNR_difference_aggrAROMA_normal_{1}.nii.gz'.format(sub_id, identifier))

        # Difference Normal vs aggrAROMA
        difference_normal_aggrAROMA =  masked_TSNR_normal - masked_TSNR_aggrAROMA
        nib.save(nib.Nifti2Image(difference_normal_aggrAROMA, affine=func_data.affine, header=func_data.header), basepath + '{0}/TSNR_difference_normal_aggrAROMA_{1}.nii.gz'.format(sub_id, identifier))

        # Difference Combination to Normal
        difference_aggrAROMARETRO_normal = masked_TSNR_aggrAROMA_RETRO - masked_TSNR_normal
        nib.save(nib.Nifti2Image(difference_aggrAROMARETRO_normal, affine=func_data.affine, header=func_data.header), basepath + '{0}/TSNR_difference_aggrAROMARETRO_normal_{1}.nii.gz'.format(sub_id, identifier))

        # Difference RETRO to Normal
        difference_RETRO_normal =  masked_TSNR_RETRO - masked_TSNR_normal
        nib.save(nib.Nifti2Image(difference_RETRO_normal, affine=func_data.affine, header=func_data.header), basepath + '{0}/TSNR_difference_RETRO_normal_{1}.nii.gz'.format(sub_id, identifier))

        # Difference RETRO to Normal
        difference_RETRO_aggrAROMA =  masked_TSNR_RETRO - masked_TSNR_aggrAROMA
        nib.save(nib.Nifti2Image(difference_RETRO_aggrAROMA, affine=func_data.affine, header=func_data.header), basepath + '{0}/TSNR_difference_RETRO_aggrAROMA_{1}.nii.gz'.format(sub_id, identifier))


        mask_list = [masked_TSNR_normal, masked_TSNR_RETRO, masked_TSNR_aggrAROMA, masked_TSNR_aggrAROMA_RETRO,
                     difference_approaches, difference_approaches_aroma, difference_aggrAROMA_normal, difference_normal_aggrAROMA,
                     difference_aggrAROMARETRO_normal, difference_RETRO_normal, difference_RETRO_aggrAROMA]


        if sub_id == part_list[0][-7:] and func_data_counter == 0:
            TSNR_noclean_MNI = masked_TSNR_normal[:,:,:,np.newaxis]
            TSNR_RETRO_MNI = masked_TSNR_RETRO[:,:,:,np.newaxis]
            TSNR_aggrAROMA_MNI = masked_TSNR_aggrAROMA[:,:,:,np.newaxis]
            TSNR_aggrAROMARETRO_MNI = masked_TSNR_aggrAROMA_RETRO[:,:,:,np.newaxis]
            TSNR_difference_aggrAROMARETRO_RETRO_MNI = difference_approaches[:,:,:,np.newaxis]
            TSNR_difference_aggrAROMARETRO_aggrAROMA_MNI = difference_approaches_aroma[:,:,:,np.newaxis]
            TSNR_difference_aggrAROMA_normal_MNI = difference_aggrAROMA_normal[:,:,:,np.newaxis]
            TSNR_difference_normal_aggrAROMA_MNI = difference_normal_aggrAROMA[:,:,:,np.newaxis]
            TSNR_difference_aggrAROMARETRO_normal_MNI = difference_aggrAROMARETRO_normal[:,:,:,np.newaxis]
            TSNR_difference_RETRO_normal_MNI = difference_RETRO_normal[:,:,:,np.newaxis]
            TSNR_difference_RETRO_aggrAROMA_MNI = difference_RETRO_aggrAROMA[:,:,:,np.newaxis]

            output_list = [TSNR_noclean_MNI, TSNR_RETRO_MNI, TSNR_aggrAROMA_MNI, TSNR_aggrAROMARETRO_MNI,
                           TSNR_difference_aggrAROMARETRO_RETRO_MNI, TSNR_difference_aggrAROMARETRO_aggrAROMA_MNI,
                           TSNR_difference_aggrAROMA_normal_MNI, TSNR_difference_normal_aggrAROMA_MNI,
                           TSNR_difference_aggrAROMARETRO_normal_MNI, TSNR_difference_RETRO_normal_MNI,
                           TSNR_difference_RETRO_aggrAROMA_MNI]

        elif sub_id != part_list[0][-7:] and func_data_counter == 0:
            for output_counter, output in enumerate(output_list):
                output_list[output_counter] = np.concatenate((output, mask_list[output_counter][:,:,:,np.newaxis]), axis=3)

for output in output_list:
    nib.save(nib.Nifti2Image(output, affine=func_data.affine, header=func_data.header), basepath + 'Overall_{0}).nii.gz'.format(object_name(output, globals())))



