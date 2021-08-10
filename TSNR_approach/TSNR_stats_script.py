#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 12:52:40 2021

@author: markre
"""
import numpy as np
import nibabel as nib
import glob
from scipy import stats
from Subject_Class import Subject
from nilearn.datasets import load_mni152_brain_mask

basepath = '/project/3013068.03/RETROICOR/TSNR/'

MNI_mask = load_mni152_brain_mask()

# Load MNI mask to used masked data matrices
mni_mat = MNI_mask.get_fdata()
mni_mat[mni_mat == 1] = 2
mni_mat[mni_mat == 0] = 1
mni_mat[mni_mat == 2] = 0

# Load brainstem mask to used masked data matrices
brainstem_mask = nib.load('/project/3013068.03/RETROICOR/MNI152lin_T1_2mm_brainstem_mask.nii.gz')
brainstem_mat = brainstem_mask.get_fdata()
brainstem_mat[brainstem_mat == 1] = 2
brainstem_mat[brainstem_mat == 0] = 1
brainstem_mat[brainstem_mat == 2] = 0

part_list = glob.glob(basepath + 'sub-*')
part_list.sort()
part_list = part_list[:3]

# Planned comparisons
var_names_MNI = ['TSNR_noclean_MNI', 'TSNR_RETRO_MNI', 'TSNR_aggrAROMA_MNI', 'TSNR_difference_aggrAROMA_normal_MNI',
                 'TSNR_difference_RETRO_normal_MNI',
                 'TSNR_difference_RETRO_aggrAROMA_MNI', 'TSNR_difference_aggrAROMARETRO_RETRO_MNI',
                 'TSNR_difference_aggrAROMARETRO_aggrAROMA_MNI',
                 'TSNR_difference_aggrAROMARETRO_normal_MNI']

var_names_native = ['TSNR_noclean_native', 'TSNR_RETRO_native', 'TSNR_aggrAROMA_native', 'TSNR_difference_aggrAROMA_normal_native',
                 'TSNR_difference_RETRO_normal_native',
                 'TSNR_difference_RETRO_aggrAROMA_native', 'TSNR_difference_aggrAROMARETRO_RETRO_native',
                 'TSNR_difference_aggrAROMARETRO_aggrAROMA_native',
                 'TSNR_difference_aggrAROMARETRO_normal_native']

# Create object dictionary
objects_MNI = dict.fromkeys(var_names_MNI)
objects_native = dict.fromkeys(var_names_native)

for subject_path in part_list:
    sub_id = subject_path[-7:]
    sub_obj = Subject(sub_id)

    if sub_id == part_list[0][-7:]:
        for keys, values in objects_MNI.items():
            objects_MNI[keys] = nib.load(glob.glob(basepath + sub_id + '/' + keys + '*')[0]).get_fdata()[:, :, :,
                            np.newaxis]
        for keys, values in objects_native.items():
            objects_native[keys] = nib.load(glob.glob(basepath + sub_id + '/' + keys + '*')[0]).get_fdata()[:, :, :,
                            np.newaxis]
    else:
        for keys, values in objects_MNI.items():
            objects_MNI[keys] = np.concatenate(
                (values, nib.load(glob.glob(basepath + sub_id + '/' + keys + '*')[0]).get_fdata()[:, :, :, np.newaxis]),
                axis=3)
        for keys, values in objects_native.items():
            objects_native[keys] = np.concatenate(
                (values, nib.load(glob.glob(basepath + sub_id + '/' + keys + '*')[0]).get_fdata()[:, :, :, np.newaxis]),
                axis=3)


# Mean Matrices for MNI template
mean_MNI_matrix = objects_MNI
mean_MNI_value = dict.fromkeys(var_names_MNI)
for keys, values in mean_MNI_matrix.items():
    mean_MNI_matrix[keys] = np.mean(values, axis=3)
    mean_MNI_value[keys] = np.ma.array(mean_MNI_matrix[keys], mask=mni_mat).mean()

# Mean Matrices for brainstem templace
mean_brainstem_matrix = objects_MNI
mean_brainstem_value = dict.fromkeys(var_names_MNI)
for keys, values in mean_brainstem_matrix.items():
    mean_brainstem_matrix[keys] = np.mean(values, axis=3)
    mean_brainstem_value[keys] = np.ma.array(mean_brainstem_matrix[keys], mask=brainstem_mat).mean()

# Mean Matrices for Grey Matter
mean_gm_matrix = objects_native
mean_gm_value = dict.fromkeys(var_names_native)

#Mean Matrices for LC
LC_mask = sub_obj.get_LC()
mean_LC_matrix = objects_native
mean_LC_value = dict.fromkeys(var_names_native)

# Stats
# Stats for MNI
# TSNR non-cleaned vs TSNR-RETRO
stats.ttest_rel(Mean_Vector_TSNR_noclean_MNI, Mean_Vector_RETRO_MNI)
stats.ttest_rel(Mean_Vector_TSNR_noclean_MNI, Mean_Vector_aggrAROMA_MNI)
stats.ttest_1samp(Mean_Vector_aggrAROMARETRO_RETRO_MNI, population=0)
stats.describe(Mean_Vector_aggrAROMARETRO_RETRO_MNI)
stats.ttest_1samp(Mean_Vector_aggrAROMARETRO_aggrAROMA_MNI, population=0)
stats.describe(Mean_Vector_aggrAROMARETRO_aggrAROMA_MNI)
# Stats for brainstem
stats.ttest_rel(Mean_Vector_TSNR_noclean_brainstem, Mean_Vector_RETRO_brainstem)
stats.ttest_rel(Mean_Vector_TSNR_noclean_brainstem, Mean_Vector_aggrAROMA_brainstem)
stats.ttest_1samp(Mean_Vector_aggrAROMARETRO_RETRO_brainstem, population=0)
stats.describe(Mean_Vector_aggrAROMARETRO_RETRO_brainstem)
stats.ttest_1samp(Mean_Vector_aggrAROMARETRO_aggrAROMA_brainstem, population=0)
stats.describe(Mean_Vector_aggrAROMARETRO_aggrAROMA_brainstem)
