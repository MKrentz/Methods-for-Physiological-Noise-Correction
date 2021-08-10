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
from nilearn.image import resample_to_img
import nilearn

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
objects_LC = dict.fromkeys(var_names_native)
objects_gm = dict.fromkeys(var_names_native)

#Mean Matrices for LC
stress_list = ['sub-002', 'sub-003', 'sub-004', 'sub-007', 'sub-009', 'sub-013', 'sub-015', 'sub-017', 'sub-021', 'sub-023', 'sub-025', 'sub-027', 'sub-029']

for subject_path in part_list:
    sub_id = subject_path[-7:]
    sub_obj = Subject(sub_id)

    # Account for balancing in stress/control session order
    ses_nr = 2 if sub_id in stress_list else 1
    sub_func = sub_obj.get_func_data(run=2, session=ses_nr)

    # Individual LC Mask
    LC_mask = sub_obj.get_LC()
    LC_mask_native = resample_to_img(LC_mask, sub_func, interpolation='nearest')
    LC_mask_mat = LC_mask_native.get_fdata()
    LC_mask_mat[LC_mask_mat == 1] = 2
    LC_mask_mat[LC_mask_mat == 0] = 1
    LC_mask_mat[LC_mask_mat == 2] = 0

    gm_mask = nib.load('/project/3013068.03/derivate/fmriprep/{0}/anat/{0}_desc-aparcaseg_dseg.nii.gz'.format(sub_id))
    gm_mask_mat = gm_mask.get_fdata()
    gm_mask_mat[gm_mask_mat < 1000] = 0
    gm_mask_mat[gm_mask_mat >= 1000] = 1
    gm_mask_native = nib.Nifti2Image(gm_mask_mat, gm_mask.affine, gm_mask.header)
    gm_mask_native = resample_to_img(gm_mask_native, sub_func, interpolation='nearest')
    gm_mask_mat = gm_mask_native.get_fdata()
    gm_mask_mat[gm_mask_mat == 1] = 2
    gm_mask_mat[gm_mask_mat == 0] = 1
    gm_mask_mat[gm_mask_mat == 2] = 0

    if sub_id == part_list[0][-7:]:
        for keys, values in objects_MNI.items():
            objects_MNI[keys] = nib.load(glob.glob(basepath + sub_id + '/' + keys + '*')[0]).get_fdata()[:, :, :,
                            np.newaxis]

        for keys, values in objects_native.items():
            objects_native[keys] = [nib.load(glob.glob(basepath + sub_id + '/' + keys + '*')[0]).get_fdata()]

        for keys, values in objects_LC.items():
            objects_LC[keys] = [np.ma.array(nib.load(glob.glob(basepath + sub_id + '/' + keys + '*')[0]).get_fdata(), mask=LC_mask_mat).filled(0)]

        for keys, values in objects_gm.items():
            objects_gm[keys] = [np.ma.array(nib.load(glob.glob(basepath + sub_id + '/' + keys + '*')[0]).get_fdata(), mask=gm_mask_mat).filled(0)]

    else:
        for keys, values in objects_MNI.items():
            objects_MNI[keys] = np.concatenate(
                (values, nib.load(glob.glob(basepath + sub_id + '/' + keys + '*')[0]).get_fdata()[:, :, :, np.newaxis]),
                axis=3)

        for keys, values in objects_native.items():
            objects_native[keys] = values + [nib.load(glob.glob(basepath + sub_id + '/' + keys + '*')[0]).get_fdata()]

        for keys, values in objects_LC.items():
            objects_LC[keys]= values + [np.ma.array(nib.load(glob.glob(basepath + sub_id + '/' + keys + '*')[0]).get_fdata(), mask=LC_mask_mat)]

        for keys, values in objects_gm.items():
            objects_gm[keys] = values + [np.ma.array(nib.load(glob.glob(basepath + sub_id + '/' + keys + '*')[0]).get_fdata(), mask=gm_mask_mat)]


# Mean Matrices for MNI template
mean_MNI_matrix = objects_MNI
mean_MNI_value = dict.fromkeys(var_names_MNI)
mean_brainstem_value = dict.fromkeys(var_names_MNI)
for keys, values in mean_MNI_matrix.items():
    mean_MNI_matrix[keys] = np.mean(values, axis=3)
    mean_MNI_value[keys] = np.ma.array(mean_MNI_matrix[keys], mask=mni_mat).mean()
    mean_brainstem_value[keys] = np.ma.array(mean_MNI_matrix[keys], mask=brainstem_mat).mean()

# Mean Matrices for Grey Matter
mean_gm_matrix = objects_native
mean_gm_value = dict.fromkeys(var_names_native)
for keys, values in mean_gm_matrix.items():
    mean_gm_value = [np.ma.mean(x) for x in values]

#Mean Matrices for LC
mean_LC_matrix = objects_LC
mean_LC_value = dict.fromkeys(var_names_native)
for keys, values in mean_LC_matrix.items():
    mean_LC_value[keys] = [np.ma.mean(x) for x in values]


# Stats
# Stats for MNI
# TSNR non-cleaned vs TSNR-RETRO
stats.ttest_rel(mean_LC_value['TSNR_noclean_native'], mean_LC_value['TSNR_RETRO_native'])
"""
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
"""