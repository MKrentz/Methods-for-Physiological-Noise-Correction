#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 12:52:40 2021
This still contains a problem with the lack of Subject-008s LC mask. Currently it leads to mean lists of different lengths.
@author: markre
"""

import numpy as np
import nibabel as nib
import glob
from scipy import stats
from Subject_Class import Subject
from nilearn.datasets import load_mni152_brain_mask
from nilearn.image import resample_to_img
import pandas as pd

BASEPATH = '/project/3013068.03/RETROICOR/TSNR/'

MNI_mask = load_mni152_brain_mask()

# Load MNI mask to used masked data matrices
mni_mat = MNI_mask.get_fdata()
mni_mat = np.where((mni_mat == 0)|(mni_mat == 1), 1-mni_mat, mni_mat)

# Load brainstem mask to used masked data matrices
brainstem_mask = nib.load('/project/3013068.03/RETROICOR/MNI152lin_T1_2mm_brainstem_mask.nii.gz')
brainstem_mat = brainstem_mask.get_fdata()
brainstem_mat = np.where((brainstem_mat == 0)|(brainstem_mat == 1), 1-brainstem_mat, brainstem_mat)

part_list = glob.glob(BASEPATH + 'sub-*')
part_list.sort()

# Planned comparisons
var_names_MNI = ['TSNR_noclean_MNI', 'TSNR_RETRO_MNI', 'TSNR_aggrAROMA_MNI', 'TSNR_difference_aggrAROMA_uncleaned_MNI',
                 'TSNR_difference_RETRO_uncleaned_MNI',
                 'TSNR_difference_RETRO_aggrAROMA_MNI', 'TSNR_difference_aggrAROMARETRO_RETRO_MNI',
                 'TSNR_difference_aggrAROMARETRO_aggrAROMA_MNI',
                 'TSNR_difference_aggrAROMARETRO_uncleaned_MNI']

var_names_native = ['TSNR_noclean_native', 'TSNR_RETRO_native', 'TSNR_aggrAROMA_native', 'TSNR_difference_aggrAROMA_uncleaned_native',
                 'TSNR_difference_RETRO_uncleaned_native',
                 'TSNR_difference_RETRO_aggrAROMA_native', 'TSNR_difference_aggrAROMARETRO_RETRO_native',
                 'TSNR_difference_aggrAROMARETRO_aggrAROMA_native',
                 'TSNR_difference_aggrAROMARETRO_uncleaned_native']

# Create object dictionary
objects_MNI = dict.fromkeys(var_names_MNI)
objects_brainstem = dict.fromkeys(var_names_MNI)
objects_native = dict.fromkeys(var_names_native)
objects_LC = dict.fromkeys(var_names_native)
objects_gm = dict.fromkeys(var_names_native)


for subject_path in part_list:
    sub_id = subject_path[-7:]
    sub_obj = Subject(sub_id)

    # Individual LC Mask
    try:
        LC_mask = nib.load(BASEPATH + '{0}/LC_mask_native.nii.gz'.format(sub_id))
        LC_mask_mat = LC_mask.get_fdata()
        print('There is an LC Mask!')

    except:
        LC_mask = None
        print('NO MASK!!')

    # Load in, resample and binarise GM mask

    gm_mask_mat = nib.load(BASEPATH + '{0}/gm_mask_native.nii.gz'.format(sub_id)).get_fdata()

    # Account for first subject difference
    if sub_id == part_list[0][-7:]:
        for keys, values in objects_MNI.items():
            objects_MNI[keys] = nib.load(glob.glob(BASEPATH + sub_id + '/' + keys + '*')[0]).get_fdata()[:, :, :,
                            np.newaxis]

        for keys, values in objects_brainstem.items():
            objects_brainstem[keys] = [np.ma.array(nib.load(glob.glob(BASEPATH + sub_id + '/' + keys + '*')[0]).get_fdata(), mask=brainstem_mat)]

        for keys, values in objects_native.items():
            objects_native[keys] = [nib.load(glob.glob(BASEPATH + sub_id + '/' + keys + '*')[0]).get_fdata()]

        if LC_mask != None:
            for keys, values in objects_LC.items():
                objects_LC[keys] = [np.ma.array(nib.load(glob.glob(BASEPATH + sub_id + '/' + keys + '*')[0]).get_fdata(), mask=LC_mask_mat)]

        for keys, values in objects_gm.items():
            objects_gm[keys] = [np.ma.array(nib.load(glob.glob(BASEPATH + sub_id + '/' + keys + '*')[0]).get_fdata(), mask=gm_mask_mat)]

    else:
        for keys, values in objects_MNI.items():
            objects_MNI[keys] = np.concatenate(
                (values, nib.load(glob.glob(BASEPATH + sub_id + '/' + keys + '*')[0]).get_fdata()[:, :, :, np.newaxis]),
                axis=3)
        for keys, values in objects_brainstem.items():
            objects_brainstem[keys] = values + [np.ma.array(nib.load(glob.glob(BASEPATH + sub_id + '/' + keys + '*')[0]).get_fdata(), mask=brainstem_mat)]

        for keys, values in objects_native.items():
            objects_native[keys] = values + [nib.load(glob.glob(BASEPATH + sub_id + '/' + keys + '*')[0]).get_fdata()]

        for keys, values in objects_gm.items():
            objects_gm[keys] = values + [np.ma.array(nib.load(glob.glob(BASEPATH + sub_id + '/' + keys + '*')[0]).get_fdata(), mask=gm_mask_mat)]

        if LC_mask != None:
            for keys, values in objects_LC.items():
                objects_LC[keys]= values + [np.ma.array(nib.load(glob.glob(BASEPATH + sub_id + '/' + keys + '*')[0]).get_fdata(), mask=LC_mask_mat)]


# Mean Matrices for MNI template
mean_MNI_matrix = objects_MNI
mean_MNI_value = dict.fromkeys(var_names_MNI)
for keys, values in mean_MNI_matrix.items():
    mean_MNI_value[keys] = np.mean(values, axis=(0,1,2))
mean_MNI_value_df = pd.DataFrame(mean_MNI_value)
mean_MNI_value_df['percent_increase_RETRO'] = ((mean_MNI_value_df['TSNR_RETRO_MNI'] / mean_MNI_value_df['TSNR_noclean_MNI']) - 1) * 100
mean_MNI_value_df['percent_increase_AROMA'] = ((mean_MNI_value_df['TSNR_aggrAROMA_MNI'] / mean_MNI_value_df['TSNR_noclean_MNI']) - 1) * 100
mean_MNI_value_df['mean_average_RETROAROMA'] = (mean_MNI_value_df['percent_increase_RETRO']+mean_MNI_value_df['percent_increase_AROMA']) / 2
mean_MNI_value_df['percent_increase_RETRO_mean_scaled'] = mean_MNI_value_df['percent_increase_RETRO'] + mean_MNI_value_df['mean_average_RETROAROMA'] - np.mean(mean_MNI_value_df['percent_increase_RETRO'])
mean_MNI_value_df['percent_increase_AROMA_mean_scaled'] = mean_MNI_value_df['percent_increase_AROMA'] + mean_MNI_value_df['mean_average_RETROAROMA'] - np.mean(mean_MNI_value_df['percent_increase_AROMA'])

mean_MNI_value_df.to_csv(BASEPATH + 'MNI_means.txt')

# Mean matrices for brainstem template
mean_brainstem_matrix = objects_brainstem
mean_brainstem_value = dict.fromkeys(var_names_MNI)
for keys, values in mean_brainstem_matrix.items():
    mean_brainstem_value[keys] = [np.ma.mean(x) for x in values]
mean_brainstem_value_df = pd.DataFrame(mean_brainstem_value)
mean_brainstem_value_df['percent_increase_RETRO'] = ((mean_brainstem_value_df['TSNR_RETRO_MNI'] / mean_brainstem_value_df['TSNR_noclean_MNI']) - 1) * 100
mean_brainstem_value_df['percent_increase_AROMA'] = ((mean_brainstem_value_df['TSNR_aggrAROMA_MNI'] / mean_brainstem_value_df['TSNR_noclean_MNI']) - 1) * 100
mean_brainstem_value_df['mean_average_RETROAROMA'] = (mean_brainstem_value_df['percent_increase_RETRO']+mean_brainstem_value_df['percent_increase_AROMA']) / 2
mean_brainstem_value_df['percent_increase_RETRO_mean_scaled'] = mean_brainstem_value_df['percent_increase_RETRO'] + mean_brainstem_value_df['mean_average_RETROAROMA'] - np.mean(mean_brainstem_value_df['percent_increase_RETRO'])
mean_brainstem_value_df['percent_increase_AROMA_mean_scaled'] = mean_brainstem_value_df['percent_increase_AROMA'] + mean_brainstem_value_df['mean_average_RETROAROMA'] - np.mean(mean_brainstem_value_df['percent_increase_AROMA'])
mean_brainstem_value_df.to_csv(BASEPATH + 'brainstem_means.txt')

# Mean Matrices for Grey Matter
mean_gm_matrix = objects_gm
mean_gm_value = dict.fromkeys(var_names_native)
for keys, values in mean_gm_matrix.items():
    mean_gm_value[keys] = [np.ma.mean(x) for x in values]
mean_gm_value_df = pd.DataFrame(mean_gm_value)
mean_gm_value_df['percent_increase_RETRO'] = ((mean_gm_value_df['TSNR_RETRO_native'] / mean_gm_value_df['TSNR_noclean_native']) - 1) * 100
mean_gm_value_df['percent_increase_AROMA'] = ((mean_gm_value_df['TSNR_aggrAROMA_native'] / mean_gm_value_df['TSNR_noclean_native']) - 1) * 100
mean_gm_value_df['mean_average_RETROAROMA'] = (mean_gm_value_df['percent_increase_RETRO']+mean_gm_value_df['percent_increase_AROMA']) / 2
mean_gm_value_df['percent_increase_RETRO_mean_scaled'] = mean_gm_value_df['percent_increase_RETRO'] + mean_gm_value_df['mean_average_RETROAROMA'] - np.mean(mean_gm_value_df['percent_increase_RETRO'])
mean_gm_value_df['percent_increase_AROMA_mean_scaled'] = mean_gm_value_df['percent_increase_AROMA'] + mean_gm_value_df['mean_average_RETROAROMA'] - np.mean(mean_gm_value_df['percent_increase_AROMA'])

mean_gm_value_df.to_csv(BASEPATH + 'graymatter_means.txt')

# Mean Matrices for LC
mean_LC_matrix = objects_LC
mean_LC_value = dict.fromkeys(var_names_native)
for keys, values in mean_LC_matrix.items():
    mean_LC_value[keys] = [np.ma.mean(x) for x in values]
mean_LC_value_df = pd.DataFrame(mean_LC_value)
mean_LC_value_df['percent_increase_RETRO'] = ((mean_LC_value_df['TSNR_RETRO_native'] / mean_LC_value_df['TSNR_noclean_native']) - 1) * 100
mean_LC_value_df['percent_increase_AROMA'] = ((mean_LC_value_df['TSNR_aggrAROMA_native'] / mean_LC_value_df['TSNR_noclean_native']) - 1) * 100
mean_LC_value_df['mean_average_RETROAROMA'] = (mean_LC_value_df['percent_increase_RETRO'] + mean_LC_value_df['percent_increase_AROMA']) / 2
mean_LC_value_df['percent_increase_RETRO_mean_scaled'] = mean_LC_value_df['percent_increase_RETRO'] + mean_LC_value_df['mean_average_RETROAROMA'] - np.mean(mean_LC_value_df['percent_increase_RETRO'])
mean_LC_value_df['percent_increase_AROMA_mean_scaled'] = mean_LC_value_df['percent_increase_AROMA'] + mean_LC_value_df['mean_average_RETROAROMA'] - np.mean(mean_LC_value_df['percent_increase_AROMA'])

mean_LC_value_df.to_csv(BASEPATH + 'LC_means.txt')

# Stats
# Stats for MNI
# TSNR non-cleaned vs TSNR-RETRO



stats_list = [mean_MNI_value_df, mean_brainstem_value_df, mean_gm_value_df, mean_LC_value_df]
space_name= ['MNI', 'Brainstem', 'GrayMatter', 'LC']
dim_dic = dict.fromkeys(space_name)
results_dic = {'RETROICOR Cleaned': dim_dic, 'AROMA Cleaned': dim_dic, 'Unique RETROICOR Effect': dim_dic, 'Unique AROMA Effect': dim_dic, 'Percent RETROICOR Effect': dim_dic, 'Percent AROMA Effect': dim_dic}
results_df = pd.DataFrame(results_dic)

counter = 0
for index, row in results_df.iterrows():
    if index == 'MNI' or index == 'Brainstem':
        results_df['RETROICOR Cleaned'][index] = stats.ttest_rel(stats_list[counter]['TSNR_noclean_MNI'], stats_list[counter]['TSNR_RETRO_MNI'])
        results_df['AROMA Cleaned'][index] = stats.ttest_rel(stats_list[counter]['TSNR_noclean_MNI'], stats_list[counter]['TSNR_aggrAROMA_MNI'])

        results_df['Unique RETROICOR Effect'][index] = (
            stats.ttest_1samp(stats_list[counter]['percent_increase_RETRO'], popmean=0))
        results_df['Unique RETROICOR Effect'][index] = (
            stats.ttest_1samp(stats_list[counter]['percent_increase_AROMA'], popmean=0))
        results_df['Percent RETROICOR Effect'][index] = (stats.ttest_1samp(stats_list[counter]['TSNR_difference_aggrAROMARETRO_aggrAROMA_MNI'], popmean = 0))
        results_df['Percent AROMA Effect'][index] = (stats.ttest_1samp(stats_list[counter]['TSNR_difference_aggrAROMARETRO_RETRO_MNI'], popmean = 0))
        counter += 1

    elif index == 'GrayMatter' or index == 'LC':
        results_df['RETROICOR Cleaned'][index] = stats.ttest_rel(stats_list[counter]['TSNR_noclean_native'], stats_list[counter]['TSNR_RETRO_native'])
        results_df['AROMA Cleaned'][index] = stats.ttest_rel(stats_list[counter]['TSNR_noclean_native'], stats_list[counter]['TSNR_aggrAROMA_native'])
        results_df['Unique RETROICOR Effect'][index] = (
            stats.ttest_1samp(stats_list[counter]['percent_increase_RETRO'], popmean=0))
        results_df['Unique RETROICOR Effect'][index] = (
            stats.ttest_1samp(stats_list[counter]['percent_increase_AROMA'], popmean=0))
        results_df['Percent RETROICOR Effect'][index] = (stats.ttest_1samp(stats_list[counter]['TSNR_difference_aggrAROMARETRO_aggrAROMA_native'], popmean = 0))
        results_df['Percent AROMA Effect'][index] = (stats.ttest_1samp(stats_list[counter]['TSNR_difference_aggrAROMARETRO_RETRO_native'], popmean = 0))
        counter += 1


results_df.to_csv(BASEPATH + 'stats_results.txt', sep = ' ')

# Add Percent transform
# Overall/Within mask
# Z change / Summe oder Average
