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
import pandas as pd
from numpy import mean, std
from math import sqrt

BASEPATH = '/project/3013068.03/test/TSNR_approach/'

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
part_list.remove('/project/3013068.03/test/TSNR_approach/sub-008')


# Planned comparisons
var_names_MNI = ['tsnr_noclean_MNI', 'tsnr_retro_MNI', 'tsnr_aroma_MNI', 'tsnr_acompcor_MNI', 'tsnr_aroma_retro_MNI', 'tsnr_aroma_acompcor_MNI', 'tsnr_aroma_retro_acompcor_MNI', 'tsnr_difference_unique_retro_to_aroma_MNI',
    'tsnr_difference_unique_aroma_to_retro_MNI', 'tsnr_difference_unique_acompcor_to_aroma_MNI','tsnr_difference_unique_retro_to_aroma_acompcor_MNI',
    'tsnr_difference_aroma_to_uncleaned_MNI',
    'tsnr_difference_aroma_retro_to_uncleaned_MNI', 'tsnr_difference_retro_to_uncleaned_MNI',
    'tsnr_difference_percent_retro_to_uncleaned_MNI',
    'tsnr_difference_percent_aroma_to_uncleaned_MNI', 'tsnr_difference_percent_acompcor_to_uncleaned_MNI', 'tsnr_difference_percent_unique_aroma_to_retro_MNI' ,'tsnr_difference_percent_unique_retro_to_aroma_MNI', 
    'tsnr_difference_percent_unique_acompcor_to_aroma_MNI', 'tsnr_difference_percent_unique_retro_to_aroma_acompcor_MNI', 'tsnr_difference_percent_unique_aroma_to_retro_vs_uncleaned_MNI',
    'tsnr_difference_percent_unique_retro_to_aroma_vs_uncleaned_MNI', 'tsnr_difference_percent_unique_acompcor_to_aroma_vs_uncleaned_MNI',
    'tsnr_difference_percent_unique_retro_to_aroma_acompcor_vs_uncleaned_MNI']

var_names_native =  ['tsnr_noclean_native', 'tsnr_retro_native', 'tsnr_aroma_native', 'tsnr_acompcor_native', 'tsnr_aroma_retro_native', 'tsnr_aroma_acompcor_native', 'tsnr_aroma_retro_acompcor_native', 'tsnr_difference_unique_retro_to_aroma_native',
    'tsnr_difference_unique_aroma_to_retro_native', 'tsnr_difference_unique_acompcor_to_aroma_native','tsnr_difference_unique_retro_to_aroma_acompcor_native',
    'tsnr_difference_aroma_to_uncleaned_native',
    'tsnr_difference_aroma_retro_to_uncleaned_native', 'tsnr_difference_retro_to_uncleaned_native',
    'tsnr_difference_percent_retro_to_uncleaned_native',
    'tsnr_difference_percent_aroma_to_uncleaned_native', 'tsnr_difference_percent_acompcor_to_uncleaned_native', 'tsnr_difference_percent_unique_aroma_to_retro_native' ,'tsnr_difference_percent_unique_retro_to_aroma_native', 
    'tsnr_difference_percent_unique_acompcor_to_aroma_native', 'tsnr_difference_percent_unique_retro_to_aroma_acompcor_native', 'tsnr_difference_percent_unique_aroma_to_retro_vs_uncleaned_native',
    'tsnr_difference_percent_unique_retro_to_aroma_vs_uncleaned_native', 'tsnr_difference_percent_unique_acompcor_to_aroma_vs_uncleaned_native',
    'tsnr_difference_percent_unique_retro_to_aroma_acompcor_vs_uncleaned_native']

# Create object dictionary
mean_MNI = pd.DataFrame(index=[sub[-7:] for sub in part_list], columns=var_names_MNI)
mean_brainstem = pd.DataFrame(index=[sub[-7:] for sub in part_list], columns=var_names_MNI)
mean_native = pd.DataFrame(index=[sub[-7:] for sub in part_list], columns=var_names_native)
mean_gm = pd.DataFrame(index=[sub[-7:] for sub in part_list], columns=var_names_native)
mean_LC = pd.DataFrame(index=[sub[-7:] for sub in part_list], columns=var_names_native)


for subject_path in part_list:
    sub_id = subject_path[-7:]
    sub_obj = Subject(sub_id)

    # Individual LC Mask
    try:
        LC_mask = nib.load(BASEPATH + '{0}/masks/LC_mask_native.nii.gz'.format(sub_id))
        LC_mask_mat = LC_mask.get_fdata()
        print('There is an LC Mask!')

    except:
        LC_mask = None
        print('NO MASK!!')

    # Load in, resample and binarise GM mask

    gm_mask_mat = nib.load(BASEPATH + '{0}/masks/gm_mask_native.nii.gz'.format(sub_id)).get_fdata()

    # Account for first subject difference
    for data_type in mean_MNI.columns:
        mean_MNI.loc[sub_id, data_type] = np.mean(nib.load(glob.glob(BASEPATH + sub_id + '/glms/' + data_type + '*')[0]).get_fdata()[:, :, :,
                        np.newaxis])

        mean_brainstem.loc[sub_id, data_type] = np.mean(np.ma.array(nib.load(glob.glob(BASEPATH + sub_id + '/glms/' + data_type + '*')[0]).get_fdata(), mask=brainstem_mat))

    for data_type, values in mean_native.items():
        mean_gm.loc[sub_id, data_type] = np.mean(np.ma.array(nib.load(glob.glob(BASEPATH + sub_id + '/glms/' + data_type + '*')[0]).get_fdata(), mask=gm_mask_mat))          
        if LC_mask != None:
                mean_LC.loc[sub_id, data_type] = np.mean(np.ma.array(nib.load(glob.glob(BASEPATH + sub_id + '/glms/' + data_type + '*')[0]).get_fdata(), mask=LC_mask_mat))


mean_MNI.to_csv(BASEPATH + 'MNI_means.txt')
mean_brainstem.to_csv(BASEPATH + 'brainstem_means.txt')
mean_gm.to_csv(BASEPATH + 'graymatter_means.txt')
mean_LC.to_csv(BASEPATH + 'LC_means.txt')


# Stats
# Stats for MNI
# TSNR non-cleaned vs TSNR-RETRO
''''tsnr_noclean_MNI', 'tsnr_retro_MNI', 'tsnr_aroma_MNI', 'tsnr_acompcor_MNI', 'tsnr_aroma_retro_MNI', 'tsnr_aroma_acompcor_MNI', 'tsnr_aroma_retro_acompcor_MNI', 'tsnr_difference_unique_retro_to_aroma_MNI',
    'tsnr_difference_unique_aroma_to_retro_MNI', 'tsnr_difference_unique_acompcor_to_aroma_MNI','tsnr_difference_unique_retro_to_aroma_acompcor_MNI',
    'tsnr_difference_aroma_to_uncleaned_MNI',
    'tsnr_difference_aroma_retro_to_uncleaned_MNI', 'tsnr_difference_retro_to_uncleaned_MNI',
    'tsnr_difference_percent_retro_to_uncleaned_MNI',
    'tsnr_difference_percent_aroma_to_uncleaned_MNI', 'tsnr_difference_percent_acompcor_to_uncleaned_MNI', 'tsnr_difference_percent_unique_aroma_to_retro_MNI' ,'tsnr_difference_percent_unique_retro_to_aroma_MNI', 
    'tsnr_difference_percent_unique_acompcor_to_aroma_MNI', 'tsnr_difference_percent_unique_retro_to_aroma_acompcor_MNI', 'tsnr_difference_percent_unique_aroma_to_retro_vs_uncleaned_MNI',
    'tsnr_difference_percent_unique_retro_to_aroma_vs_uncleaned_MNI', 'tsnr_difference_percent_unique_acompcor_to_aroma_vs_uncleaned_MNI',
    'tsnr_difference_percent_unique_retro_to_aroma_acompcor_vs_uncleaned_MNI']'''

stats_list = [mean_MNI, mean_brainstem, mean_gm, mean_LC]
space_name= ['MNI', 'Brainstem', 'GrayMatter', 'LC']
output_names = ['RETROICOR Cleaned', 'AROMA Cleaned', 'aCompCor Cleaned', 'Unique aCompCor Effect to AROMA', 'Unique RETROICOR Effect to AROMA', 'Unique RETROICOR Effect to AROMA and aCompCor',
                'Percent RETROICOR Effect', 'Percent AROMA Effect', 'Percent aCompCor Effect', 'Percent RETROICOR Effect vs AROMA', 'Percent RETROICOR Effect vs AROMA and aCompCor']

results_df = pd.DataFrame(index=output_names, columns=space_name)

def cohen_d(x,y):
        return (mean(x) - mean(y)) / sqrt((std(x, ddof=1) ** 2 + std(y, ddof=1) ** 2) / 2.0)
def cohen_d_within(x):
    return mean(x) / std(x, ddof = 1)

for counter, index in enumerate(results_df.columns):
    if index == 'MNI' or index == 'Brainstem':
        results_df.loc['RETROICOR Cleaned', index] = [stats.ttest_rel(stats_list[counter]['tsnr_noclean_MNI'], stats_list[counter]['tsnr_retro_MNI']),
                      cohen_d(stats_list[counter]['tsnr_noclean_MNI'], stats_list[counter]['tsnr_retro_MNI'])]
        results_df.loc['AROMA Cleaned', index] = [stats.ttest_rel(stats_list[counter]['tsnr_noclean_MNI'], stats_list[counter]['tsnr_aroma_MNI']),
                      cohen_d(stats_list[counter]['tsnr_noclean_MNI'], stats_list[counter]['tsnr_aroma_MNI'])]
        results_df.loc['aCompCor Cleaned', index] = [stats.ttest_rel(stats_list[counter]['tsnr_noclean_MNI'], stats_list[counter]['tsnr_acompcor_MNI']),
                      cohen_d(stats_list[counter]['tsnr_noclean_MNI'], stats_list[counter]['tsnr_acompcor_MNI'])]
        results_df.loc['Unique aCompCor Effect to AROMA', index] = [stats.ttest_1samp(stats_list[counter]['tsnr_difference_unique_acompcor_to_aroma_MNI'], popmean=0),
                      cohen_d_within(stats_list[counter]['tsnr_difference_unique_acompcor_to_aroma_MNI'])]
        results_df.loc['Unique RETROICOR Effect to AROMA', index] = [stats.ttest_1samp(stats_list[counter]['tsnr_difference_unique_retro_to_aroma_MNI'], popmean=0),
                      cohen_d_within(stats_list[counter]['tsnr_difference_unique_retro_to_aroma_MNI'])]
        results_df.loc['Unique RETROICOR Effect to AROMA and aCompCor', index] = [stats.ttest_1samp(stats_list[counter]['tsnr_difference_unique_retro_to_aroma_acompcor_MNI'], popmean=0),
                      cohen_d_within(stats_list[counter]['tsnr_difference_unique_retro_to_aroma_acompcor_MNI'])]
        results_df.loc['Percent aCompCor Effect to AROMA', index] =  [stats.ttest_1samp(stats_list[counter]['tsnr_difference_percent_unique_acompcor_to_aroma_vs_uncleaned_MNI'], popmean=0),
                      cohen_d_within(stats_list[counter]['tsnr_difference_percent_unique_acompcor_to_aroma_vs_uncleaned_MNI'])]
        results_df.loc['Percent AROMA Effect', index] = [stats.ttest_1samp(stats_list[counter]['tsnr_difference_percent_aroma_to_uncleaned_MNI'], popmean = 0),
                      cohen_d_within(stats_list[counter]['tsnr_difference_percent_aroma_to_uncleaned_MNI'])]
        results_df.loc['Percent aCompCor Effect', index] = [stats.ttest_1samp(stats_list[counter]['tsnr_difference_percent_acompcor_to_uncleaned_MNI'], popmean = 0),
                      cohen_d_within(stats_list[counter]['tsnr_difference_percent_acompcor_to_uncleaned_MNI'])]
        results_df.loc['Percent RETROICOR Effect', index] = [stats.ttest_1samp(stats_list[counter]['tsnr_difference_percent_retro_to_uncleaned_MNI'], popmean = 0),
                      cohen_d_within(stats_list[counter]['tsnr_difference_percent_retro_to_uncleaned_MNI'])]
        results_df.loc['Percent RETROICOR Effect vs AROMA', index] = [stats.ttest_1samp(stats_list[counter]['tsnr_difference_percent_unique_retro_to_aroma_vs_uncleaned_MNI'], popmean = 0),
                       cohen_d_within(stats_list[counter]['tsnr_difference_percent_unique_retro_to_aroma_vs_uncleaned_MNI'])]
        results_df.loc['Percent RETROICOR Effect vs AROMA and aCompCor', index] = [stats.ttest_1samp(stats_list[counter]['tsnr_difference_percent_unique_retro_to_aroma_acompcor_vs_uncleaned_MNI'], popmean = 0),
                      cohen_d_within(stats_list[counter]['tsnr_difference_percent_unique_retro_to_aroma_acompcor_vs_uncleaned_MNI'])]
   
    elif index == 'GrayMatter' or index == 'LC':
        results_df.loc['RETROICOR Cleaned', index] = [stats.ttest_rel(stats_list[counter]['tsnr_noclean_native'], stats_list[counter]['tsnr_retro_native']),
                      cohen_d(stats_list[counter]['tsnr_noclean_native'], stats_list[counter]['tsnr_retro_native'])]
        results_df.loc['AROMA Cleaned', index] = [stats.ttest_rel(stats_list[counter]['tsnr_noclean_native'], stats_list[counter]['tsnr_aroma_native']),
                      cohen_d(stats_list[counter]['tsnr_noclean_native'], stats_list[counter]['tsnr_aroma_native'])]
        results_df.loc['aCompCor Cleaned', index] = [stats.ttest_rel(stats_list[counter]['tsnr_noclean_native'], stats_list[counter]['tsnr_acompcor_native']),
                      cohen_d(stats_list[counter]['tsnr_noclean_native'], stats_list[counter]['tsnr_acompcor_native'])]
        results_df.loc['Unique aCompCor Effect to AROMA', index] = [stats.ttest_1samp(stats_list[counter]['tsnr_difference_unique_acompcor_to_aroma_native'], popmean=0),
                      cohen_d_within(stats_list[counter]['tsnr_difference_unique_acompcor_to_aroma_native'])]
        results_df.loc['Unique RETROICOR Effect to AROMA', index] = [stats.ttest_1samp(stats_list[counter]['tsnr_difference_unique_retro_to_aroma_native'], popmean=0),
                      cohen_d_within(stats_list[counter]['tsnr_difference_unique_retro_to_aroma_native'])]
        results_df.loc['Unique RETROICOR Effect to AROMA and aCompCor', index] = [stats.ttest_1samp(stats_list[counter]['tsnr_difference_unique_retro_to_aroma_acompcor_native'], popmean=0),
                      cohen_d_within(stats_list[counter]['tsnr_difference_unique_retro_to_aroma_acompcor_native'])]
        results_df.loc['Percent aCompCor Effect to AROMA', index] =  [stats.ttest_1samp(stats_list[counter]['tsnr_difference_percent_unique_acompcor_to_aroma_vs_uncleaned_native'], popmean=0),
                      cohen_d_within(stats_list[counter]['tsnr_difference_percent_unique_acompcor_to_aroma_vs_uncleaned_native'])]
        results_df.loc['Percent AROMA Effect', index] = [stats.ttest_1samp(stats_list[counter]['tsnr_difference_percent_aroma_to_uncleaned_native'], popmean = 0),
                      cohen_d_within(stats_list[counter]['tsnr_difference_percent_aroma_to_uncleaned_native'])]
        results_df.loc['Percent aCompCor Effect', index] = [stats.ttest_1samp(stats_list[counter]['tsnr_difference_percent_acompcor_to_uncleaned_native'], popmean = 0),
                      cohen_d_within(stats_list[counter]['tsnr_difference_percent_acompcor_to_uncleaned_native'])]
        results_df.loc['Percent RETROICOR Effect', index] = [stats.ttest_1samp(stats_list[counter]['tsnr_difference_percent_retro_to_uncleaned_native'], popmean = 0),
                      cohen_d_within(stats_list[counter]['tsnr_difference_percent_retro_to_uncleaned_native'])]
        results_df.loc['Percent RETROICOR Effect vs AROMA', index] = [stats.ttest_1samp(stats_list[counter]['tsnr_difference_percent_unique_retro_to_aroma_vs_uncleaned_native'], popmean = 0),
                       cohen_d_within(stats_list[counter]['tsnr_difference_percent_unique_retro_to_aroma_vs_uncleaned_native'])]
        results_df.loc['Percent RETROICOR Effect vs AROMA and aCompCor', index] = [stats.ttest_1samp(stats_list[counter]['tsnr_difference_percent_unique_retro_to_aroma_acompcor_vs_uncleaned_native'], popmean = 0),
                      cohen_d_within(stats_list[counter]['tsnr_difference_percent_unique_retro_to_aroma_acompcor_vs_uncleaned_native'])]


results_df.to_csv(BASEPATH + 'stats_results.txt', sep = ' ')

# Add Percent transform
# Overall/Within mask
# Z change / Summe oder Average
