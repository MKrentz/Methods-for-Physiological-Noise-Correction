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
from Subject_Class_new import Subject
from nilearn.datasets import load_mni152_brain_mask
import pandas as pd
from numpy import mean, std
from math import sqrt

BASEPATH = '/project/3013068.03/physio_revision/TSNR_approach/'

part_list = glob.glob(BASEPATH + 'sub-*')
part_list.sort()
part_list.remove('/project/3013068.03/physio_revision/TSNR_approach/sub-008')
part_list = part_list[:-2]

# Planned comparisons
var_names_MNI = ['tsnr_difference_aroma_to_uncleaned_MNI',
                 'tsnr_difference_retro_to_uncleaned_MNI',
                 'tsnr_difference_acompcor_to_uncleaned_MNI',
                 'tsnr_difference_aroma_acompcor_to_uncleaned_MNI',
                 'tsnr_difference_aroma_retro_to_uncleaned_MNI',
                 'tsnr_difference_unique_aroma_to_retro_MNI',
                 'tsnr_difference_unique_retro_to_aroma_MNI',
                 'tsnr_difference_unique_acompcor_to_aroma_MNI',
                 'tsnr_difference_unique_retro_to_aroma_acompcor_MNI',
                 'tsnr_difference_percent_unique_retro_to_aroma_MNI',
                 'tsnr_difference_percent_unique_aroma_to_retro_MNI',
                 'tsnr_difference_percent_unique_retro_to_aroma_acompcor_MNI',
                 'tsnr_difference_percent_unique_acompcor_to_aroma_MNI',
                 'tsnr_difference_percent_unique_retro_to_aroma_vs_uncleaned_MNI',
                 'tsnr_difference_percent_unique_aroma_to_retro_vs_uncleaned_MNI',
                 'tsnr_difference_percent_unique_acompcor_to_aroma_vs_uncleaned_MNI',
                 'tsnr_difference_percent_unique_retro_to_aroma_acompcor_vs_uncleaned_MNI',
                 'tsnr_difference_percent_retro_to_uncleaned_MNI',
                 'tsnr_difference_percent_aroma_to_uncleaned_MNI',
                 'tsnr_difference_percent_acompcor_to_uncleaned_MNI',
                 'tsnr_difference_hr_rvt_to_uncleaned_MNI',
                 'tsnr_difference_percent_hr_rvt_to_uncleaned_MNI',
                 'tsnr_difference_retro_hr_rvt_to_uncleaned_MNI',
                 'tsnr_difference_percent_unique_retro_hr_rvt_to_aroma_MNI',
                 'tsnr_difference_percent_unique_retro_hr_rvt_to_aroma_acompcor_MNI',
                 'tsnr_difference_percent_retro_hr_rvt_to_uncleaned_MNI',
                 'tsnr_difference_percent_aroma_acompcor_to_uncleaned_MNI']

var_names_native = ['tsnr_difference_aroma_to_uncleaned_native',
                    'tsnr_difference_retro_to_uncleaned_native',
                    'tsnr_difference_acompcor_to_uncleaned_native',
                    'tsnr_difference_aroma_acompcor_to_uncleaned_native',
                    'tsnr_difference_aroma_retro_to_uncleaned_native',
                    'tsnr_difference_unique_aroma_to_retro_native',
                    'tsnr_difference_unique_retro_to_aroma_native',
                    'tsnr_difference_unique_acompcor_to_aroma_native',
                    'tsnr_difference_unique_retro_to_aroma_acompcor_native',
                    'tsnr_difference_percent_unique_retro_to_aroma_native',
                    'tsnr_difference_percent_unique_aroma_to_retro_native',
                    'tsnr_difference_percent_unique_retro_to_aroma_acompcor_native',
                    'tsnr_difference_percent_unique_acompcor_to_aroma_native',
                    'tsnr_difference_percent_unique_retro_to_aroma_vs_uncleaned_native',
                    'tsnr_difference_percent_unique_aroma_to_retro_vs_uncleaned_native',
                    'tsnr_difference_percent_unique_acompcor_to_aroma_vs_uncleaned_native',
                    'tsnr_difference_percent_unique_retro_to_aroma_acompcor_vs_uncleaned_native',
                    'tsnr_difference_percent_retro_to_uncleaned_native',
                    'tsnr_difference_percent_aroma_to_uncleaned_native',
                    'tsnr_difference_percent_acompcor_to_uncleaned_native',
                    'tsnr_difference_hr_rvt_to_uncleaned_native',
                    'tsnr_difference_percent_hr_rvt_to_uncleaned_native',
                    'tsnr_difference_retro_hr_rvt_to_uncleaned_native',
                    'tsnr_difference_percent_unique_retro_hr_rvt_to_aroma_native',
                    'tsnr_difference_percent_unique_retro_hr_rvt_to_aroma_acompcor_native',
                    'tsnr_difference_percent_retro_hr_rvt_to_uncleaned_native',
                    'tsnr_difference_percent_aroma_acompcor_to_uncleaned_native']

# Create object dictionary
mean_MNI = pd.DataFrame(index=[sub[-7:] for sub in part_list], columns=var_names_MNI)
mean_brainstem = pd.DataFrame(index=[sub[-7:] for sub in part_list], columns=var_names_MNI)
mean_native = pd.DataFrame(index=[sub[-7:] for sub in part_list], columns=var_names_native)
mean_gm = pd.DataFrame(index=[sub[-7:] for sub in part_list], columns=var_names_native)
mean_LC = pd.DataFrame(index=[sub[-7:] for sub in part_list], columns=var_names_native)
stress_list = ['sub-002', 'sub-003', 'sub-004', 'sub-007', 'sub-009', 'sub-013', 'sub-015', 'sub-017', 'sub-021',
               'sub-023', 'sub-025', 'sub-027', 'sub-029']

for subject_path in part_list:
    sub_id = subject_path[-7:]
    sub = Subject(sub_id)

    ses_nr = 2 if sub_id in stress_list else 1

    # Individual LC Mask
    try:
        LC_mask = nib.load(BASEPATH + '{0}/masks/LC_mask_native.nii.gz'.format(sub_id))
        LC_mask_mat = LC_mask.get_fdata()
        print('There is an LC Mask!')

    except:
        LC_mask = None
        print('NO MASK!!')

    # Load in, resample and binarise GM mask
    mni_mat = sub.get_brainmask(MNI=True, session=ses_nr, run=2).get_fdata()
    mni_mat = np.where((mni_mat == 0) | (mni_mat == 1), 1 - mni_mat, mni_mat)
    gm_mask_mat = nib.load(BASEPATH + '{0}/masks/gm_mask_native.nii.gz'.format(sub_id)).get_fdata()
    brainstem_mat = nib.load(BASEPATH + '{0}/masks/brainstem_mask_MNI.nii.gz'.format(sub_id)).get_fdata()

    # Account for first subject difference
    for data_type in mean_MNI.columns:
        mean_MNI.loc[sub_id, data_type] = np.mean(nib.load(glob.glob(BASEPATH
                                                                     + sub_id
                                                                     + '/glms/'
                                                                     + data_type
                                                                     + '*')[0]).get_fdata()[:, :, :, np.newaxis])

        mean_brainstem.loc[sub_id, data_type] = np.mean(np.ma.array(nib.load(glob.glob(BASEPATH
                                                                                       + sub_id
                                                                                       + '/glms/'
                                                                                       + data_type
                                                                                       + '*')[0]).get_fdata(),
                                                                    mask=brainstem_mat))

    for data_type, values in mean_native.items():
        mean_gm.loc[sub_id, data_type] = np.mean(np.ma.array(nib.load(glob.glob(BASEPATH
                                                                                + sub_id
                                                                                + '/glms/'
                                                                                + data_type
                                                                                + '*')[0]).get_fdata(),
                                                             mask=gm_mask_mat))
        if LC_mask != None:
                mean_LC.loc[sub_id, data_type] = np.mean(np.ma.array(nib.load(glob.glob(BASEPATH
                                                                                        + sub_id
                                                                                        + '/glms/'
                                                                                        + data_type
                                                                                        + '*')[0]).get_fdata(),
                                                                     mask=LC_mask_mat))


mean_MNI.to_csv(BASEPATH + 'MNI_means.txt')
mean_brainstem.to_csv(BASEPATH + 'brainstem_means.txt')
mean_gm.to_csv(BASEPATH + 'graymatter_means.txt')
mean_LC.to_csv(BASEPATH + 'LC_means.txt')

stats_list = [mean_MNI.astype('float'), mean_brainstem.astype('float'), mean_gm.astype('float'), mean_LC.astype('float')]
space_name= ['MNI', 'Brainstem', 'GrayMatter', 'LC']


output_names = ['AROMA Cleaned',
                'RETROICOR Cleaned',
                'aCompCor Cleaned',
                'Unique aCompCor Effect to AROMA',
                'Unique RETROICOR Effect to AROMA',
                'Unique RETROICOR Effect to AROMA and aCompCor',
                'Percent RETROICOR Effect',
                'Percent AROMA Effect',
                'Percent aCompCor Effect',
                'Percent RETROICOR Effect vs AROMA',
                'Percent RETROICOR Effect vs AROMA and aCompCor',
                'HR/RVT Cleaned',
                'Percent HR/RVT Cleaned',
                'Percent RETROICOR and HR/RVT Cleaned'
                'Percent RETROICOR and HR/RVT Effect vs AROMA',
                'Percent RETROICOR and HR/RVT Effect vs AROMA and aCompCor'
                'Percent Aroma and aCompCor to Uncleaned']

results_df = pd.DataFrame(index=output_names, columns=space_name)

def cohen_d(x,y):
        return (mean(x) - mean(y)) / sqrt((std(x, ddof=1) ** 2 + std(y, ddof=1) ** 2) / 2.0)
def cohen_d_within(x):
    return mean(x) / std(x, ddof=1)

for counter, index in enumerate(results_df.columns):
    if index == 'MNI' or index == 'Brainstem':
        results_df.loc['AROMA Cleaned', index] = [stats.ttest_1samp(stats_list[counter]['tsnr_difference_aroma_to_uncleaned_MNI'], popmean=0),
                      cohen_d_within(stats_list[counter]['tsnr_difference_aroma_to_uncleaned_MNI'])]
        results_df.loc['RETROICOR Cleaned', index] = [stats.ttest_1samp(stats_list[counter]['tsnr_difference_retro_to_uncleaned_MNI'], popmean=0),
                      cohen_d_within(stats_list[counter]['tsnr_difference_retro_to_uncleaned_MNI'])]
        results_df.loc['aCompCor Cleaned', index] = [stats.ttest_1samp(stats_list[counter]['tsnr_difference_acompcor_to_uncleaned_MNI'], popmean=0),
                      cohen_d_within(stats_list[counter]['tsnr_difference_acompcor_to_uncleaned_MNI'])]
        results_df.loc['Unique aCompCor Effect to AROMA', index] = [stats.ttest_1samp(stats_list[counter]['tsnr_difference_unique_acompcor_to_aroma_MNI'], popmean=0),
                      cohen_d_within(stats_list[counter]['tsnr_difference_unique_acompcor_to_aroma_MNI'])]
        results_df.loc['Unique RETROICOR Effect to AROMA', index] = [stats.ttest_1samp(stats_list[counter]['tsnr_difference_unique_retro_to_aroma_MNI'], popmean=0),
                      cohen_d_within(stats_list[counter]['tsnr_difference_unique_retro_to_aroma_MNI'])]
        results_df.loc['Unique RETROICOR Effect to AROMA and aCompCor', index] = [stats.ttest_1samp(stats_list[counter]['tsnr_difference_unique_retro_to_aroma_acompcor_MNI'], popmean=0),
                      cohen_d_within(stats_list[counter]['tsnr_difference_unique_retro_to_aroma_acompcor_MNI'])]
        results_df.loc['Percent aCompCor Effect to AROMA', index] =  [stats.ttest_1samp(stats_list[counter]['tsnr_difference_percent_unique_acompcor_to_aroma_vs_uncleaned_MNI'], popmean=0),
                      cohen_d_within(stats_list[counter]['tsnr_difference_percent_unique_acompcor_to_aroma_vs_uncleaned_MNI'])]
        results_df.loc['Percent AROMA Effect', index] = [stats.ttest_1samp(stats_list[counter]['tsnr_difference_percent_aroma_to_uncleaned_MNI'], popmean=0),
                      cohen_d_within(stats_list[counter]['tsnr_difference_percent_aroma_to_uncleaned_MNI'])]
        results_df.loc['Percent aCompCor Effect', index] = [stats.ttest_1samp(stats_list[counter]['tsnr_difference_percent_acompcor_to_uncleaned_MNI'], popmean=0),
                      cohen_d_within(stats_list[counter]['tsnr_difference_percent_acompcor_to_uncleaned_MNI'])]
        results_df.loc['Percent RETROICOR Effect', index] = [stats.ttest_1samp(stats_list[counter]['tsnr_difference_percent_retro_to_uncleaned_MNI'], popmean=0),
                      cohen_d_within(stats_list[counter]['tsnr_difference_percent_retro_to_uncleaned_MNI'])]
        results_df.loc['Percent RETROICOR Effect vs AROMA', index] = [stats.ttest_1samp(stats_list[counter]['tsnr_difference_percent_unique_retro_to_aroma_vs_uncleaned_MNI'], popmean=0),
                       cohen_d_within(stats_list[counter]['tsnr_difference_percent_unique_retro_to_aroma_vs_uncleaned_MNI'])]
        results_df.loc['Percent RETROICOR Effect vs AROMA and aCompCor', index] = [stats.ttest_1samp(stats_list[counter]['tsnr_difference_percent_unique_retro_to_aroma_acompcor_vs_uncleaned_MNI'], popmean=0),
                      cohen_d_within(stats_list[counter]['tsnr_difference_percent_unique_retro_to_aroma_acompcor_vs_uncleaned_MNI'])]
        results_df.loc['HR/RVT Cleaned', index] = [stats.ttest_1samp(stats_list[counter]['tsnr_difference_hr_rvt_to_uncleaned_MNI'], popmean=0),
                      cohen_d_within(stats_list[counter]['tsnr_difference_hr_rvt_to_uncleaned_MNI'])]
        results_df.loc['Percent HR/RVT Effect', index] = [stats.ttest_1samp(stats_list[counter]['tsnr_difference_percent_hr_rvt_to_uncleaned_MNI'], popmean=0),
                      cohen_d_within(stats_list[counter]['tsnr_difference_percent_hr_rvt_to_uncleaned_MNI'])]
        results_df.loc['Percent RETROICOR and HR/RVT Cleaned', index] = [stats.ttest_1samp(stats_list[counter]['tsnr_difference_retro_hr_rvt_to_uncleaned_MNI'], popmean=0),
                      cohen_d_within(stats_list[counter]['tsnr_difference_retro_hr_rvt_to_uncleaned_MNI'])]
        results_df.loc['Percent RETROICOR and HR/RVT Effect vs AROMA', index] = [stats.ttest_1samp(stats_list[counter]['tsnr_difference_percent_unique_retro_hr_rvt_to_aroma_MNI'], popmean=0),
                      cohen_d_within(stats_list[counter]['tsnr_difference_percent_unique_retro_hr_rvt_to_aroma_MNI'])]
        results_df.loc['Percent RETROICOR and HR/RVT Effect vs AROMA and aCompCor', index] = [stats.ttest_1samp(stats_list[counter]['tsnr_difference_percent_unique_retro_hr_rvt_to_aroma_acompcor_MNI'], popmean=0),
                      cohen_d_within(stats_list[counter]['tsnr_difference_percent_unique_retro_hr_rvt_to_aroma_acompcor_MNI'])]
        results_df.loc['Percent RETROICOR and HR/RVT', index] = [stats.ttest_1samp(stats_list[counter]['tsnr_difference_percent_retro_hr_rvt_to_uncleaned_MNI'], popmean=0),
                      cohen_d_within(stats_list[counter]['tsnr_difference_percent_retro_hr_rvt_to_uncleaned_MNI'])]
        results_df.loc['Percent Aroma and aCompCor to Uncleaned', index] = [
            stats.ttest_1samp(stats_list[counter]['tsnr_difference_percent_retro_acomcpoto_uncleaned_MNI'], popmean=0),
            cohen_d_within(stats_list[counter]['tsnr_difference_percent_retro_acomcpoto_uncleaned_MNI'])]

    elif index == 'GrayMatter' or index == 'LC':
        results_df.loc['AROMA Cleaned', index] = [stats.ttest_1samp(stats_list[counter]['tsnr_difference_aroma_to_uncleaned_native'], popmean=0),
                      cohen_d_within(stats_list[counter]['tsnr_difference_aroma_to_uncleaned_native'])]
        results_df.loc['RETROICOR Cleaned', index] = [stats.ttest_1samp(stats_list[counter]['tsnr_difference_retro_to_uncleaned_native'], popmean=0),
                      cohen_d_within(stats_list[counter]['tsnr_difference_retro_to_uncleaned_native'])]
        results_df.loc['aCompCor Cleaned', index] = [stats.ttest_1samp(stats_list[counter]['tsnr_difference_acompcor_to_uncleaned_native'], popmean=0),
                      cohen_d_within(stats_list[counter]['tsnr_difference_acompcor_to_uncleaned_native'])]
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
        results_df.loc['HR/RVT Cleaned', index] = [stats.ttest_1samp(stats_list[counter]['tsnr_difference_hr_rvt_to_uncleaned_native'], popmean=0),
                      cohen_d_within(stats_list[counter]['tsnr_difference_hr_rvt_to_uncleaned_native'])]
        results_df.loc['Percent HR/RVT Effect', index] = [stats.ttest_1samp(stats_list[counter]['tsnr_difference_percent_hr_rvt_to_uncleaned_native'], popmean=0),
                      cohen_d_within(stats_list[counter]['tsnr_difference_percent_hr_rvt_to_uncleaned_native'])]
        results_df.loc['Percent RETROICOR and HR/RVT Cleaned', index] = [stats.ttest_1samp(stats_list[counter]['tsnr_difference_retro_hr_rvt_to_uncleaned_native'], popmean=0),
                      cohen_d_within(stats_list[counter]['tsnr_difference_retro_hr_rvt_to_uncleaned_native'])]
        results_df.loc['Percent RETROICOR and HR/RVT Effect vs AROMA', index] = [stats.ttest_1samp(stats_list[counter]['tsnr_difference_percent_unique_retro_hr_rvt_to_aroma_native'], popmean=0),
                      cohen_d_within(stats_list[counter]['tsnr_difference_percent_unique_retro_hr_rvt_to_aroma_native'])]
        results_df.loc['Percent RETROICOR and HR/RVT Effect vs AROMA and aCompCor', index] = [stats.ttest_1samp(stats_list[counter]['tsnr_difference_percent_unique_retro_hr_rvt_to_aroma_acompcor_native'], popmean=0),
                      cohen_d_within(stats_list[counter]['tsnr_difference_percent_unique_retro_hr_rvt_to_aroma_acompcor_native'])]
        results_df.loc['Percent RETROICOR and HR/RVT', index] = [
            stats.ttest_1samp(stats_list[counter]['tsnr_difference_percent_retro_hr_rvt_to_uncleaned_native'], popmean=0),
            cohen_d_within(stats_list[counter]['tsnr_difference_percent_retro_hr_rvt_to_uncleaned_native'])]
        results_df.loc['Percent Aroma and aCompCor to Uncleaned', index] = [
            stats.ttest_1samp(stats_list[counter]['tsnr_difference_percent_retro_acomcpoto_uncleaned_native'], popmean=0),
            cohen_d_within(stats_list[counter]['tsnr_difference_percent_retro_acomcpoto_uncleaned_native'])]
results_df.to_csv(BASEPATH + 'stats_results.txt', sep=' ')

