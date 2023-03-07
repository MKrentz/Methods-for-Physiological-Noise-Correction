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

format_matrix = pd.DataFrame(columns=['MNI', 'GM', 'Brainstem', 'LC'])

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
        mean_MNI.loc[sub_id, data_type] = nib.load(glob.glob(BASEPATH + sub_id + '/glms/' + data_type + '*')[0]).get_fdata()[:, :, :,
                        np.newaxis]

        mean_brainstem.loc[sub_id, data_type] = np.ma.array(nib.load(glob.glob(BASEPATH + sub_id + '/glms/' + data_type + '*')[0]).get_fdata(), mask=brainstem_mat)

    for data_type, values in mean_native.items():
        mean_gm.loc[sub_id, data_type] = np.ma.array(nib.load(glob.glob(BASEPATH + sub_id + '/glms/' + data_type + '*')[0]).get_fdata(), mask=gm_mask_mat)
        if LC_mask != None:
                mean_LC.loc[sub_id, data_type] = np.ma.array(nib.load(glob.glob(BASEPATH + sub_id + '/glms/' + data_type + '*')[0]).get_fdata(), mask=LC_mask_mat)


mean_MNI.to_csv(BASEPATH + 'MNI_raw.txt')
mean_brainstem.to_csv(BASEPATH + 'brainstem_raw.txt')
mean_gm.to_csv(BASEPATH + 'graymatter_raw.txt')
mean_LC.to_csv(BASEPATH + 'LC_raw.txt')

names_correlation = ['corr_aroma', 'corr_retro', 'corr_diff']
arrays = [[sub[-7:] for sub in part_list], ['MNI', 'GM', 'Brainstem', 'LC']]
index_space = pd.MultiIndex.from_product(arrays, names=['sub_id', 'space'])
mean_correlation = pd.DataFrame(columns=names_correlation, index = index_space)

for subject_path in part_list:
    sub_id = subject_path[-7:]
    for data_type in mean_correlation.columns:
        if data_type == 'corr_aroma':
            mean_correlation.loc[sub_id, 'GM']['corr_aroma'] = np.corrcoef(
                mean_gm['tsnr_difference_percent_aroma_to_uncleaned_native'].loc[sub_id][
                    mean_gm['tsnr_difference_percent_aroma_to_uncleaned_native'].loc[sub_id].mask == False],
                mean_gm['tsnr_difference_percent_unique_aroma_to_retro_vs_uncleaned_native'].loc[sub_id][
                    mean_gm['tsnr_difference_percent_unique_aroma_to_retro_vs_uncleaned_native'].loc[sub_id].mask == False])[0,1]

            mean_correlation.loc[sub_id, 'LC']['corr_aroma'] = np.corrcoef(
                mean_LC['tsnr_difference_percent_aroma_to_uncleaned_native'].loc[sub_id][
                    mean_LC['tsnr_difference_percent_aroma_to_uncleaned_native'].loc[sub_id].mask == False],
                mean_LC['tsnr_difference_percent_unique_aroma_to_retro_vs_uncleaned_native'].loc[sub_id][
                    mean_LC['tsnr_difference_percent_unique_aroma_to_retro_vs_uncleaned_native'].loc[sub_id].mask == False])[0,1]

            mean_correlation.loc[sub_id, 'MNI']['corr_aroma'] = np.corrcoef(
                mean_MNI['tsnr_difference_percent_aroma_to_uncleaned_MNI'].loc[sub_id].flatten(),
                mean_MNI['tsnr_difference_percent_unique_aroma_to_retro_vs_uncleaned_MNI'].loc[sub_id].flatten())[0,1]

            mean_correlation.loc[sub_id, 'Brainstem']['corr_aroma'] = np.corrcoef(
                mean_brainstem['tsnr_difference_percent_aroma_to_uncleaned_MNI'].loc[sub_id][
                    mean_brainstem['tsnr_difference_percent_aroma_to_uncleaned_MNI'].loc[sub_id].mask == False],
                mean_brainstem['tsnr_difference_percent_unique_aroma_to_retro_vs_uncleaned_MNI'].loc[sub_id][
                    mean_brainstem['tsnr_difference_percent_unique_aroma_to_retro_vs_uncleaned_MNI'].loc[sub_id].mask == False])[0,1]

        elif data_type == 'corr_retro':
            mean_correlation.loc[sub_id, 'GM']['corr_retro'] = np.corrcoef(
                mean_gm['tsnr_difference_percent_retro_to_uncleaned_native'].loc[sub_id][
                    mean_gm['tsnr_difference_percent_retro_to_uncleaned_native'].loc[sub_id].mask == False],
                mean_gm['tsnr_difference_percent_unique_retro_to_aroma_vs_uncleaned_native'].loc[sub_id][
                    mean_gm['tsnr_difference_percent_unique_retro_to_aroma_vs_uncleaned_native'].loc[sub_id].mask == False])[0, 1]

            mean_correlation.loc[sub_id, 'LC']['corr_retro'] = np.corrcoef(
                mean_LC['tsnr_difference_percent_retro_to_uncleaned_native'].loc[sub_id][
                    mean_LC['tsnr_difference_percent_retro_to_uncleaned_native'].loc[sub_id].mask == False],
                mean_LC['tsnr_difference_percent_unique_retro_to_aroma_vs_uncleaned_native'].loc[sub_id][
                    mean_LC['tsnr_difference_percent_unique_retro_to_aroma_vs_uncleaned_native'].loc[sub_id].mask == False])[0, 1]

            mean_correlation.loc[sub_id, 'MNI']['corr_retro'] = np.corrcoef(
                mean_MNI['tsnr_difference_percent_retro_to_uncleaned_MNI'].loc[sub_id].flatten(),
                mean_MNI['tsnr_difference_percent_unique_retro_to_aroma_vs_uncleaned_MNI'].loc[sub_id].flatten())[0, 1]

            mean_correlation.loc[sub_id, 'Brainstem']['corr_retro'] = np.corrcoef(
                mean_brainstem['tsnr_difference_percent_retro_to_uncleaned_MNI'].loc[sub_id][
                    mean_brainstem['tsnr_difference_percent_retro_to_uncleaned_MNI'].loc[sub_id].mask == False],
                mean_brainstem['tsnr_difference_percent_unique_retro_to_aroma_vs_uncleaned_MNI'].loc[sub_id][
                    mean_brainstem['tsnr_difference_percent_unique_retro_to_aroma_vs_uncleaned_MNI'].loc[sub_id].mask == False])[0, 1]

        elif data_type == 'corr_diff':
            mean_correlation.loc[sub_id, 'GM']['corr_diff'] = np.corrcoef(
                mean_gm['tsnr_difference_percent_aroma_to_uncleaned_native'].loc[sub_id][
                    mean_gm['tsnr_difference_percent_aroma_to_uncleaned_native'].loc[sub_id].mask == False],
                mean_gm['tsnr_difference_percent_unique_retro_to_aroma_vs_uncleaned_native'].loc[sub_id][
                    mean_gm['tsnr_difference_percent_unique_retro_to_aroma_vs_uncleaned_native'].loc[
                        sub_id].mask == False])[0, 1]

            mean_correlation.loc[sub_id, 'LC']['corr_diff'] = np.corrcoef(
                mean_LC['tsnr_difference_percent_aroma_to_uncleaned_native'].loc[sub_id][
                    mean_LC['tsnr_difference_percent_aroma_to_uncleaned_native'].loc[sub_id].mask == False],
                mean_LC['tsnr_difference_percent_unique_retro_to_aroma_vs_uncleaned_native'].loc[sub_id][
                    mean_LC['tsnr_difference_percent_unique_retro_to_aroma_vs_uncleaned_native'].loc[
                        sub_id].mask == False])[0, 1]

            mean_correlation.loc[sub_id, 'MNI']['corr_diff'] = np.corrcoef(
                mean_MNI['tsnr_difference_percent_aroma_to_uncleaned_MNI'].loc[sub_id].flatten(),
                mean_MNI['tsnr_difference_percent_unique_retro_to_aroma_vs_uncleaned_MNI'].loc[sub_id].flatten())[0, 1]

            mean_correlation.loc[sub_id, 'Brainstem']['corr_diff'] = np.corrcoef(
                mean_brainstem['tsnr_difference_percent_aroma_to_uncleaned_MNI'].loc[sub_id][
                    mean_brainstem['tsnr_difference_percent_aroma_to_uncleaned_MNI'].loc[sub_id].mask == False],
                mean_brainstem['tsnr_difference_percent_unique_retro_to_aroma_vs_uncleaned_MNI'].loc[sub_id][
                    mean_brainstem['tsnr_difference_percent_unique_retro_to_aroma_vs_uncleaned_MNI'].loc[
                        sub_id].mask == False])[0, 1]

data_dic = {'aroma_MNI': mean_correlation.loc[mean_correlation.index.get_level_values(1) == "MNI", 'corr_aroma'],
            'aroma_GM': mean_correlation.loc[mean_correlation.index.get_level_values(1) == "GM", 'corr_aroma'],
            'aroma_Brainstem':mean_correlation.loc[mean_correlation.index.get_level_values(1) == "Brainstem", 'corr_aroma'],
            'aroma_LC': mean_correlation.loc[mean_correlation.index.get_level_values(1) == "LC", 'corr_aroma'],
            'retro_MNI': mean_correlation.loc[mean_correlation.index.get_level_values(1) == "MNI", 'corr_retro'],
            'retro_GM': mean_correlation.loc[mean_correlation.index.get_level_values(1) == "GM", 'corr_retro'],
            'retro_Brainstem': mean_correlation.loc[mean_correlation.index.get_level_values(1) == "Brainstem", 'corr_retro'],
            'retro_LC': mean_correlation.loc[mean_correlation.index.get_level_values(1) == "LC", 'corr_retro'],
            'diff_MNI': mean_correlation.loc[mean_correlation.index.get_level_values(1) == "MNI", 'corr_diff'],
            'diff_GM': mean_correlation.loc[mean_correlation.index.get_level_values(1) == "GM", 'corr_diff'],
            'diff_Brainstem': mean_correlation.loc[
                mean_correlation.index.get_level_values(1) == "Brainstem", 'corr_diff'],
            'diff_LC': mean_correlation.loc[mean_correlation.index.get_level_values(1) == "LC", 'corr_diff'],
            }

import scipy
import seaborn as sns

corr_list = [scipy.stats.pearsonr(mean_MNI['tsnr_difference_percent_aroma_to_uncleaned_MNI'].astype(float),
                                mean_MNI['tsnr_difference_percent_unique_retro_to_aroma_MNI'].astype(float)),
             scipy.stats.pearsonr(mean_gm['tsnr_difference_percent_aroma_to_uncleaned_native'].astype(float),
                                mean_gm['tsnr_difference_percent_unique_retro_to_aroma_native'].astype(float)),
             scipy.stats.pearsonr(mean_brainstem['tsnr_difference_percent_aroma_to_uncleaned_MNI'].astype(float),
                                mean_brainstem['tsnr_difference_percent_unique_retro_to_aroma_MNI'].astype(float)),
             scipy.stats.pearsonr(mean_LC['tsnr_difference_percent_aroma_to_uncleaned_native'].astype(float),
                                mean_LC['tsnr_difference_percent_unique_retro_to_aroma_native'].astype(float))]

titles = ['Correlation MNI', 'Correlation Gray Matter', 'Correlation Brainstem', 'Correlation LC']

x_temp = 'Percent TSNR improvement from AROMA'
y_temp = 'Unique Percent TSNR improvement from RETROICOR'
testlist = [mean_MNI, mean_gm, mean_brainstem, mean_LC]

for counter, x in enumerate(testlist):
    if counter == 0 or counter == 2:
        ax = sns.regplot(x = x['tsnr_difference_percent_aroma_to_uncleaned_MNI'].astype(float),
                     y = x['tsnr_difference_percent_unique_retro_to_aroma_MNI'].astype(float),
                     line_kws={'label': f"r = {corr_list[counter][0]: .3f}, p = {corr_list[counter][1]: .3f}"})
        ax.set_xlabel(x_temp)
        ax.set_ylabel(y_temp)
        ax.set_title(titles[counter])
        ax.legend()
        plt.savefig(f'/project/3013068.03/RETROICOR/{titles[counter]}')
        plt.close()
    elif counter == 1 or counter == 3:
        ax = sns.regplot(x = x['tsnr_difference_percent_aroma_to_uncleaned_native'].astype(float),
                     y = x['tsnr_difference_percent_unique_retro_to_aroma_native'].astype(float),
                     line_kws={'label': f"r = {corr_list[counter][0]: .3f}, p = {corr_list[counter][1]: .3f}"})
        ax.set_xlabel(x_temp)
        ax.set_ylabel(y_temp)
        ax.set_title(titles[counter])
        ax.legend()
        plt.savefig(f'/project/3013068.03/RETROICOR/{titles[counter]}')
        plt.close()