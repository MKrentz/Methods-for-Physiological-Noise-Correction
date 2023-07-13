#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 12:52:40 2021
@author: markre
"""

import scipy
import seaborn as sns
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
import glob

BASEPATH = '/project/3013068.03/physio_revision/TSNR_approach/'

part_list = glob.glob(BASEPATH + 'sub-*')
part_list.sort()
part_list.remove('/project/3013068.03/physio_revision/TSNR_approach/sub-008')

mean_MNI = pd.read_csv(BASEPATH + 'MNI_means.txt', index_col=0)
mean_brainstem = pd.read_csv(BASEPATH + 'brainstem_means.txt', index_col=0)
mean_gm = pd.read_csv(BASEPATH + 'graymatter_means.txt', index_col=0)
mean_LC = pd.read_csv(BASEPATH + 'LC_means.txt', index_col=0)

corr_list = [scipy.stats.pearsonr(mean_MNI['tsnr_difference_percent_aroma_acompcor_to_uncleaned_MNI'].astype(float),
                                mean_MNI['tsnr_difference_percent_unique_retro_to_aroma_acompcor_MNI'].astype(float)),
             scipy.stats.pearsonr(mean_gm['tsnr_difference_percent_aroma_acompcor_to_uncleaned_native'].astype(float),
                                mean_gm['tsnr_difference_percent_unique_retro_to_aroma_acompcor_native'].astype(float)),
             scipy.stats.pearsonr(mean_brainstem['tsnr_difference_percent_aroma_acompcor_to_uncleaned_MNI'].astype(float),
                                mean_brainstem['tsnr_difference_percent_unique_retro_to_aroma_acompcor_MNI'].astype(float)),
             scipy.stats.pearsonr(mean_LC['tsnr_difference_percent_aroma_acompcor_to_uncleaned_native'].astype(float),
                                mean_LC['tsnr_difference_percent_unique_retro_to_aroma_acompcor_native'].astype(float))]

titles = ['Correlation MNI', 'Correlation Gray Matter', 'Correlation Brainstem', 'Correlation LC']

x_temp = 'Percent TSNR improvement from AROMA and aCompCor'
y_temp = 'Unique Percent TSNR improvement from RETROICOR'
testlist = [mean_MNI, mean_gm, mean_brainstem, mean_LC]

for counter, x in enumerate(testlist):
    if counter == 0 or counter == 2:
        ax = sns.regplot(x=x['tsnr_difference_percent_aroma_acompcor_to_uncleaned_MNI'].astype(float),
                     y=x['tsnr_difference_percent_unique_retro_to_aroma_acompcor_vs_uncleaned_MNI'].astype(float),
                     line_kws={'label': f"r = {corr_list[counter][0]: .3f}, p = {corr_list[counter][1]: .3f}"})
        ax.set_xlabel(x_temp)
        ax.set_ylabel(y_temp)
        ax.set_title(titles[counter])
        ax.legend()
        plt.savefig(f'{BASEPATH}{titles[counter]}_RETRO')
        plt.close()
    elif counter == 1 or counter == 3:
        ax = sns.regplot(x=x['tsnr_difference_percent_aroma_acompcor_to_uncleaned_native'].astype(float),
                     y=x['tsnr_difference_percent_unique_retro_to_aroma_acompcor_vs_uncleaned_native'].astype(float),
                     line_kws={'label': f"r = {corr_list[counter][0]: .3f}, p = {corr_list[counter][1]: .3f}"})
        ax.set_xlabel(x_temp)
        ax.set_ylabel(y_temp)
        ax.set_title(titles[counter])
        ax.legend()
        plt.savefig(f'{BASEPATH}{titles[counter]}_RETRO')
        plt.close()

#RVT addition
corr_list = [scipy.stats.pearsonr(mean_MNI['tsnr_difference_percent_aroma_acompcor_to_uncleaned_MNI'].astype(float),
                                mean_MNI['tsnr_difference_percent_unique_retro_hr_rvt_to_aroma_acompcor_MNI'].astype(float)),
             scipy.stats.pearsonr(mean_gm['tsnr_difference_percent_aroma_acompcor_to_uncleaned_native'].astype(float),
                                mean_gm['tsnr_difference_percent_unique_retro_hr_rvt_to_aroma_acompcor_native'].astype(float)),
             scipy.stats.pearsonr(mean_brainstem['tsnr_difference_percent_aroma_acompcor_to_uncleaned_MNI'].astype(float),
                                mean_brainstem['tsnr_difference_percent_unique_retro_hr_rvt_to_aroma_acompcor_MNI'].astype(float)),
             scipy.stats.pearsonr(mean_LC['tsnr_difference_percent_aroma_acompcor_to_uncleaned_native'].astype(float),
                                mean_LC['tsnr_difference_percent_unique_retro_hr_rvt_to_aroma_acompcor_native'].astype(float))]

titles = ['Correlation MNI', 'Correlation Gray Matter', 'Correlation Brainstem', 'Correlation LC']

x_temp = 'Percent TSNR improvement from AROMA and aCompCor'
y_temp = 'Unique Percent TSNR improvement from RETROICOR+HR/RVT'
testlist = [mean_MNI, mean_gm, mean_brainstem, mean_LC]

for counter, x in enumerate(testlist):
    if counter == 0 or counter == 2:
        ax = sns.regplot(x=x['tsnr_difference_percent_aroma_acompcor_to_uncleaned_MNI'].astype(float),
                     y=x['tsnr_difference_percent_unique_retro_hr_rvt_to_aroma_acompcor_MNI'].astype(float),
                     line_kws={'label': f"r = {corr_list[counter][0]: .3f}, p = {corr_list[counter][1]: .3f}"})
        ax.set_xlabel(x_temp)
        ax.set_ylabel(y_temp)
        ax.set_title(titles[counter])
        ax.legend()
        plt.savefig(f'{BASEPATH}{titles[counter]}_RETRO_HR_RVT')
        plt.close()
    elif counter == 1 or counter == 3:
        ax = sns.regplot(x=x['tsnr_difference_percent_aroma_acompcor_to_uncleaned_native'].astype(float),
                     y=x['tsnr_difference_percent_unique_retro_hr_rvt_to_aroma_acompcor_native'].astype(float),
                     line_kws={'label': f"r = {corr_list[counter][0]: .3f}, p = {corr_list[counter][1]: .3f}"})
        ax.set_xlabel(x_temp)
        ax.set_ylabel(y_temp)
        ax.set_title(titles[counter])
        ax.legend()
        plt.savefig(f'{BASEPATH}{titles[counter]}_RETRO_HR_RVT')
        plt.close()
