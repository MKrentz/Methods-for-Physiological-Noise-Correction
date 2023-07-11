#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Fri Aug 13 11:16:15 2021
This script is used to create graphs for the TSNR main effects as well as difference effects as calculated.

@author: markre
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import glob



BASEPATH = '/project/3013068.03/physio_revision/TSNR_approach/'

output_names = ['RETROICOR Cleaned', 'AROMA Cleaned', 'aCompCor Cleaned', 'Unique aCompCor Effect to AROMA', 'Unique RETROICOR Effect to AROMA', 'Unique RETROICOR Effect to AROMA and aCompCor',
                'Percent RETROICOR Effect', 'Percent AROMA Effect', 'Percent aCompCor Effect', 'Percent RETROICOR Effect vs AROMA', 'Percent RETROICOR Effect vs AROMA and aCompCor']


#Create a dataframe calculating mean and confidence intervals for the full-brain TSNR map
mean_MNI_value_df = pd.read_csv(BASEPATH + 'MNI_means.txt', index_col=0)
new_column_names = [None]*len(mean_MNI_value_df.columns)
for counter, columns in enumerate(mean_MNI_value_df.columns):
    new_column_names[counter] = columns.replace('_MNI', '')
mean_MNI_value_df.columns = new_column_names

MNI_plotting_df = pd.DataFrame(columns=mean_MNI_value_df.columns, index=['Mean', 'Confidence Interval'])
for column_counter, column in enumerate(mean_MNI_value_df.columns):
    mean, sigma = np.mean(mean_MNI_value_df[column]), np.std(mean_MNI_value_df[column])
    confidence_interval = stats.norm.interval(0.95, loc=mean, scale=sigma/np.sqrt(len(mean_MNI_value_df[column])))
    confidence_interval = confidence_interval[1] - mean
    MNI_plotting_df[column][0], MNI_plotting_df[column][1] = mean, confidence_interval

#Create a dataframe calculating mean and confidence intervals for the cortex gray-matter TSNR map
mean_gm_value_df = pd.read_csv(BASEPATH + 'graymatter_means.txt', index_col=0)
mean_gm_value_df.columns = new_column_names
gm_plotting_df = pd.DataFrame(columns=mean_gm_value_df.columns, index=['Mean', 'Confidence Interval'])
for column_counter, column in enumerate(mean_gm_value_df.columns):
    mean, sigma = np.mean(mean_gm_value_df[column]), np.std(mean_gm_value_df[column])
    confidence_interval = stats.norm.interval(0.95, loc=mean, scale=sigma/np.sqrt(len(mean_gm_value_df[column])))
    confidence_interval = confidence_interval[1] - mean
    gm_plotting_df[column][0], gm_plotting_df[column][1] = mean, confidence_interval

#Create a dataframe calculating mean and confidence intervals for the brainstem TSNR map
mean_brainstem_value_df = pd.read_csv(BASEPATH + 'brainstem_means.txt', index_col = 0)
mean_brainstem_value_df.columns = new_column_names
brainstem_plotting_df = pd.DataFrame(columns = mean_brainstem_value_df.columns, index=['Mean', 'Confidence Interval'])
for column_counter, column in enumerate(mean_brainstem_value_df.columns):
    mean, sigma = np.mean(mean_brainstem_value_df[column]), np.std(mean_brainstem_value_df[column])
    confidence_interval = stats.norm.interval(0.95, loc=mean, scale=sigma/np.sqrt(len(mean_brainstem_value_df[column])))
    confidence_interval = confidence_interval[1] - mean
    brainstem_plotting_df[column][0], brainstem_plotting_df[column][1] = mean, confidence_interval

#Create a dataframe calculating mean and confidence intervals for the LC TSNR map
mean_LC_value_df = pd.read_csv(BASEPATH + 'LC_means.txt', index_col = 0)
mean_LC_value_df.columns = new_column_names
LC_plotting_df = pd.DataFrame(columns = mean_LC_value_df.columns, index = ['Mean', 'Confidence Interval'])
for column_counter, column in enumerate(mean_LC_value_df.columns):
    mean, sigma = np.mean(mean_LC_value_df[column]), np.std(mean_LC_value_df[column])
    confidence_interval = stats.norm.interval(0.95, loc=mean, scale=sigma/np.sqrt(len(mean_LC_value_df[column])))
    confidence_interval = confidence_interval[1] - mean
    LC_plotting_df[column][0], LC_plotting_df[column][1] = mean, confidence_interval

df_dic = ({'Whole Brain': MNI_plotting_df, 'Gray Matter': gm_plotting_df, 'Brainstem': brainstem_plotting_df, 'LC': LC_plotting_df})


#============================================
#      ===== MIXED GRAPH =====
#============================================


x_pos = [[2, 3, 4], [6, 7, 8], [10, 11, 12], [14, 15, 16]]
group_pos = [3, 7, 11, 15]
fig = plt.figure()
colors = ['dimgray', 'silver', 'whitesmoke']
bar_width = 1
group_main_effect = ['Whole Brain', 'Grey Matter', 'Brainstem', 'LC']
bars2_difference_effect = ['RETROICOR', 'AROMA', 'aCompCor']
plt.ylim(0, 100)

retro_patch = mpatches.Patch(color='dimgray', label='RETROICOR')
aroma_patch = mpatches.Patch(color='silver', label='AROMA')
acompcor_patch = mpatches.Patch(color='whitesmoke', label='aCompCor')

plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.subplots_adjust(bottom=0.23)
plt.xticks(group_pos, group_main_effect, rotation=-45)
plt.title('% TSNR Improvement Compared to Uncleaned', size=11, y=1.1)
plt.legend(loc='upper left', handles=[retro_patch, aroma_patch, acompcor_patch])


for counter, (keys, values) in enumerate(df_dic.items()):
    #ax1 = fig.add_subplot()
    bars =plt.bar(x_pos[counter],
                   height=[values.loc['Mean']['tsnr_difference_percent_retro_to_uncleaned'], values.loc['Mean']['tsnr_difference_percent_aroma_to_uncleaned'],
                           values.loc['Mean']['tsnr_difference_percent_acompcor_to_uncleaned']],
                   yerr=[values.loc['Confidence Interval']['tsnr_difference_percent_retro_to_uncleaned'],
                         values.loc['Confidence Interval']['tsnr_difference_percent_aroma_to_uncleaned'],
                         values.loc['Confidence Interval']['tsnr_difference_percent_acompcor_to_uncleaned']],
                   width=bar_width,
                   capsize=5,
                   edgecolor='black',
                   color=colors)
    
plt.savefig(BASEPATH + 'main_effects.svg', dpi=1200)


x_pos = [[2, 3, 4], [6, 7, 8], [10, 11, 12], [14, 15, 16]]
group_pos = [3, 7, 11, 15]
fig = plt.figure()
colors = ['dimgray', 'silver', 'whitesmoke']
bar_width = 1
group_main_effect = ['Whole Brain', 'Grey Matter', 'Brainstem', 'LC']
plt.ylim(0, 35)
retro_patch = mpatches.Patch(color='dimgray', label='RETROICOR vs. Uncleaned')
aroma_patch = mpatches.Patch(color='silver', label='RETROICOR vs. AROMA')
acompcor_patch = mpatches.Patch(color='whitesmoke', label='RETROICOR vs. AROMA + aCompCor')

plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.subplots_adjust(bottom=0.23)
plt.xticks(group_pos, group_main_effect, rotation=-45)
plt.title('Unique RETROICOR effect in % TSNR improvement', size=11, y=1.1)
plt.legend(loc='upper left', handles=[retro_patch, aroma_patch, acompcor_patch])

for counter, (keys, values) in enumerate(df_dic.items()):
    bars = plt.bar(x_pos[counter],
                  height = [values.loc['Mean']['tsnr_difference_percent_retro_to_uncleaned'], values.loc['Mean']['tsnr_difference_percent_unique_retro_to_aroma'],
                            values.loc['Mean']['tsnr_difference_percent_unique_retro_to_aroma_acompcor']],
                  yerr = [values.loc['Confidence Interval']['tsnr_difference_percent_retro_to_uncleaned'], values.loc['Confidence Interval']['tsnr_difference_percent_unique_retro_to_aroma'],
                            values.loc['Confidence Interval']['tsnr_difference_percent_unique_retro_to_aroma_acompcor']],
                  width = bar_width,
                  capsize = 5,
                  edgecolor = 'black',
                  color = colors)
    

    plt.savefig(BASEPATH + 'unique_retro_effect.svg', dpi = 1200)


# New FIGURE
x_pos = [[2, 3, 4], [6, 7, 8], [10, 11, 12], [14, 15, 16]]
group_pos = [3, 7, 11, 15]
fig = plt.figure()
colors = ['dimgray', 'silver', 'whitesmoke']
bar_width = 1
group_main_effect = ['Whole Brain', 'Grey Matter', 'Brainstem', 'LC']
plt.ylim(0, 35)
retro_patch = mpatches.Patch(color='dimgray', label='RETROICOR + HR/RVT vs. Uncleaned')
aroma_patch = mpatches.Patch(color='silver', label='RETROICOR + HR/RVT vs. AROMA')
acompcor_patch = mpatches.Patch(color='whitesmoke', label='RETROICOR + HR/RVT vs. AROMA + aCompCor')

plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.subplots_adjust(bottom=0.23)
plt.xticks(group_pos, group_main_effect, rotation=-45)
plt.title('Unique RETROICOR+HR/RVT effect in % TSNR improvement', size=11, y=1.1)
plt.legend(loc='upper left', handles=[retro_patch, aroma_patch, acompcor_patch])

for counter, (keys, values) in enumerate(df_dic.items()):
    bars = plt.bar(x_pos[counter],
                   height=[values.loc['Mean']['tsnr_difference_percent_hr_rvt_to_uncleaned'],
                           values.loc['Mean']['tsnr_difference_percent_unique_retro_hr_rvt_to_aroma'],
                           values.loc['Mean']['tsnr_difference_percent_unique_retro_hr_rvt_to_aroma_acompcor']],
                   yerr=[values.loc['Confidence Interval']['tsnr_difference_percent_hr_rvt_to_uncleaned'],
                         values.loc['Confidence Interval']['tsnr_difference_percent_unique_retro_hr_rvt_to_aroma'],
                         values.loc['Confidence Interval']['tsnr_difference_percent_unique_retro_hr_rvt_to_aroma_acompcor']],
                   width=bar_width,
                   capsize=5,
                   edgecolor='black',
                   color=colors)

    plt.savefig(BASEPATH + 'unique_retro_hr_rvt_effect.svg', dpi=1200)

# New FIGURE Retro vs RETRO HR RVT
x_pos = [[2, 3], [5, 6], [8, 9], [11, 12]]
group_pos = [2.5, 5.5, 8.5, 11.5]
fig = plt.figure()
colors = ['dimgray', 'silver']
bar_width = 1
group_main_effect = ['Whole Brain', 'Grey Matter', 'Brainstem', 'LC']
plt.ylim(0, 35)
retro_patch = mpatches.Patch(color='dimgray', label='RETROICOR vs. Uncleaned')
retro_hr_rvt_patch = mpatches.Patch(color='silver', label='RETROICOR + HR/RVT vs. Uncleaned')

plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.subplots_adjust(bottom=0.23)
plt.xticks(group_pos, group_main_effect, rotation=-45)
plt.title('RETROICOR+HR/RVT effect in % TSNR improvement', size=11, y=1.1)
plt.legend(loc='upper left', handles=[retro_patch, retro_hr_rvt_patch])

for counter, (keys, values) in enumerate(df_dic.items()):
    bars = plt.bar(x_pos[counter],
                   height=[values.loc['Mean']['tsnr_difference_percent_retro_to_uncleaned'],
                           values.loc['Mean']['tsnr_difference_percent_hr_rvt_to_uncleaned']],
                   yerr=[values.loc['Confidence Interval']['tsnr_difference_percent_retro_to_uncleaned'],
                         values.loc['Confidence Interval']['tsnr_difference_percent_hr_rvt_to_uncleaned']],
                   width=bar_width,
                   capsize=5,
                   edgecolor='black',
                   color=colors)

    plt.savefig(BASEPATH + 'combined_retro_hr_rvt_effect.svg', dpi=1200)
