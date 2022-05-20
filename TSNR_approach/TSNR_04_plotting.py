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



BASEPATH = '/project/3013068.03/test/TSNR_approach/'

output_names = ['RETROICOR Cleaned', 'AROMA Cleaned', 'aCompCor Cleaned', 'Unique aCompCor Effect to AROMA', 'Unique RETROICOR Effect to AROMA', 'Unique RETROICOR Effect to AROMA and aCompCor',
                'Percent RETROICOR Effect', 'Percent AROMA Effect', 'Percent aCompCor Effect', 'Percent RETROICOR Effect vs AROMA', 'Percent RETROICOR Effect vs AROMA and aCompCor']

NEW_COLUMN_NAMES = ['Index', 'TSNR_noclean', 'TSNR_RETRO',
       'TSNR_AROMA', 'TSNR_difference_AROMA_to_uncleaned',
       'TSNR_difference_RETRO_to_uncleaned',
       'TSNR_difference_aCompCor_to_uncleaned',
       'TSNR_difference_RETRO_aggrAROMA',
       'TSNR_difference_aggrAROMARETRO_RETRO',
       'TSNR_difference_aggrAROMARETRO_aggrAROMA',
       'TSNR_difference_aggrAROMARETRO_uncleaned',
                    'percent_increase_RETRO',
                    'percent_increase_AROMA',
                    'mean_average_RETROAROMA',
                    'percent_increase_RETRO_mean_scaled',
                    'percent_increase_AROMA_mean_scaled'
                    ]

#Create a dataframe calculating mean and confidence intervals for the full-brain TSNR map
mean_MNI_value_df = pd.read_csv(BASEPATH + 'MNI_means.txt', index_col = 0)
new_column_names = [None]*len(mean_MNI_value_df.columns)
for counter, columns in enumerate(mean_MNI_value_df.columns):
    new_column_names[counter] = columns.replace('_MNI', '')
mean_MNI_value_df.columns = new_column_names

MNI_plotting_df = pd.DataFrame(columns=mean_MNI_value_df.columns, index = ['Mean', 'Confidence Interval'])
for column_counter, column in enumerate(mean_MNI_value_df.columns):
    mean, sigma = np.mean(mean_MNI_value_df[column]), np.std(mean_MNI_value_df[column])
    confidence_interval = stats.norm.interval(0.95, loc=mean, scale=sigma/np.sqrt(len(mean_MNI_value_df[column])))
    confidence_interval = confidence_interval[1] - mean
    MNI_plotting_df[column][0], MNI_plotting_df[column][1] = mean, confidence_interval

#Create a dataframe calculating mean and confidence intervals for the cortex gray-matter TSNR map
mean_gm_value_df = pd.read_csv(BASEPATH + 'graymatter_means.txt', index_col = 0)
mean_gm_value_df.columns = new_column_names
gm_plotting_df = pd.DataFrame(columns = mean_gm_value_df.columns, index = ['Mean', 'Confidence Interval'])
for column_counter, column in enumerate(mean_gm_value_df.columns):
    mean, sigma = np.mean(mean_gm_value_df[column]), np.std(mean_gm_value_df[column])
    confidence_interval = stats.norm.interval(0.95, loc=mean, scale=sigma/np.sqrt(len(mean_gm_value_df[column])))
    confidence_interval = confidence_interval[1] - mean
    gm_plotting_df[column][0], gm_plotting_df[column][1] = mean, confidence_interval

#Create a dataframe calculating mean and confidence intervals for the brainstem TSNR map
mean_brainstem_value_df = pd.read_csv(BASEPATH + 'brainstem_means.txt', index_col = 0)
mean_brainstem_value_df.columns = new_column_names

brainstem_plotting_df = pd.DataFrame(columns = mean_brainstem_value_df.columns, index = ['Mean', 'Confidence Interval'])
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


x_pos = [[2, 3, 4],[6, 7, 8],[10, 11, 12],[ 14, 15, 16]]
group_pos = [3, 7, 11, 15]
fig = plt.figure()

for counter, (keys, values) in enumerate(df_dic.items()):   
    ax1 = fig.add_subplot()
    colors = ['dimgray','silver', 'whitesmoke']
    bar_width = 1
    group_main_effect = ['MNI', 'Grey Matter', 'Brainstem', 'LC']
    bars2_difference_effect = ('RETROICOR', 'AROMA', 'aCompCor')
    plt.ylim(0,80)
    
    bars = ax1.bar(x_pos[counter],
                  height = [values.loc['Mean']['tsnr_difference_percent_retro_to_uncleaned'], values.loc['Mean']['tsnr_difference_percent_aroma_to_uncleaned'], 
                            values.loc['Mean']['tsnr_difference_percent_acompcor_to_uncleaned']],
                  yerr = [values.loc['Confidence Interval']['tsnr_difference_percent_retro_to_uncleaned'], values.loc['Confidence Interval']['tsnr_difference_percent_aroma_to_uncleaned'], 
                            values.loc['Confidence Interval']['tsnr_difference_percent_acompcor_to_uncleaned']],
                  width = bar_width,
                  capsize = 5,
                  edgecolor = 'black',
                  color = colors)
    
    retro_patch = mpatches.Patch(color = 'dimgray', label = 'RETROICOR')
    aroma_patch = mpatches.Patch(color = 'silver', label = 'AROMA')
    acompcor_patch = mpatches.Patch(color = 'whitesmoke', label = 'aCompCor')
    
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.xticks(group_pos, group_main_effect, rotation = -45)
    plt.title('% TSNR Improvement Compared to Uncleaned', size = 11, y = 1.1)
    plt.legend(loc='upper right', handles= [retro_patch, aroma_patch, acompcor_patch])
    plt.savefig('/project/3013068.03/test/TSNR_approach/main_effects.png')


x_pos = [[2, 3, 4],[6, 7, 8],[10, 11, 12],[ 14, 15, 16]]
group_pos = [3, 7, 11, 15]
fig = plt.figure()

for counter, (keys, values) in enumerate(df_dic.items()):   
    ax1 = fig.add_subplot()
    colors = ['dimgray','silver', 'whitesmoke']
    bar_width = 1
    group_main_effect = ['MNI', 'Grey Matter', 'Brainstem', 'LC']
    plt.ylim(0,25)
    
    bars = ax1.bar(x_pos[counter],
                  height = [values.loc['Mean']['tsnr_difference_percent_retro_to_uncleaned'], values.loc['Mean']['tsnr_difference_percent_unique_retro_to_aroma'], 
                            values.loc['Mean']['tsnr_difference_percent_unique_retro_to_aroma_acompcor']],
                  yerr = [values.loc['Confidence Interval']['tsnr_difference_percent_retro_to_uncleaned'], values.loc['Confidence Interval']['tsnr_difference_percent_unique_retro_to_aroma'], 
                            values.loc['Confidence Interval']['tsnr_difference_percent_unique_retro_to_aroma_acompcor']],
                  width = bar_width,
                  capsize = 5,
                  edgecolor = 'black',
                  color = colors)
    
    retro_patch = mpatches.Patch(color = 'dimgray', label = 'RETROICOR vs. Uncleaned')
    aroma_patch = mpatches.Patch(color = 'silver', label = 'RETROICOR vs. AROMA')
    acompcor_patch = mpatches.Patch(color = 'whitesmoke', label = 'RETROICOR vs AROMA + aCompCor')
    
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.xticks(group_pos, group_main_effect, rotation = -45)
    plt.title('Unique RETROICOR effect in % TSNR improvement', size = 11, y = 1.1)
    plt.legend(loc='upper center', handles= [retro_patch, aroma_patch, acompcor_patch])
    plt.savefig('/project/3013068.03/test/TSNR_approach/unique_retro_effect.png')
#============================================
#      ===== MAIN EFFECT GRAPH =====
# Create a 4x2 graph of the main TSNR effects
# as a function of image space
#============================================

fig = plt.figure()
patterns1 = ["\\" , "/"]
bar_width = 0.5
key_list = ['percent_increase_RETRO', 'percent_increase_AROMA']

bars1 = [df_dic[keys][id][0] for keys, values in df_dic.items() for id in values if id == 'percent_increase_RETRO']
bars2 = [df_dic[keys][id][0] for keys, values in df_dic.items() for id in values if id == 'percent_increase_AROMA']
yerr1 = [df_dic[keys][id][1] for keys, values in df_dic.items() for id in values if id == 'percent_increase_RETRO']
yerr2 = [df_dic[keys][id][1] for keys, values in df_dic.items() for id in values if id == 'percent_increase_AROMA']

r1 = [1, 3, 5, 7]
r2 = [x + bar_width for x in r1]
r3 = [x + bar_width*2 for x in r1]

plt.xlim((0, 9))

plt.bar(r1,
        bars1,
        yerr = yerr1,
        width = bar_width,
        capsize = 5,
        edgecolor = 'black',
        color = 'dimgray',
        label = 'RETROICOR')

plt.bar(r2,
        bars2,
        yerr = yerr2,
        width = bar_width,
        capsize = 5,
        edgecolor = 'black',
        color = 'whitesmoke',
        label = 'AROMA')


plt.xticks(r2, [keys for keys in df_dic.keys()], rotation=-45)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.legend(frameon=False)
plt.title('Percent TSNR Change', size = 11, y = 1.1)
fig.subplots_adjust(bottom=0.2)
plt.savefig(BASEPATH + '/Overall_Main_Effects.png')


# ============================================
#     ===== UNIQUE EFFECT GRAPH =====
# Create a 4x2 graph of the unique TSNR effects
# as a function of image space
# ============================================

fig = plt.figure()
patterns1 = ["\\" , "/",  "."]
bar_width = 0.5
key_list = ['TSNR_difference_aggrAROMARETRO_aggrAROMA', 'TSNR_difference_aggrAROMARETRO_RETRO']
bars2_difference_effect = ('RETROICOR', 'AROMA')

bars1 = [df_dic[keys][id][0] for keys, values in df_dic.items() for id in values if id == key_list[0]]
bars2 = [df_dic[keys][id][0] for keys, values in df_dic.items() for id in values if id == key_list[1]]
yerr1 = [df_dic[keys][id][1] for keys, values in df_dic.items() for id in values if id == key_list[0]]
yerr2 = [df_dic[keys][id][1] for keys, values in df_dic.items() for id in values if id == key_list[1]]

r1 = [1, 3, 5, 7]
r2 = [x + bar_width for x in r1]
xtick = [x + bar_width/2 for x in r1]

plt.xlim((0, 9))

plt.bar(r1,
        bars1,
        yerr = yerr1,
        width = bar_width,
        capsize = 5,
        edgecolor = 'black',
        color = 'dimgray',
        hatch = "\\",
        label = 'RETROICOR')

plt.bar(r2,
        bars2,
        yerr = yerr2,
        width = bar_width,
        capsize = 5,
        edgecolor = 'black',
        color = 'whitesmoke',
        hatch = "/",
        label = 'AROMA')

plt.xticks(xtick, [keys for keys in df_dic.keys()], rotation=-45)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.legend(frameon=False)
plt.title('Overall Unique Effects', size = 11, y = 1.1)
fig.subplots_adjust(bottom=0.2)
plt.savefig(BASEPATH + '/Overall_UNIQUE_Effects.png')
