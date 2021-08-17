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
import glob

BASEPATH = '/project/3013068.03/RETROICOR/TSNR/'

NEW_COLUMN_NAMES = ['Index', 'TSNR_noclean', 'TSNR_RETRO',
       'TSNR_aggrAROMA', 'TSNR_difference_aggrAROMA_uncleaned',
       'TSNR_difference_RETRO_uncleaned',
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
mean_MNI_value_df = pd.read_csv(BASEPATH + 'MNI_means.txt')
mean_MNI_value_df.columns = NEW_COLUMN_NAMES
MNI_plotting_df = pd.DataFrame(columns=NEW_COLUMN_NAMES[1:], index=['Mean', 'Confidence Interval'])
for column_counter, column in enumerate(mean_MNI_value_df.columns[1:]):
    mean, sigma = np.mean(mean_MNI_value_df[column]), np.std(mean_MNI_value_df[column])
    confidence_interval = stats.norm.interval(0.95, loc=mean, scale=sigma/np.sqrt(len(mean_MNI_value_df[column])))
    confidence_interval = confidence_interval[1] - mean
    MNI_plotting_df[column][0], MNI_plotting_df[column][1] = mean, confidence_interval

#Create a dataframe calculating mean and confidence intervals for the cortex gray-matter TSNR map
mean_gm_value_df = pd.read_csv(BASEPATH + 'graymatter_means.txt')
mean_gm_value_df.columns = NEW_COLUMN_NAMES
gm_plotting_df = pd.DataFrame(columns=NEW_COLUMN_NAMES[1:], index=['Mean', 'Confidence Interval'])
for column_counter, column in enumerate(mean_gm_value_df.columns[1:]):
    mean, sigma = np.mean(mean_gm_value_df[column]), np.std(mean_gm_value_df[column])
    confidence_interval = stats.norm.interval(0.95, loc=mean, scale=sigma/np.sqrt(len(mean_gm_value_df[column])))
    confidence_interval = confidence_interval[1] - mean
    gm_plotting_df[column][0], gm_plotting_df[column][1] = mean, confidence_interval

#Create a dataframe calculating mean and confidence intervals for the brainstem TSNR map
mean_brainstem_value_df = pd.read_csv(BASEPATH + 'brainstem_means.txt')
mean_brainstem_value_df.columns = NEW_COLUMN_NAMES
brainstem_plotting_df = pd.DataFrame(columns=NEW_COLUMN_NAMES[1:], index=['Mean', 'Confidence Interval'])
for column_counter, column in enumerate(mean_brainstem_value_df.columns[1:]):
    mean, sigma = np.mean(mean_brainstem_value_df[column]), np.std(mean_brainstem_value_df[column])
    confidence_interval = stats.norm.interval(0.95, loc=mean, scale=sigma/np.sqrt(len(mean_brainstem_value_df[column])))
    confidence_interval = confidence_interval[1] - mean
    brainstem_plotting_df[column][0], brainstem_plotting_df[column][1] = mean, confidence_interval

#Create a dataframe calculating mean and confidence intervals for the LC TSNR map
mean_LC_value_df = pd.read_csv(BASEPATH + 'LC_means.txt')
mean_LC_value_df.columns = NEW_COLUMN_NAMES
LC_plotting_df = pd.DataFrame(columns=NEW_COLUMN_NAMES[1:], index=['Mean', 'Confidence Interval'])
for column_counter, column in enumerate(mean_LC_value_df.columns[1:]):
    mean, sigma = np.mean(mean_LC_value_df[column]), np.std(mean_LC_value_df[column])
    confidence_interval = stats.norm.interval(0.95, loc=mean, scale=sigma/np.sqrt(len(mean_LC_value_df[column])))
    confidence_interval = confidence_interval[1] - mean
    LC_plotting_df[column][0], LC_plotting_df[column][1] = mean, confidence_interval

df_dic = ({'Whole Brain': MNI_plotting_df, 'Gray Matter': gm_plotting_df, 'Brainstem': brainstem_plotting_df, 'LC': LC_plotting_df})


#============================================
#      ===== MIXED GRAPH =====
#============================================


for keys, values in df_dic.items():

    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    patterns1 = ["\\" , "/",  "."]
    patterns2 = ["/",  "."]

    bar_width = 0.5
    bars_main_effect = ('RETROICOR', 'AROMA')
    bars2_difference_effect = ('RETROICOR', 'AROMA')

    y_pos1 = [0.6 , 1.2]
    y_pos2 = [0.6, 1.2]
    bars = ax1.bar(y_pos1,
                  height = [values['percent_increase_RETRO'][0], values['percent_increase_AROMA'][0]],
                  yerr = [values['percent_increase_RETRO_mean_scaled'][1], values['percent_increase_AROMA_mean_scaled'][1]],
                  width = bar_width,
                  capsize = 5,
                  edgecolor = 'black',
                  color = 'white')

    for bar, pattern in zip(bars, patterns1):
        bar.set_hatch(pattern)

    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.xticks(y_pos1, bars_main_effect, rotation = -45)
    plt.title('Mean TSNR', size = 11, y = 1.1)

    ax2 = fig.add_subplot(122)
    bars2 = ax2.bar(y_pos2,
                  height = [values['TSNR_difference_aggrAROMARETRO_aggrAROMA'][0], values['TSNR_difference_aggrAROMARETRO_RETRO'][0]],
                  yerr = [values['TSNR_difference_aggrAROMARETRO_aggrAROMA'][1], values['TSNR_difference_aggrAROMARETRO_RETRO'][1]],
                  width = bar_width,
                  capsize = 7,
                  edgecolor = 'black',
                  color = 'white')

    for bar, pattern in zip(bars2, patterns2):
        bar.set_hatch(pattern)


    plt.xticks(y_pos2, bars2_difference_effect, rotation = -45)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.title('Unique TSNR Improvement', size = 11, y = 1.1)
    fig.suptitle('{}'.format(keys), size = 15, y = 0.9)
    fig.tight_layout(pad = 2)
    plt.savefig(BASEPATH + '/Mixed_{}_Effects.png'.format(keys))

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
        color = 'white',
        hatch = "\\",
        label = 'Uncleaned')

plt.bar(r2,
        bars2,
        yerr = yerr2,
        width = bar_width,
        capsize = 5,
        edgecolor = 'black',
        color = 'white',
        hatch = "/",
        label = 'RETROICOR')


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
        color = 'white',
        hatch = "\\",
        label = 'RETROICOR')

plt.bar(r2,
        bars2,
        yerr = yerr2,
        width = bar_width,
        capsize = 5,
        edgecolor = 'black',
        color = 'white',
        hatch = "/",
        label = 'AROMA')

plt.xticks(xtick, [keys for keys in df_dic.keys()], rotation=-45)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.legend(frameon=False)
plt.title('Overall Unique Effects', size = 11, y = 1.1)
fig.subplots_adjust(bottom=0.2)
plt.savefig(BASEPATH + '/Overall_UNIQUE_Effects.png')
