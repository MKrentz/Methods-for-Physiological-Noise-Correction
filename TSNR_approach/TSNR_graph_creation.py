#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 13 11:16:15 2021

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
       'TSNR_difference_aggrAROMARETRO_uncleaned']


mean_MNI_value_df = pd.read_csv(BASEPATH + 'MNI_means.txt')
mean_MNI_value_df.columns = NEW_COLUMN_NAMES
MNI_plotting_df = pd.DataFrame(columns=NEW_COLUMN_NAMES[1:], index=['Mean', 'Confidence Interval'])
for column_counter, column in enumerate(mean_MNI_value_df.columns[1:]):
    mean, sigma = np.mean(mean_MNI_value_df[column]), np.std(mean_MNI_value_df[column])
    confidence_interval = stats.norm.interval(0.95, loc=mean, scale=sigma/np.sqrt(len(mean_MNI_value_df[column])))
    confidence_interval = confidence_interval[1] - mean
    MNI_plotting_df[column][0], MNI_plotting_df[column][1] = mean, confidence_interval


mean_brainstem_value_df = pd.read_csv(BASEPATH + 'brainstem_means.txt')
mean_brainstem_value_df.columns = NEW_COLUMN_NAMES
brainstem_plotting_df = pd.DataFrame(columns=NEW_COLUMN_NAMES[1:], index=['Mean', 'Confidence Interval'])
for column_counter, column in enumerate(mean_brainstem_value_df.columns[1:]):
    mean, sigma = np.mean(mean_brainstem_value_df[column]), np.std(mean_brainstem_value_df[column])
    confidence_interval = stats.norm.interval(0.95, loc=mean, scale=sigma/np.sqrt(len(mean_brainstem_value_df[column])))
    confidence_interval = confidence_interval[1] - mean
    brainstem_plotting_df[column][0], brainstem_plotting_df[column][1] = mean, confidence_interval


mean_gm_value_df = pd.read_csv(BASEPATH + 'graymatter_means.txt')
mean_gm_value_df.columns = NEW_COLUMN_NAMES
gm_plotting_df = pd.DataFrame(columns=NEW_COLUMN_NAMES[1:], index=['Mean', 'Confidence Interval'])
for column_counter, column in enumerate(mean_gm_value_df.columns[1:]):
    mean, sigma = np.mean(mean_gm_value_df[column]), np.std(mean_gm_value_df[column])
    confidence_interval = stats.norm.interval(0.95, loc=mean, scale=sigma/np.sqrt(len(mean_gm_value_df[column])))
    confidence_interval = confidence_interval[1] - mean
    gm_plotting_df[column][0], gm_plotting_df[column][1] = mean, confidence_interval


mean_LC_value_df = pd.read_csv(BASEPATH + 'LC_means.txt')
mean_LC_value_df.columns = NEW_COLUMN_NAMES
LC_plotting_df = pd.DataFrame(columns=NEW_COLUMN_NAMES[1:], index=['Mean', 'Confidence Interval'])
for column_counter, column in enumerate(mean_LC_value_df.columns[1:]):
    mean, sigma = np.mean(mean_LC_value_df[column]), np.std(mean_LC_value_df[column])
    confidence_interval = stats.norm.interval(0.95, loc=mean, scale=sigma/np.sqrt(len(mean_LC_value_df[column])))
    confidence_interval = confidence_interval[1] - mean
    LC_plotting_df[column][0], LC_plotting_df[column][1] = mean, confidence_interval

df_dic = ({'Whole Brain': MNI_plotting_df, 'Brainstem': brainstem_plotting_df, 'Gray Matter': gm_plotting_df, 'LC': LC_plotting_df})
for keys, values in df_dic.items():

    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    patterns1 = ["\\" , "/",  "."]
    patterns2 = ["/",  "."]
    #csfont = {'fontname':'Helvetica'}

    bar_width = 0.5
    bars_main_effect = ('Uncleaned', 'RETROICOR', 'AROMA')
    bars2_difference_effect = ('RETROICOR', 'AROMA')

    y_pos1 = [0.6 , 1.2, 1.8]
    y_pos2 = [0.6, 1.2]
    bars = ax1.bar(y_pos1,
                  height = [values['TSNR_noclean'][0], values['TSNR_RETRO'][0], values['TSNR_aggrAROMA'][0]],
                  yerr = [values['TSNR_noclean'][1], values['TSNR_RETRO'][1], values['TSNR_aggrAROMA'][1]],
                  width = bar_width,
                  capsize = 7,
                  edgecolor = 'black',
                  color = 'white')

    for bar, pattern in zip(bars, patterns1):
        bar.set_hatch(pattern)

    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.xticks(y_pos1, bars_main_effect, rotation = -45)
    plt.title('Mean TSNR', size = 11, y = 1.1, fontname = 'Helvetica')

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
    plt.title('Unique TSNR Improvement', size = 11, y = 1.1, fontname = 'Helvetica')
    fig.suptitle('{}'.format(keys), size = 15, y = 0.9, fontname = 'Helvetica')
    fig.tight_layout(pad = 2)