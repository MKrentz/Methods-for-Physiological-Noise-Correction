#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 22 12:34:33 2021

@author: markre

Script to create plots of thresholded maps for AROMA vs. RETRO comparison


"""


import glob
from nilearn import plotting
import matplotlib.pyplot as plt

#Set general path to GLM folder
BASEPATH = '/project/3013068.03/test/GLM_approach/'

#Source all subjects within the folder
part_list = glob.glob(BASEPATH + 'sub-*')
part_list.sort() 

#Loop over subjects
for subs in part_list:
    #Create a glassbrain-graph per subject for each GLM contrast in each of the 6GLMs
    try:
        sub_id = subs[-7:]
        sub_path = '/project/3013068.03/test/GLM_approach/{}/glm_output/'.format(sub_id)
        zmaps_total = glob.glob(sub_path + 'glm*/*.nii.gz')
        
        for zmap_counter, subject_zmap in enumerate(zmaps_total):
            plotting.plot_glass_brain(subject_zmap,
                                      colorbar = True,
                                      threshold = None,
                                      title = sub_id + zmaps_total[zmap_counter][zmaps_total[zmap_counter].rfind('/')+1:-7],
                                      output_file = sub_path + zmaps_total[zmap_counter][zmaps_total[zmap_counter].rfind('glm'):-7] + '.png',
                                      plot_abs = False)
            plt.close()
    
    #In case not all subjects have been processed in GLM_02_run 
    except:
        print('{} does not have calculated Z-Maps'.format(sub_id))
        continue

#Create a list of all GLM contrasts across subjects
approaches_fdr = []
approaches_fwe = []
for subs in part_list:
    sub_id = subs[-7:]
    sub_path = '/project/3013068.03/test/GLM_approach/{}/glm_output/'.format(sub_id)
    zmaps_fdr = glob.glob(sub_path + '*/*fdr_corrected.nii.gz')
    zmaps_fwe = glob.glob(sub_path + '*/*fwe_corrected.nii.gz')
    if approaches_fdr == []:
        for count,x in enumerate(zmaps_fdr):
                approaches_fdr.append([x])
    else:
        for count,x in enumerate(zmaps_fdr):
            approaches_fdr[count].append(x)
    if approaches_fwe == []:
        for count,x in enumerate(zmaps_fwe):
                approaches_fwe.append([x])
    else:
        for count,x in enumerate(zmaps_fwe):
            approaches_fwe[count].append(x)
        

#Create a contrast glassbrain collection across subjects for each GLM contrast in each GLM\

vmin_list = []
vmax_list = []

for approach_counter, approach in enumerate(approaches_fdr):
    fig, axes = plt.subplots(nrows = 9, ncols = 3, figsize = [15, 25])
    for cidx, zmap in enumerate(approach): 
        subject_id = zmap[zmap.find('sub-'):zmap.find('sub-') +7 ]
        print(subject_id)
        plotting.plot_glass_brain(zmap,
                                  colorbar = True,
                                  threshold = None,
                                  title = subject_id,
                                  axes = axes[int(cidx / 3), int(cidx % 3)],
                                  annotate = False,
                                  plot_abs = True)
    plt.savefig(BASEPATH + 'fdr_plot/' + approaches_fdr[approach_counter][0][approaches_fdr[approach_counter][0].rfind('glm'):-7].replace('/', '_') + '.png')
    plt.close()

#Create a contrast glassbrain collection across subjects for each GLM contrast in each GLM
for approach_counter, approach in enumerate(approaches_fwe):
    fig, axes = plt.subplots(nrows = 9, ncols = 3, figsize = [15, 25])
    for cidx, zmap in enumerate(approach): 
        subject_id = zmap[zmap.find('sub-'):zmap.find('sub-') +7 ]
        print(subject_id)
        plotting.plot_glass_brain(zmap,
                                  colorbar = True,
                                  threshold = None,
                                  title = subject_id,
                                  axes = axes[int(cidx / 3), int(cidx % 3)],
                                  annotate = False,
                                  plot_abs = False)
    plt.savefig(BASEPATH + 'fwe_plot/' + approaches_fwe[approach_counter][0][approaches_fwe[approach_counter][0].rfind('glm'):-7].replace('/', '_') + '.png')
    plt.close()