#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 22 12:34:33 2021

@author: markre

Script to create plots of thresholded maps for AROMA vs. RETRO comparison


"""


import glob
import numpy as np
import pandas as pd
from nilearn import glm
from nilearn import plotting
import matplotlib.pyplot as plt



#Load FWE-thresholded z-maps for unique variance of RETRO
zmaps_RETRO = glob.glob('/project/3013068.03/RETROICOR/Example_Visualisation/sub-*'\
                        '/RETRO_vs_AROMA_revised/Unique_Variance_RETRO_fwe_corrected.nii.gz')
zmaps_RETRO.sort()

#Load FWE-thresholded z-maps for unique variance of AROMA
zmaps_AROMA = glob.glob('/project/3013068.03/RETROICOR/Example_Visualisation/sub-*'\
                        '/RETRO_vs_AROMA_revised/Unique_Variance_AROMA_fwe_corrected.nii.gz')
zmaps_AROMA.sort()

#Load FWE-thresholded z-maps for shared variance of AROMA and RETROICOR
zmaps_shared = glob.glob('/project/3013068.03/RETROICOR/Example_Visualisation/sub-*'\
                        '/RETRO_vs_AROMA_revised/Shared_Variance_AROMA_RETRO_fwe_corrected.nii.gz')
zmaps_shared.sort()

if len(zmaps_RETRO) != len(zmaps_AROMA) or len(zmaps_RETRO) != len(zmaps_shared):
    print('Not all processed images are present in the dataset!')
    
for subject_zmap in zmap_RETRO:
    subject_id = subject[subject_zmap.find('sub-'):subject_zmap.find('sub-')+7]
    plotting.plot_glass_brain(subject_zmap, colorbar=True, threshold=None, title=subject_id + ': Unique Variance of RETROICOR',\
                              output_file = '/project/3013068.03/RETROICOR/{0}'\
                              '/RETRO_vs_AROMA_revised/Unique_Variance_RETRO_fwe_corrected.png'.format(subject_id), \
                              plot_abs=False)
    plt.close()

for subject_zmap in zmap_AROMA:
    subject_id = subject[subject_zmap.find('sub-'):subject_zmap.find('sub-')+7]
    plotting.plot_glass_brain(subject_zmap, colorbar=True, threshold=None, title=subject_id + ': Unique Variance of RETROICOR',\
                              output_file = '/project/3013068.03/RETROICOR/{0}'\
                              '/RETRO_vs_AROMA_revised/Unique_Variance_AROMA_fwe_corrected.png'.format(subject_id), \
                              plot_abs=False)
    plt.close()

for subject_zmap in zmap_shared:
    subject_id = subject[subject_zmap.find('sub-'):subject_zmap.find('sub-')+7]
    plotting.plot_glass_brain(subject_zmap, colorbar=True, threshold=None, title=subject_id + ': Unique Variance of RETROICOR',\
                              output_file = '/project/3013068.03/RETROICOR/{0}'\
                              '/RETRO_vs_AROMA_revised/Shared_Variance_AROMA_RETRO_fwe_corrected.png'.format(subject_id), \
                              plot_abs=False)
    plt.close()
    
    
    
#Overall plotting of z-maps for RETRO unique variance
fig, axes = plt.subplots(nrows=13, ncols=2, figsize=[15,25]) 
for cidx, zmap in enumerate(zmaps_RETRO): 
    plotting.plot_glass_brain(zmap, colorbar=True, threshold=None, title=subs[cidx], \
                              axes=axes[int(cidx / 2), int(cidx % 2)],annotate=False, \
                              output_file = '/project/3013068.03/RETROICOR/z_map_collection_RETRO.png', \
                              plot_abs=False) 
plt.close()

#Overall plotting of z-maps for AROMA unique variance
fig, axes = plt.subplots(nrows=13, ncols=2, figsize=[15,25]) 
for cidx, zmap in enumerate(zmaps_AROMA): 
    plotting.plot_glass_brain(zmap, colorbar=True, threshold=None, title=subs[cidx], \
                              axes=axes[int(cidx / 2), int(cidx % 2)],annotate=False, \
                              output_file = '/project/3013068.03/RETROICOR/z_map_collection_RETRO.png', \
                              plot_abs=False) 
plt.close()

#Overall plotting of z-maps for RETRO and AROMA shared variance
fig, axes = plt.subplots(nrows=13, ncols=2, figsize=[15,25]) 
for cidx, zmap in enumerate(zmaps_shared: 
    plotting.plot_glass_brain(zmap, colorbar=True, threshold=None, title=subs[cidx], \
                              axes=axes[int(cidx / 2), int(cidx % 2)],annotate=False, \
                              output_file = '/project/3013068.03/RETROICOR/z_map_collection_shared.png', \
                              plot_abs=False) 
plt.close()