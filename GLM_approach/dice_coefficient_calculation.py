"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

This script compares overlap of the unique variance maps of RETROICOR before/after addition of other melodic components.
Those components, selected based on their spatial overlap with the RETROICOR pattern, 

1. Calculate binarised maps from new thresholded zmaps and calculate overlap


"""

import glob
import numpy as np
import nibabel as nib
import pandas as pd
from Subject_Class import Subject
from nilearn import glm
from nilearn import plotting
from nilearn.glm import threshold_stats_img
import matplotlib.pyplot as plt
from scipy.stats import norm


participant_list = glob.glob('/project/3013068.03/RETROICOR/Example_Visualisation/sub-*/RETRO_vs_AROMA_revised'\
                             '/Unique_Variance_RETRO_fwe_corrected_binarised.nii.gz')
participant_list.sort()
participant_list = participant_list[0:5]

index_list = []
for subject in participant_list:
    
    #Load respective subject
    subject_id = subject[subject.find('sub-'): subject.find('sub-')+7]

    #Binarise and load thresholded z-map of RETROICOR explained variance beyond AROMA
    zmap_original_binarised = nib.load(subject)
    zmap_original_binarised = zmap_original_binarised.get_fdata()
    
    #Loop over potential misclassifications
    zmaps_melodic_added = glob.glob('/project/3013068.03/RETROICOR/Example_Visualisation/{0}/'\
                                     'Melodic_Matching_corrected/potential_misclassfications/AddComp*fwe_corrected.nii.gz'.format(subject_id))
    zmaps_melodic_added.sort()
    dice_list = []
    component_list = []
    for zmap in zmaps_melodic_added:
        melodic_added_nii_old = nib.load(zmap)
        melodic_added = melodic_added_nii_old.get_fdata()
        melodic_added_binarised = melodic_added
        melodic_added_binarised[melodic_added_binarised > 0] = 1
        
        melodic_added_nii = nib.Nifti2Image(melodic_added_binarised, melodic_added_nii_old.affine, melodic_added_nii_old.header)
        component_number = zmap[-23:-21]
        
        dice = np.sum(melodic_added_binarised[zmap_original_binarised==1])*2.0 / (np.sum(melodic_added_binarised) + np.sum(zmap_original_binarised))
        dice_rounded = np.round(dice,3)
        dice_list.append(dice)
        component_list.append(component_number)
        nib.save(melodic_added_nii, zmap[:-7] + '_binarised_dice_{0}.nii.gz'.format(str(dice_rounded)))
        
        dice_frame = pd.DataFrame({'Dice Coefficient': dice_list, 'Melodic_Component': component_list, 'Subject': [subject_id]*len(dice_list)})
    index_list.append(dice_frame)
    overall_dice = pd.concat(index_list)
    overall_dice.to_csv('/project/3013068.03/RETROICOR/Example_Visualisation/misclassification_dice_overview.txt', index=False)