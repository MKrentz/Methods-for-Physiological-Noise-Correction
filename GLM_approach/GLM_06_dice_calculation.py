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
from nilearn import plotting
import matplotlib.pyplot as plt

BASEPATH = '/project/3013068.03/RETROICOR/GLM_approach/'

participant_list = glob.glob(BASEPATH + 'sub-*/RETRO_vs_AROMA_revised'\
                             '/Unique_Variance_RETRO_fwe_corrected_binarised.nii.gz')
participant_list.sort()

index_list = []

for subject in participant_list:

    # Load respective subject
    sub_id = subject[subject.find('sub-'): subject.find('sub-') + 7]

    # Binarise and load thresholded z-map of RETROICOR explained variance beyond AROMA
    zmap_original_binarised = nib.load(subject).get_fdata()

    # Loop over potential misclassifications
    zmaps_melodic_added = glob.glob(BASEPATH + '{0}/'\
                                     'Melodic_Matching_corrected/potential_misclassifications/AddComp*fwe_corrected.nii.gz'.format(sub_id))
    zmaps_melodic_added.sort()

    # Check whether there has been possible misclassifications detected previously for this subject
    if zmaps_melodic_added != []:

        dice_list = [None] * len(zmaps_melodic_added)
        component_list = [None] * len(zmaps_melodic_added)

        # Loop over all potential misclassifications
        for zmap_counter, zmap in enumerate(zmaps_melodic_added):
            melodic_added_nii_old = nib.load(zmap)
            melodic_added = melodic_added_nii_old.get_fdata()
            melodic_added_binarised = melodic_added
            melodic_added_binarised[melodic_added_binarised > 0] = 1

            # Create a binarised mask nii-img
            melodic_added_nii = nib.Nifti2Image(melodic_added_binarised, melodic_added_nii_old.affine, melodic_added_nii_old.header)
            component_number = [int(s) for s in zmaps_melodic_added[zmap_counter][zmaps_melodic_added[zmap_counter].find('AddComp'):].split('_') if s.isdigit()]

            # Calculate dice index
            dice = np.sum(melodic_added_binarised[zmap_original_binarised==1])*2.0 / (np.sum(melodic_added_binarised) + np.sum(zmap_original_binarised))
            dice_list[zmap_counter] = np.round(dice,3)
            component_list[zmap_counter] = component_number[0]

            # Save created binarised map
            nib.save(melodic_added_nii, zmap[:-7] + '_binarised_dice_{0}.nii.gz'.format(str(np.round(dice,3))))


            dice_frame = pd.DataFrame({'Dice Coefficient': dice_list, 'Melodic_Component': component_list, 'Subject': [sub_id]*len(dice_list)})

    #Create overall index
        index_list.append(dice_frame)

# Gather results across subjects
overall_dice = pd.concat(index_list)
overall_dice.sort_values('Dice Coefficient', inplace = True)
overall_dice.to_csv(BASEPATH + 'misclassification_dice_overview.txt', index = False)

component_list = []
title_list = []

for index, lines in x.iterrows():
    component_list.append(nib.load(BASEPATH + '{0}/Melodic_Matching_corrected/z_map_{0}_{1}.nii.gz'.\
                               format(lines['Subject'], lines['Melodic_Component'] - 1)))
    title_list.append(lines['Subject'] + ': ' + str(lines['Melodic_Component']) + ' / ' + str(np.round(lines['Dice Coefficient'], 3)))

# Create plot for all misclassifications sorted by dice
fig, axes = plt.subplots(nrows = 14, ncols = 2, figsize = [20, 40])
for component_counter, component in enumerate(component_list):
    plotting.plot_glass_brain(component,
                              colorbar = True,
                              threshold = None,
                              title = title_list[component_counter],
                              axes = axes[int(component_counter / 2), int(component_counter % 2)],
                              annotate=False,
                              output_file = '/project/3013068.03/RETROICOR/overall_dice_misclassifications.png',
                              plot_abs = False)
