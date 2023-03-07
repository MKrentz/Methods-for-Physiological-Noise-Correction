"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

import glob
import numpy as np
import nibabel as nib
import pandas as pd
from nilearn import plotting
import matplotlib.pyplot as plt

# Path to data
BASEPATH = '/project/3013068.03/RETROICOR/Example_Visualisation/'

#Path to FDR corrected model of unique RETROICOR variance
participant_list = glob.glob(BASEPATH + 'sub-*/RETRO_vs_AROMA_revised'\
                             '/Unique_Variance_RETRO_fdr_corrected.nii.gz')
participant_list.sort()


index_list = []

# Loop over all unique RETROICOR outputs
for subject in participant_list:

    # Load respective subject
    sub_id = subject[subject.find('sub-'): subject.find('sub-') + 7]

    # Binarise and load thresholded z-map of RETROICOR explained variance beyond AROMA + Swap 0 and 1
    zmap_original_nii = nib.load(subject)
    zmap_original_data = zmap_original_nii.get_fdata()
    zmap_original_binarized = zmap_original_data.copy()
    zmap_original_binarized[zmap_original_binarized > 0] = 1
    zmap_mat = np.where((zmap_original_binarized == 0) | (zmap_original_binarized == 1), 1 - zmap_original_binarized, zmap_original_binarized)
    zmap_masked = np.ma.array(np.nan_to_num(zmap_original_data, neginf=0, posinf=0),
                                             mask=zmap_mat)

    # Add z-maps for uncorrected addition of potential misclassifications
    zmaps_melodic_added = glob.glob(BASEPATH + '{0}/'\
                                     'Melodic_Matching_corrected/potential_misclassifications/AddComp*uncorrected.nii.gz'.format(sub_id))
    zmaps_melodic_added.sort()

    # Check whether there has been possible misclassifications detected previously for this subject
    if zmaps_melodic_added != []:

        z_change_list = [None] * len(zmaps_melodic_added)
        component_list = [None] * len(zmaps_melodic_added)

        # Loop over all potential misclassifications
        for zmap_counter, zmap in enumerate(zmaps_melodic_added):
            melodic_added_nii = nib.load(zmap)
            melodic_added_data = melodic_added_nii.get_fdata()

            melodic_added_masked = np.ma.array(np.nan_to_num(melodic_added_data, neginf=0, posinf=0),
                                               mask=zmap_mat)
            masked_difference = zmap_masked - melodic_added_masked
            masked_difference_mean = masked_difference.mean()

            # Create a binarised mask nii-img
            component_number = [int(s) for s in zmaps_melodic_added[zmap_counter][zmaps_melodic_added[zmap_counter].find('AddComp'):].split('_') if s.isdigit()]

            # Calculate z_change index
            z_change_list[zmap_counter] = masked_difference_mean
            component_list[zmap_counter] = component_number[0]

            z_change_frame = pd.DataFrame({'Z Change': z_change_list, 'Melodic Component': component_list, 'Subject': [sub_id]*len(z_change_list)})

        #Create overall index
        index_list.append(z_change_frame)

# Gather results across subjects
overall_z_change = pd.concat(index_list)
overall_z_change.sort_values('Z Change', inplace = True, ascending=False)
overall_z_change.to_csv(BASEPATH + 'misclassification_zchange_overview.txt', index = False)

component_list = []
title_list = []

for index, lines in overall_z_change.iterrows():
    component_list.append(nib.load(BASEPATH + '{0}/Melodic_Matching_corrected/z_map_{0}_{1}.nii.gz'.\
                               format(lines['Subject'], lines['Melodic Component'] - 1)))
    title_list.append(lines['Subject'] + ': Component ' + str(lines['Melodic Component']) + ' / Z Change ' + str(np.round(lines['Z Change'], 3)))


# Create plot for all misclassifications sorted by z_change
fig, axes = plt.subplots(nrows = 14, ncols = 2, figsize = [20, 40])
for component_counter, component in enumerate(component_list):
    plotting.plot_glass_brain(component,
                              colorbar = True,
                              threshold = None,
                              title = title_list[component_counter],
                              axes = axes[int(component_counter / 2), int(component_counter % 2)],
                              annotate=False,
                              plot_abs = False)
    print('{}% Done!'.format(component_counter / len(component_list) * 100))

plt.savefig('/project/3013068.03/RETROICOR/overall_z_change_misclassifications.png')

