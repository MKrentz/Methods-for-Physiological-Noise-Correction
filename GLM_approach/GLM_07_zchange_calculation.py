"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
@author: markre
"""

import glob
import numpy as np
import nibabel as nib
import pandas as pd
from nilearn import plotting
import matplotlib.pyplot as plt
'''
# Path to data
BASEPATH = '/project/3013068.03/physio_revision_fixed/GLM_approach/'

#Path to FDR corrected model of unique RETROICOR variance
participant_list = glob.glob(BASEPATH + 'sub-*/glm_output/glm5_retro_aroma/'
                             'unique_retro_z_score_bonferroni_corrected.nii.gz')
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
    zmap_mat = np.where((zmap_original_binarized == 0) | (zmap_original_binarized == 1), 1 - zmap_original_binarized,
                        zmap_original_binarized)
    zmap_masked = np.ma.array(np.nan_to_num(zmap_original_data, neginf=0, posinf=0),
                                             mask=zmap_mat)

    # Add z-maps for uncorrected addition of potential misclassifications
    zmaps_melodic_added = glob.glob(BASEPATH + f'{sub_id}/melodic_misclassifications/AddComp*uncorrected.nii.gz')
    zmaps_melodic_added.sort()
    zmap_voxel_number = np.count_nonzero(zmap_masked.data)
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
            masked_difference = melodic_added_masked - zmap_masked
            masked_difference_mean = masked_difference.mean()

            # Create a binarised mask nii-img
            component_number = \
                [int(s) for s in zmaps_melodic_added[zmap_counter][zmaps_melodic_added[zmap_counter].
                                                                   find('AddComp'):].split('_') if s.isdigit()]

            # Calculate z_change index
            z_change_list[zmap_counter] = masked_difference_mean
            component_list[zmap_counter] = component_number[0]

            z_change_frame = pd.DataFrame({'Z Change': z_change_list, 'Melodic Component': component_list,
                                           'Subject': [sub_id]*len(z_change_list), 'Voxel Number': zmap_voxel_number})

        #Create overall index
        index_list.append(z_change_frame)

# Gather results across subjects
overall_z_change = pd.concat(index_list)
overall_z_change.sort_values('Z Change',
                             inplace=True,
                             ascending=True)
overall_z_change.to_csv(BASEPATH + 'misclassification_zchange_overview.txt',
                        index=False)

component_list = []
title_list = []

for index, lines in overall_z_change.iterrows():
    component_list.append(nib.load(BASEPATH + '{0}/melodic_glms_output/z_map_{0}_{1}.nii.gz'.\
                               format(lines['Subject'], lines['Melodic Component'] - 1)))
    title_list.append(lines['Subject'] + ': Component ' + str(lines['Melodic Component']) + ' / Z Change ' + str(np.round(lines['Z Change'], 3)))


# Create plot for all misclassifications sorted by z_change
fig, axes = plt.subplots(nrows=5,
                         ncols=2,
                         figsize=[20, 40])

for component_counter, component in enumerate(component_list[:10]):
    plotting.plot_glass_brain(component,
                              colorbar=True,
                              threshold=None,
                              title=title_list[component_counter],
                              axes=axes[int(component_counter / 2), int(component_counter % 2)],
                              annotate=False,
                              plot_abs=False)
    print('{}%'.format(int(component_counter + 1 / 10) * 100))

plt.savefig('/project/3013068.03/physio_revision_fixed/GLM_approach/overall_z_change_misclassifications.png')

'''

BASEPATH = '/project/3013068.03/physio_revision_fixed/GLM_approach/'

# Load unified model results
participant_list = glob.glob(BASEPATH + 'sub-*/full_model_RETROICOR_FWE.nii.gz')
participant_list.sort()

# Initialize data storage
results = []

for subject_path in participant_list:
    sub_id = subject_path.split('/')[-2]

    # Load unified model result
    unified_zmap = nib.load(subject_path).get_fdata()

    # Load original RETROICOR map (from previous processing)
    original_path = f"{BASEPATH}{sub_id}/glm_output/glm5_retro_aroma/unique_retro_z_score_bonferroni_corrected.nii.gz"
    original_zmap = nib.load(original_path).get_fdata()

    # Calculate difference (unified model vs original)
    z_diff = unified_zmap - original_zmap

    # Create mask of significant voxels in original model
    original_mask = original_zmap > 0

    # Calculate metrics
    voxel_count = np.count_nonzero(original_mask)
    mean_z_change = z_diff[original_mask].mean() if voxel_count > 0 else 0

    # Store results
    results.append({
        'Subject': sub_id,
        'Voxel Number': voxel_count,
        'Z Change': mean_z_change,
        'Z-Map Path': subject_path
    })

# Create DataFrame and save
overall_z_change = pd.DataFrame(results)
overall_z_change.sort_values('Z Change', ascending=True, inplace=True)
overall_z_change.to_csv(BASEPATH + 'unified_model_zchange_overview.txt', index=False)

# Visualization
fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(20, 40))

for idx, (_, row) in enumerate(overall_z_change[:10].iterrows()):
    z_map = nib.load(row['Z-Map Path'])
    title = f"{row['Subject']} | ZΔ: {row['Z Change']:.2f} | Voxels: {row['Voxel Number']}"

    plotting.plot_glass_brain(z_map,
                              colorbar=True,
                              threshold=None,
                              title=title,
                              axes=axes[idx // 2, idx % 2],
                              plot_abs=False)

plt.savefig(f'{BASEPATH}unified_model_zchange_comparison.png')
plt.close()