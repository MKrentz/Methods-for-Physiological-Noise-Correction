#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 11:43:04 2021

@author: markre
"""
import glob
import numpy as np
import nibabel as nib
import pandas as pd
from Subject_Class_new import Subject
from nilearn import glm
from nilearn import plotting
from scipy.stats import norm
from nilearn.datasets import load_mni152_brain_mask

BASEPATH = '/project/3013068.03/physio_revision/GLM_approach/'
FMRIPREP_PATH = "/project/3013068.03/fmriprep_test/"

# List of all pre-thresholded z-maps (FWE/p<.05) displaying unique RETROICOR variance with AROMA in the model.
participant_list = glob.glob(BASEPATH + 'sub-*/glm_output/glm5_retro_aroma'
                             '/unique_retro_z_score_bonferroni_corrected.nii.gz')
participant_list.sort()

# Subjects having the stress sessions as their first functional session (ses-mri02)
stress_list = ['sub-002', 'sub-003', 'sub-004', 'sub-007', 'sub-009', 'sub-013', 'sub-015',
               'sub-017', 'sub-021', 'sub-023', 'sub-025', 'sub-027', 'sub-029']

for subject in participant_list:

    # Load respective subject
    sub_id = subject[subject.find('sub-'):subject.find('sub-') + 7]
    sub = Subject(sub_id)

    # Fix scan parameters
    t_r = 2.02
    n_scans = 240
    frame_times = np.arange(n_scans) * t_r

    # Account for balancing in stress/control session order
    ses_nr = 2 if sub_id in stress_list else 1

    # Binarise and load thresholded z-map of RETROICOR explained variance beyond AROMA
    zmap = nib.load(subject)
    zmap_data = zmap.get_fdata()
    zmap_data_binarised = zmap_data.copy()
    zmap_data_binarised[zmap_data_binarised > 0] = 1

    # Had to do this, no idea why. This part is cursed and I would like to know which devil made this happen.
    zmap_data_binarised_inverted = nib.load(subject).get_fdata()
    zmap_data_binarised_inverted[zmap_data_binarised_inverted > 0] = 1
    zmap_data_binarised_inverted[zmap_data_binarised_inverted == 1] = 2
    zmap_data_binarised_inverted[zmap_data_binarised_inverted == 0] = 1
    zmap_data_binarised_inverted[zmap_data_binarised_inverted == 2] = 0

    # Save binarised z-map
    zmap_nii = nib.Nifti2Image(zmap_data_binarised, zmap.affine, zmap.header)
    nib.save(zmap_nii, subject[:-7] + '_binarised.nii.gz')

    # Account for subjects without supra-threshold voxels in the initial zmap
    if 1.0 not in zmap_data_binarised:
        continue

    # Load respective functional Data
    func_data = sub.get_func_data(session=ses_nr, run=2, task='RS', MNI=True)
    mni_mask = sub.get_brainmask(session=ses_nr, run=2, task='RS', MNI=True)

    # Create GLM with 6mm smoothing and no convolution
    melodic_GLM = glm.first_level.FirstLevelModel(t_r=2.02,
                                                  slice_time_ref=0,
                                                  high_pass=0,
                                                  smoothing_fwhm=6,
                                                  drift_model=None,
                                                  hrf_model=None,
                                                  mask_img=mni_mask,
                                                  verbose=1)

    # Load melodic mixing matrix
    melodic_mixing_matrix = pd.read_csv(glob.glob(FMRIPREP_PATH + f'{sub_id}/ses-mri0{ses_nr+1}/func/{sub_id}_'
                                                                  f'ses-mri0{ses_nr+1}_task'
                                                                  f'-15isomb3TR2020TE28RS*run-2*'
                                                                  f'desc-MELODIC_mixing.tsv')[0],
                                        header=None,
                                        sep='\t')

    # Vector of AROMA classified noise ICs
    sub_noise_components = pd.read_csv(glob.glob(FMRIPREP_PATH + f'{sub_id}/ses-mri0{ses_nr+1}/func/'
                                                                 f'{sub_id}_ses-mri0{ses_nr+1}_'
                                                                 f'task-15isomb3TR2020TE28RS*_dir-COL_run-2*'
                                                                 f'AROMAnoiseICs.csv')[0],
                                       header=None)
    sub_sum = []
    for x in range(0, np.shape(melodic_mixing_matrix)[1]):
        # Bring together Model information and use nilearn function to create design matrix.
        # Note: Modulation in this case is the expression of the regressor.

        # GLM Components and readable model structure for nilearn.glm
        melodic_list = list(melodic_mixing_matrix[x])
        constant = [1] * 240
        design = pd.DataFrame({'1': melodic_list, 'constant': constant}, index=frame_times)

        # Compute GLM
        glm_output = melodic_GLM.fit(func_data, design_matrices=design)

        # Create contrast '1' for simple t-contrast of the one model regressor.
        # Output here is nii-img like nibabel object.
        contrast = glm_output.compute_contrast(contrast_def='1', stat_type='t', output_type='z_score')
        nib.save(contrast, BASEPATH + '{0}/melodic_glms_output/z_map_{0}_{1}.nii.gz'.format(sub_id, x))

        # Create Z-map image of easier visualisation
        plotting.plot_glass_brain(contrast, colorbar=True,
                                  threshold=norm.isf(0.001),
                                  title='Nilearn Z map of Melodic Component {0} (unc p<0.001)'.format(str(x+1)),
                                  plot_abs=False,
                                  display_mode='ortho',
                                  output_file=BASEPATH + f'{sub_id}/melodic_glms_output/z_map_{sub_id}_{x}.png')

        # Masking of data for goodness of fit calculation
        masked_difference = np.ma.masked_array(contrast.get_fdata(), mask=zmap_data_binarised)
        masked_difference_sum = masked_difference.sum()
        masked_difference_mean = masked_difference.mean()

        masked_difference_inverted = np.ma.masked_array(contrast.get_fdata(), mask=zmap_data_binarised_inverted)
        masked_difference_inverted_sum = masked_difference_inverted.sum()
        masked_difference_inverted_mean = masked_difference_inverted.mean()

        goodness_of_fit = masked_difference_inverted_mean - masked_difference_mean
        unique, counts = np.unique(zmap_data_binarised, return_counts=True)

        # Add Classification as Noise or Signal based on AROMA classification
        if x+1 in list(sub_noise_components.T[0]):
            noise = 'Noise'
        else:
            noise = 'Signal'

        sub_sum.append([masked_difference_inverted_sum, masked_difference_inverted_mean, masked_difference_sum,
                        masked_difference_mean, goodness_of_fit, counts[1], noise])

    Subject_Data = pd.DataFrame(sub_sum, columns=['Inside Mask Sum', 'Inside Mask Mean', 'Outside Mask Sum',
                                                  'Outside Mask Mean', 'Goodness of Fit', 'Mask Voxel Count',
                                                  'Component Classification'])
    Subject_Data.to_csv(BASEPATH + '{0}/melodic_glms_output/{0}_summary.txt'.format(sub_id))
