#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 12:52:40 2021

@author: markre
"""

import nibabel as nib
from Subject_Class_new import Subject
from nilearn.image import resample_to_img
import glob
import numpy as np
from nilearn import plotting

BASEPATH = '/project/3013068.03/physio_revision/TSNR_approach/'

part_list = glob.glob(BASEPATH + 'sub-*')
part_list.sort()
part_list = part_list[:-2]
#Mean Matrices for LC
stress_list = ['sub-002', 'sub-003', 'sub-004', 'sub-007', 'sub-009', 'sub-013', 'sub-015', 'sub-017', 'sub-021',
               'sub-023', 'sub-025', 'sub-027', 'sub-029']

for subject_path in part_list:
    sub_id = subject_path[-7:]
    sub = Subject(sub_id)

    # Account for balancing in stress/control session order
    ses_nr = 2 if sub_id in stress_list else 1
    sub_func_native = sub.get_func_data(run=2, session=ses_nr, MNI=False)
    sub_func_MNI = sub.get_func_data(run=2, session=ses_nr, MNI=True)
    try:
        # LC mask resampling and binarising
        LC_mask = sub.get_LC()
        LC_mask_native = resample_to_img(LC_mask, sub_func_native, interpolation='nearest')
        LC_mask_mat = LC_mask_native.get_fdata()
        LC_mask_mat = np.where((LC_mask_mat == 0) | (LC_mask_mat == 1), 1 - LC_mask_mat, LC_mask_mat)
        LC_mask_nii = nib.Nifti2Image(LC_mask_mat, LC_mask_native.affine, LC_mask_native.header)
        nib.save(LC_mask_nii, BASEPATH + '{0}/masks/LC_mask_native.nii.gz'.format(sub_id))
        plotting.plot_roi(LC_mask_nii,
                          bg_img=sub.get_T1(),
                          display_mode='ortho',
                          output_file=BASEPATH + f'{sub_id}/masks/LC_mask_native')
    except:
        print('No LC mask for {}'.format(sub_id))

    brainstem_mask = nib.load('/project/3013068.03/RETROICOR/MNI152lin_T1_2mm_brainstem_mask.nii.gz')
    brainstem_mask_MNI = resample_to_img(brainstem_mask, sub_func_MNI, interpolation='nearest')
    brainstem_mat = brainstem_mask_MNI.get_fdata()
    brainstem_mat = np.where((brainstem_mat == 0) | (brainstem_mat == 1), 1 - brainstem_mat, brainstem_mat)
    brainstem_mask_nii = nib.Nifti2Image(brainstem_mat, brainstem_mask_MNI.affine, brainstem_mask_MNI.header)
    nib.save(brainstem_mask_nii, BASEPATH + '{0}/masks/brainstem_mask_MNI.nii.gz'.format(sub_id))

    plotting.plot_roi(brainstem_mask_nii,
                      bg_img=sub.get_T1(MNI=True),
                      display_mode='ortho',
                      output_file=BASEPATH + f'{sub_id}/masks/brainstem_mask_MNI')

    # Load in, resample and binarise GM mask
    gm_mask = nib.load(glob.glob(f'/project/3013068.03/fmriprep_test/{sub_id}/ses-mri01/anat/{sub_id}*'
                                 f'desc-aparcaseg_dseg.nii.gz')[0])
    gm_mask_native = resample_to_img(gm_mask, sub_func_native, interpolation='nearest')
    gm_mask_mat = gm_mask_native.get_fdata()
    gm_mask_mat[gm_mask_mat < 1000], gm_mask_mat[gm_mask_mat >= 1000] = 1, 0
    gm_mask_native = nib.Nifti2Image(gm_mask_mat, gm_mask_native.affine, gm_mask_native.header)
    gm_mask_native = resample_to_img(gm_mask_native, sub_func_native, interpolation='nearest')
    nib.save(gm_mask_native, BASEPATH + '{0}/masks/gm_mask_native.nii.gz'.format(sub_id))
    plotting.plot_roi(gm_mask_native,
                      bg_img=sub.get_T1(),
                      display_mode='ortho',
                      output_file=BASEPATH + f'{sub_id}/masks/gm_mask_native')
