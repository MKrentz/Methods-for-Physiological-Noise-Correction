#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 12:52:40 2021

@author: markre
"""
import numpy as np
import nibabel as nib
import pandas as pd
import os
import glob
from Subject_Class import Subject
import nilearn
from nilearn import image
import numpy.ma as ma
from scipy import stats



MNI_mask = nib.load('/project/3013068.03/RETROICOR/TSNR/MNI152lin_T1_2mm_brain_mask.nii.gz')
mni_mat = MNI_mask.get_fdata()
mni_mat[mni_mat==1] = 2
mni_mat[mni_mat==0] = 1
mni_mat[mni_mat==2] = 0


brainstem_mask = nib.load('/project/3013068.03/RETROICOR/MNI152lin_T1_2mm_brainstem_mask.nii.gz')
brainstem_mat = brainstem_mask.get_fdata()
brainstem_mat[brainstem_mat==1] = 2
brainstem_mat[brainstem_mat==0] = 1
brainstem_mat[brainstem_mat==2] = 0


part_list = glob.glob('/project/3013068.03/RETROICOR/TSNR/sub-*')
part_list.sort() 

for subject_long in part_list:    
    parts = subject_long[-7:]
    if parts == 'sub-001':
        TSNR_noclean = nib.load('/project/3013068.03/RETROICOR/TSNR/{}/TSNR_noclean.nii.gz'.format(parts)).get_fdata()
        TSNR_noclean = TSNR_noclean[:,:,:,np.newaxis]
        TSNR_nonaggrAROMA = nib.load('/project/3013068.03/RETROICOR/TSNR/{}/TSNR_nonaggrAROMA.nii.gz'.format(parts)).get_fdata()
        TSNR_nonaggrAROMA = TSNR_nonaggrAROMA[:,:,:,np.newaxis]
        TSNR_RETRO = nib.load('/project/3013068.03/RETROICOR/TSNR/{}/TSNR_RETRO.nii.gz'.format(parts)).get_fdata()
        TSNR_RETRO = TSNR_RETRO[:,:,:,np.newaxis]
        TSNR_aggrAROMA = nib.load('/project/3013068.03/RETROICOR/TSNR/{}/TSNR_aggrAROMA.nii.gz'.format(parts)).get_fdata()
        TSNR_aggrAROMA = TSNR_aggrAROMA[:,:,:,np.newaxis]
        TSNR_difference_aggrAROMA_normal = nib.load('/project/3013068.03/RETROICOR/TSNR/{}/TSNR_difference_aggrAROMA_normal.nii.gz'.format(parts)).get_fdata()
        TSNR_difference_aggrAROMA_normal = TSNR_difference_aggrAROMA_normal[:,:,:,np.newaxis]
        TSNR_difference_RETRO_normal = nib.load('/project/3013068.03/RETROICOR/TSNR/{}/TSNR_difference_RETRO_normal.nii.gz'.format(parts)).get_fdata()
        TSNR_difference_RETRO_normal = TSNR_difference_RETRO_normal[:,:,:,np.newaxis]
        TSNR_difference_RETRO_aggrAROMA = nib.load('/project/3013068.03/RETROICOR/TSNR/{}/TSNR_difference_RETRO_aggrAROMA.nii.gz'.format(parts)).get_fdata()
        TSNR_difference_RETRO_aggrAROMA = TSNR_difference_RETRO_aggrAROMA[:,:,:,np.newaxis]
        TSNR_difference_aggrAROMARETRO_RETRO = nib.load('/project/3013068.03/RETROICOR/TSNR/{}/TSNR_difference_aggrAROMARETRO_RETRO.nii.gz'.format(parts)).get_fdata()
        TSNR_difference_aggrAROMARETRO_RETRO = TSNR_difference_aggrAROMARETRO_RETRO[:,:,:,np.newaxis]
        TSNR_difference_aggrAROMARETRO_aggrAROMA = nib.load('/project/3013068.03/RETROICOR/TSNR/{}/TSNR_difference_aggrAROMARETRO_aggrAROMA.nii.gz'.format(parts)).get_fdata()
        TSNR_difference_aggrAROMARETRO_aggrAROMA = TSNR_difference_aggrAROMARETRO_aggrAROMA[:,:,:,np.newaxis]
        TSNR_difference_aggrAROMARETRO_normal = nib.load('/project/3013068.03/RETROICOR/TSNR/{}/TSNR_difference_aggrAROMARETRO_normal.nii.gz'.format(parts)).get_fdata()
        TSNR_difference_aggrAROMARETRO_normal = TSNR_difference_aggrAROMARETRO_normal[:,:,:,np.newaxis]
        TSNR_difference_AROMA_normal = nib.load('/project/3013068.03/RETROICOR/TSNR/{}/TSNR_difference_AROMA_normal.nii.gz'.format(parts)).get_fdata()
        TSNR_difference_AROMA_normal = TSNR_difference_AROMA_normal[:,:,:,np.newaxis]

    else:
        TSNR_noclean = np.concatenate((TSNR_noclean, nib.load('/project/3013068.03/RETROICOR/TSNR/{}/TSNR_noclean.nii.gz'.format(parts)).get_fdata()[:,:,:,np.newaxis]), axis=3)
        TSNR_nonaggrAROMA = np.concatenate((TSNR_nonaggrAROMA, nib.load('/project/3013068.03/RETROICOR/TSNR/{}/TSNR_nonaggrAROMA.nii.gz'.format(parts)).get_fdata()[:,:,:,np.newaxis]), axis=3)
        TSNR_RETRO = np.concatenate((TSNR_RETRO, nib.load('/project/3013068.03/RETROICOR/TSNR/{}/TSNR_RETRO.nii.gz'.format(parts)).get_fdata()[:,:,:,np.newaxis]), axis=3)
        TSNR_aggrAROMA = np.concatenate((TSNR_aggrAROMA, nib.load('/project/3013068.03/RETROICOR/TSNR/{}/TSNR_aggrAROMA.nii.gz'.format(parts)).get_fdata()[:,:,:,np.newaxis]), axis=3)
        TSNR_difference_aggrAROMA_normal = np.concatenate((TSNR_difference_aggrAROMA_normal, nib.load('/project/3013068.03/RETROICOR/TSNR/{}/TSNR_difference_aggrAROMA_normal.nii.gz'.format(parts)).get_fdata()[:,:,:,np.newaxis]), axis=3)
        TSNR_difference = np.concatenate((TSNR_difference_RETRO_normal, nib.load('/project/3013068.03/RETROICOR/TSNR/{}/TSNR_difference_RETRO_normal.nii.gz'.format(parts)).get_fdata()[:,:,:,np.newaxis]), axis=3)
        TSNR_difference_RETRO_aggrAROMA = np.concatenate((TSNR_difference_RETRO_aggrAROMA, nib.load('/project/3013068.03/RETROICOR/TSNR/{}/TSNR_difference_RETRO_aggrAROMA.nii.gz'.format(parts)).get_fdata()[:,:,:,np.newaxis]), axis=3)
        TSNR_difference = np.concatenate((TSNR_difference_aggrAROMARETRO_RETRO, nib.load('/project/3013068.03/RETROICOR/TSNR/{}/TSNR_difference_aggrAROMARETRO_RETRO.nii.gz'.format(parts)).get_fdata()[:,:,:,np.newaxis]), axis=3)
        TSNR_difference_aggrAROMARETRO_normal = np.concatenate((TSNR_difference_aggrAROMARETRO_normal, nib.load('/project/3013068.03/RETROICOR/TSNR/{}/TSNR_difference_aggrAROMARETRO_normal.nii.gz'.format(parts)).get_fdata()[:,:,:,np.newaxis]), axis=3)
        TSNR_difference_aggrAROMARETRO_aggrAROMA = np.concatenate((TSNR_difference_aggrAROMARETRO_aggrAROMA, nib.load('/project/3013068.03/RETROICOR/TSNR/{}/TSNR_difference_aggrAROMARETRO_aggrAROMA.nii.gz'.format(parts)).get_fdata()[:,:,:,np.newaxis]), axis=3)
        TSNR_difference_AROMA_normal = np.concatenate((TSNR_difference_AROMA_normal, nib.load('/project/3013068.03/RETROICOR/TSNR/{}/TSNR_difference_AROMA_normal.nii.gz'.format(parts)).get_fdata()[:,:,:,np.newaxis]), axis=3)
  

#Mean Vectors for MNI template      
Mean_Vector_TSNR_noclean_MNI = []
for subject in range(0, np.shape(TSNR_noclean)[3]): Mean_Vector_TSNR_noclean_MNI.append(ma.array(TSNR_noclean[:,:,:,subject], mask=mni_mat).mean())
Mean_Vector_nonaggrAROMA_MNI = []
for subject in range(0, np.shape(TSNR_nonaggrAROMA)[3]): Mean_Vector_nonaggrAROMA_MNI.append(ma.array(TSNR_nonaggrAROMA[:,:,:,subject], mask=mni_mat).mean())
Mean_Vector_RETRO_MNI = []
for subject in range(0, np.shape(TSNR_RETRO)[3]): Mean_Vector_RETRO_MNI.append(ma.array(TSNR_RETRO[:,:,:,subject], mask=mni_mat).mean())
Mean_Vector_aggrAROMA_MNI = []
for subject in range(0, np.shape(TSNR_aggrAROMA)[3]): Mean_Vector_aggrAROMA_MNI.append(ma.array(TSNR_aggrAROMA[:,:,:,subject], mask=mni_mat).mean())
Mean_Vector_RETRO_normal_MNI = []
for subject in range(0, np.shape(TSNR_difference_RETRO_normal)[3]): Mean_Vector_RETRO_normal_MNI.append(ma.array(TSNR_difference_RETRO_normal[:,:,:,subject], mask=mni_mat).mean())
Mean_Vector_RETRO_aggrAROMA_MNI = []
for subject in range(0, np.shape(TSNR_difference_RETRO_aggrAROMA)[3]): Mean_Vector_RETRO_aggrAROMA_MNI.append(ma.array(TSNR_difference_RETRO_aggrAROMA[:,:,:,subject], mask=mni_mat).mean())
Mean_Vector_aggrAROMARETRO_RETRO_MNI = []
for subject in range(0, np.shape(TSNR_difference_aggrAROMARETRO_RETRO)[3]): Mean_Vector_aggrAROMARETRO_RETRO_MNI.append(ma.array(TSNR_difference_aggrAROMARETRO_RETRO[:,:,:,subject], mask=mni_mat).mean())
Mean_Vector_aggrAROMARETRO_aggrAROMA_MNI = []
for subject in range(0, np.shape(TSNR_difference_aggrAROMARETRO_aggrAROMA)[3]): Mean_Vector_aggrAROMARETRO_aggrAROMA_MNI.append(ma.array(TSNR_difference_aggrAROMARETRO_aggrAROMA[:,:,:,subject], mask=mni_mat).mean())
Mean_Vector_aggrAROMARETRO_normal_MNI = []
for subject in range(0, np.shape(TSNR_difference_aggrAROMARETRO_normal)[3]): Mean_Vector_aggrAROMARETRO_normal_MNI.append(ma.array(TSNR_difference_aggrAROMARETRO_normal[:,:,:,subject], mask=mni_mat).mean())
Mean_Vector_AROMA_normal_MNI = []
for subject in range(0, np.shape(TSNR_difference_AROMA_normal)[3]): Mean_Vector_AROMA_normal_MNI.append(ma.array(TSNR_difference_AROMA_normal[:,:,:,subject], mask=mni_mat).mean())

#Mean Vectors for Brainstem
Mean_Vector_TSNR_noclean_brainstem = []
for subject in range(0, np.shape(TSNR_noclean)[3]): Mean_Vector_TSNR_noclean_brainstem.append(ma.array(TSNR_noclean[:,:,:,subject], mask=brainstem_mat).mean())
Mean_Vector_nonaggrAROMA_brainstem = []
for subject in range(0, np.shape(TSNR_nonaggrAROMA)[3]): Mean_Vector_nonaggrAROMA_brainstem.append(ma.array(TSNR_nonaggrAROMA[:,:,:,subject], mask=brainstem_mat).mean())
Mean_Vector_RETRO_brainstem = []
for subject in range(0, np.shape(TSNR_RETRO)[3]): Mean_Vector_RETRO_brainstem.append(ma.array(TSNR_RETRO[:,:,:,subject], mask=brainstem_mat).mean())
Mean_Vector_aggrAROMA_brainstem = []
for subject in range(0, np.shape(TSNR_aggrAROMA)[3]): Mean_Vector_aggrAROMA_brainstem.append(ma.array(TSNR_aggrAROMA[:,:,:,subject], mask=brainstem_mat).mean())
Mean_Vector_RETRO_normal_brainstem = []
for subject in range(0, np.shape(TSNR_difference_RETRO_normal)[3]): Mean_Vector_RETRO_normal_brainstem.append(ma.array(TSNR_difference_RETRO_normal[:,:,:,subject], mask=brainstem_mat).mean())
Mean_Vector_RETRO_aggrAROMA_brainstem = []
for subject in range(0, np.shape(TSNR_difference_RETRO_aggrAROMA)[3]): Mean_Vector_RETRO_aggrAROMA_brainstem.append(ma.array(TSNR_difference_RETRO_aggrAROMA[:,:,:,subject], mask=brainstem_mat).mean())
Mean_Vector_aggrAROMARETRO_RETRO_brainstem = []
for subject in range(0, np.shape(TSNR_difference_aggrAROMARETRO_RETRO)[3]): Mean_Vector_aggrAROMARETRO_RETRO_brainstem.append(ma.array(TSNR_difference_aggrAROMARETRO_RETRO[:,:,:,subject], mask=brainstem_mat).mean())
Mean_Vector_aggrAROMARETRO_aggrAROMA_brainstem = []
for subject in range(0, np.shape(TSNR_difference_aggrAROMARETRO_aggrAROMA)[3]): Mean_Vector_aggrAROMARETRO_aggrAROMA_brainstem.append(ma.array(TSNR_difference_aggrAROMARETRO_aggrAROMA[:,:,:,subject], mask=brainstem_mat).mean())
Mean_Vector_aggrAROMARETRO_normal_brainstem = []
for subject in range(0, np.shape(TSNR_difference_aggrAROMARETRO_normal)[3]): Mean_Vector_aggrAROMARETRO_normal_brainstem.append(ma.array(TSNR_difference_aggrAROMARETRO_normal[:,:,:,subject], mask=brainstem_mat).mean())
Mean_Vector_AROMA_normal_brainstem = []
for subject in range(0, np.shape(TSNR_difference_AROMA_normal)[3]): Mean_Vector_AROMA_normal_brainstem.append(ma.array(TSNR_difference_AROMA_normal[:,:,:,subject], mask=brainstem_mat).mean())

#Stats
#Stats for MNI
#TSNR non-cleaned vs TSNR-RETRO
stats.ttest_rel(Mean_Vector_TSNR_noclean_MNI, Mean_Vector_RETRO_MNI)
stats.ttest_rel(Mean_Vector_TSNR_noclean_MNI, Mean_Vector_aggrAROMA_MNI)
stats.ttest_1samp(Mean_Vector_aggrAROMARETRO_RETRO_MNI, population=0)
stats.describe(Mean_Vector_aggrAROMARETRO_RETRO_MNI)
stats.ttest_1samp(Mean_Vector_aggrAROMARETRO_aggrAROMA_MNI, population=0)
stats.describe(Mean_Vector_aggrAROMARETRO_aggrAROMA_MNI)
#Stats for brainstem
stats.ttest_rel(Mean_Vector_TSNR_noclean_brainstem, Mean_Vector_RETRO_brainstem)
stats.ttest_rel(Mean_Vector_TSNR_noclean_brainstem, Mean_Vector_aggrAROMA_brainstem)
stats.ttest_1samp(Mean_Vector_aggrAROMARETRO_RETRO_brainstem, population=0)
stats.describe(Mean_Vector_aggrAROMARETRO_RETRO_brainstem)
stats.ttest_1samp(Mean_Vector_aggrAROMARETRO_aggrAROMA_brainstem, population=0)
stats.describe(Mean_Vector_aggrAROMARETRO_aggrAROMA_brainstem)
