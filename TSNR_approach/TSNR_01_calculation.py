#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 17:55:52 2021

This script implements the calculation of TSNR for different noise-cleaning procedures.

Creating different TSNR map for:
    Uncleaned data
    aroma cleaned data
    retroICOR cleaned data
    aroma AND retroICOR cleaned data

Additionally maps are created visualising the unique contributions of a method OVER another.
    Unique TSNR improvement of aroma (TSNR of aroma+retro - TSNR of retro)
    Unique TSNR improvement of retroICOR (TSNR of aroma+retro - TSNR of aroma)
    Difference in TSNR improvement between retroICOR and aroma (TSNR of aroma - TSNR of retro)
    TSNR improvement of uncleaned data for aroma (TSNR of aroma - TSNR of uncleaned data)
    TSNR improvement of uncleaned data for retroICOR (TSNR of retroICOR - TSNR of uncleaned data)

@author: MKrentz
"""

import numpy as np
import nibabel as nib
import pandas as pd
import glob
from Subject_Class import Subject
import nilearn
from nilearn import image
import numpy.ma as ma
import random
from nilearn.plotting import plot_design_matrix
import matplotlib.pyplot as plt
import json
import gc

MNI_MASK = nib.load('/project/3013068.03/retroICOR/Filled_Nilearn_MNI_Mask.nii.gz')
BASEPATH = '/project/3013068.03/test/TSNR_approach/'

# Load MNI mask to used masked data matrices and switch 0 to 1 and 1 to 0
mni_mat = MNI_MASK.get_fdata()
mni_mat = np.where((mni_mat == 0)|(mni_mat == 1), 1-mni_mat, mni_mat)

# Load all available participants
part_list = glob.glob(BASEPATH + 'sub-*')
part_list.sort()


# Indicating subject having the 'stress' condition during their FIRST functional session
stress_list = ['sub-002', 'sub-003', 'sub-004', 'sub-007', 'sub-009', 'sub-013', 'sub-015', 'sub-017', 'sub-021', 'sub-023', 'sub-025', 'sub-027', 'sub-029']

# Chosen retroICOR components
retro_model = ['cardiac_phase_sin_1', 'cardiac_phase_cos_1', 'cardiac_phase_sin_2', 'cardiac_phase_cos_2', 'cardiac_phase_sin_3', 'cardiac_phase_cos_3', \
                                  'respiratory_phase_sin_1', 'respiratory_phase_cos_1', 'respiratory_phase_sin_2', 'respiratory_phase_cos_2', 'respiratory_phase_sin_3', 'respiratory_phase_cos_3', \
                                  'respiratory_phase_sin_4', 'respiratory_phase_cos_4']

# Used for variable naming
def object_name(obj, name_pos):
    return [name for name in name_pos if name_pos[name] is obj]


# Fix the random seed for padding regressor (comment out when using an existing seed dictionary)
def seed_dict_check(existing_dict = None, path = BASEPATH):
    if existing_dict == None:
        seed_dict = {key: [[], []] for key in [subject[-7:] for subject in part_list]}
        return(seed_dict)
    else:
        with open(path + 'seed_dict.txt') as f:
            data = f.read()
            seed_dict = json.loads(data)
        return(seed_dict)

seed_dict = seed_dict_check()

def shuffling(array, sub_id, seed_dict, seed_start, loop_position):
    if not all([all(x) for lists in seed_dict.values() for x in lists]):
        seed = random.random()
        new_array = array.copy();  random.Random(seed).shuffle(new_array)
        seed_dict[sub_id][seed_start][loop_position] = seed
        return new_array

    elif all([all(x) for lists in seed_dict.values() for x in lists]):
        seed = seed_dict[sub_id][seed_start][loop_position]
        new_array = array.copy();  random.Random(seed).shuffle(new_array)
        return new_array
    else:
        print('Whoops!')

# Add uncleaned data

# Subject loop
for subject in part_list:

    # Subject space_identifier
    sub_id = subject[-7:]
    sub_obj = Subject(sub_id)

    # Account for balancing in stress/control session order
    ses_nr = 2 if sub_id in stress_list else 1

    # Loading respective functional data into memory and online-smooth with 6mm FWHM
    func_data_mni = sub_obj.get_func_data(session=ses_nr,run=2,task='RS', MNI=True)
    func_data_mni = image.smooth_img(func_data_mni, fwhm=6)
    func_data_native = sub_obj.get_func_data(session=ses_nr,run=2,task='RS', MNI=False)
    func_data_native = image.smooth_img(func_data_native, fwhm=6)

    # Load fmriprep confound files for respective runs, T1-brainmask and retroICOR regressors
    sub_confounds = sub_obj.get_confounds(session=ses_nr, run=2, task='RS')
    sub_brainmask = sub_obj.get_brainmask(session=ses_nr, run=2, MNI=False).get_fdata()
    sub_brainmask = np.where((sub_brainmask == 0) | (sub_brainmask == 1), 1 - sub_brainmask, sub_brainmask)
    sub_phys = sub_obj.get_physio(session=ses_nr, run=2, task='RS')

    # confound creation
    sub_phys_3C4R = sub_phys[retro_model]

    # Expansion of retroICOR model with multiplication terms
    multi_01 = sub_phys_3C4R['cardiac_phase_sin_1'] * sub_phys_3C4R['respiratory_phase_sin_1']
    multi_02 = sub_phys_3C4R['cardiac_phase_sin_1'] * sub_phys_3C4R['respiratory_phase_cos_2']
    multi_03 = sub_phys_3C4R['cardiac_phase_cos_1'] * sub_phys_3C4R['respiratory_phase_sin_1']
    multi_04 = sub_phys_3C4R['cardiac_phase_cos_1'] * sub_phys_3C4R['respiratory_phase_sin_2']
    sub_phys_3C4R1M = pd.concat([sub_phys_3C4R, multi_01, multi_02, multi_03, multi_04], axis=1)


    confounds_column_index = sub_confounds.columns.tolist()
    aroma_sum = sum((itm.count("aroma_motion") for itm in confounds_column_index))
    aroma_variables = confounds_column_index[-aroma_sum:]
    aroma_regressors = sub_confounds[aroma_variables]
    
    #aCompCor addition
    acompcor_regressors = sub_confounds[['a_comp_cor_0{0}'.format(x) for x in range(5)]] 

    # New column names for padding variables (retro)
    sub_phys_3C4R1M_shuffled = sub_phys_3C4R1M.copy()
    new_names_retro = []
    for counter in range(0,len(sub_phys_3C4R1M_shuffled.columns)):
        new_names_retro.append(str(sub_phys_3C4R1M_shuffled.columns[counter]) + '_randomised')
    sub_phys_3C4R1M_shuffled.columns = new_names_retro

    # New column names for padding variables (AROMA)
    aroma_regressors_shuffled = aroma_regressors.copy()
    new_names_aroma = []
    for counter in range(0,len(aroma_regressors_shuffled.columns)):
        new_names_aroma.append(str(aroma_regressors_shuffled.columns[counter]) + '_randomised')
    aroma_regressors_shuffled.columns = new_names_aroma
    
    # New column names for padding variables (aCompCor)
    acompcor_regressors_shuffled = acompcor_regressors.copy()
    new_names_acompcor = []
    for counter in range(0,len(acompcor_regressors_shuffled.columns)):
        new_names_acompcor.append(str(acompcor_regressors_shuffled.columns[counter]) + '_randomised')
    acompcor_regressors_shuffled.columns = new_names_acompcor

    # Shuffling
    aroma_regressors_shuffled = aroma_regressors.copy()
    sub_phys_3C4R1M_shuffled = sub_phys_3C4R1M.copy()
    acompcor_regressors_shuffled = acompcor_regressors.copy()
    
    if not any(lists for lists in seed_dict[sub_id]):
        seed_dict[sub_id][0] = [None] * len(sub_phys_3C4R1M_shuffled.columns)
        seed_dict[sub_id][1] = [None] * len(aroma_regressors_shuffled.columns)
        seed_dict[sub_id][2] = [None] * len(acompcor_regressors_shuffled.columns)
    
    seed_start = 0
    for vector_number, vectors in enumerate(sub_phys_3C4R1M_shuffled):
        sub_phys_3C4R1M_shuffled[vectors] = shuffling(sub_phys_3C4R1M_shuffled[vectors].copy(),
                                                      sub_id=sub_id,
                                                      seed_dict=seed_dict,
                                                      seed_start=seed_start,
                                                      loop_position=vector_number)

    seed_start = 1
    for vector_number, vectors in enumerate(aroma_regressors_shuffled):
        aroma_regressors_shuffled[vectors] = shuffling(aroma_regressors_shuffled[vectors].copy(),
                                                       sub_id=sub_id,
                                                       seed_dict=seed_dict,
                                                       seed_start=seed_start,
                                                       loop_position=vector_number)
    seed_start = 2
    for vector_number, vectors in enumerate(acompcor_regressors_shuffled):
        acompcor_regressors_shuffled[vectors] = shuffling(acompcor_regressors_shuffled[vectors].copy(),
                                                       sub_id=sub_id,
                                                       seed_dict=seed_dict,
                                                       seed_start=seed_start,
                                                       loop_position=vector_number)

    # Account for processing in MNI space (for MNI-mask and Brainstem Mask) as well as native space (LC mask and GM mask)
    func_data_list = [func_data_mni, func_data_native]
    for func_data_counter, func_data in enumerate(func_data_list):

        if func_data_counter == 0:
            mask = mni_mat
            space_identifier = 'MNI'

        elif func_data_counter == 1:
            mask = sub_brainmask
            space_identifier = 'native'
            del func_data_mni

        # Full brain uncleaned TSNR map
        noclean_dummy = pd.concat([sub_phys_3C4R1M_shuffled, aroma_regressors_shuffled, acompcor_regressors_shuffled], axis = 1)
        fig = plot_design_matrix(noclean_dummy)
        plt.savefig(BASEPATH + '{0}/confounds_cleaning_noclean.png'.format(sub_id))
        plt.close()
        func_data_uncleaned_dummy = nilearn.image.clean_img(func_data, standardize = False, detrend = False, confounds = noclean_dummy, t_r = 2.02)
        tsnr_matrix_uncleaned = np.divide(np.mean(func_data_uncleaned_dummy.get_fdata(), axis = 3), np.std(func_data_uncleaned_dummy.get_fdata(), axis = 3))
        tsnr_matrix_noinf_uncleaned = np.nan_to_num(tsnr_matrix_uncleaned, neginf = 0, posinf = 0)
        del func_data_uncleaned_dummy
        masked_tsnr_uncleaned = ma.array(tsnr_matrix_noinf_uncleaned, mask = mask).filled(0)
        masked_tsnr_uncleaned[masked_tsnr_uncleaned > 500], masked_tsnr_uncleaned[masked_tsnr_uncleaned < -100] = 500, -100
        nib.save(nib.Nifti2Image(masked_tsnr_uncleaned, affine = func_data.affine, header = func_data.header), BASEPATH + '{0}/TSNR_noclean_{1}.nii.gz'.format(sub_id, space_identifier))

        # retroICOR TSNR map
        retro_dummy = pd.concat([sub_phys_3C4R1M, aroma_regressors_shuffled, acompcor_regressors_shuffled], axis = 1)
        fig = plot_design_matrix(retro_dummy)
        plt.savefig(BASEPATH + '{0}/confounds_cleaning_retroICOR.png'.format(sub_id))
        plt.close()
        func_data_phys_cleaned = nilearn.image.clean_img(func_data, standardize = False, detrend = False, confounds = retro_dummy, t_r = 2.02)
        tsnr_matrix_retro = np.divide(np.mean(func_data_phys_cleaned.get_fdata(), axis = 3), np.std(func_data_phys_cleaned.get_fdata(), axis = 3))
        del func_data_phys_cleaned
        masked_tsnr_retro = ma.array(np.nan_to_num(tsnr_matrix_retro, neginf = 0, posinf = 0), mask = mask).filled(0)
        masked_tsnr_retro[masked_tsnr_retro > 500], masked_tsnr_retro[masked_tsnr_retro < -100] = 500, -100
        nib.save(nib.Nifti2Image(masked_tsnr_retro, affine = func_data.affine, header = func_data.header), BASEPATH + '{0}/TSNR_retro_{1}.nii.gz'.format(sub_id, space_identifier))

        # AROMA TSNR map
        aroma_dummy = pd.concat([aroma_regressors, sub_phys_3C4R1M_shuffled, acompcor_regressors_shuffled], axis = 1)
        fig = plot_design_matrix(aroma_dummy)
        plt.savefig(BASEPATH + '{0}/confounds_cleaning_AROMA.png'.format(sub_id))
        plt.close()
        func_data_aroma_cleaned = nilearn.image.clean_img(func_data, standardize = False, detrend = False, confounds = aroma_dummy, t_r = 2.02)
        tsnr_matrix_aroma = np.divide(np.mean(func_data_aroma_cleaned.get_fdata(), axis = 3), np.std(func_data_aroma_cleaned.get_fdata(),axis = 3))
        del func_data_aroma_cleaned
        masked_tsnr_aroma = ma.array(np.nan_to_num(tsnr_matrix_aroma, neginf = 0, posinf = 0), mask = mask).filled(0)
        masked_tsnr_aroma[masked_tsnr_aroma>500], masked_tsnr_aroma[masked_tsnr_aroma<-100] = 500, -100
        nib.save(nib.Nifti2Image(masked_tsnr_aroma, affine = func_data.affine, header = func_data.header), BASEPATH + '{0}/TSNR_aroma_{1}.nii.gz'.format(sub_id,space_identifier))

        # aCompCor TSNR map
        acompcor_dummy = pd.concat([acompcor_regressors, sub_phys_3C4R1M_shuffled, aroma_regressors_shuffled], axis = 1)
        fig = plot_design_matrix(aroma_dummy)
        plt.savefig(BASEPATH + '{0}/confounds_cleaning_aCompCor.png'.format(sub_id))
        plt.close()
        func_data_acompcor_cleaned = nilearn.image.clean_img(func_data, standardize = False, detrend = False, confounds = acompcor_dummy, t_r = 2.02)
        tsnr_matrix_acompcor = np.divide(np.mean(func_data_acompcor_cleaned.get_fdata(), axis = 3), np.std(func_data_acompcor_cleaned.get_fdata(),axis = 3))
        del func_data_acompcor_cleaned
        masked_tsnr_acompcor = ma.array(np.nan_to_num(tsnr_matrix_acompcor, neginf = 0, posinf = 0), mask = mask).filled(0)
        masked_tsnr_acompcor[masked_tsnr_acompcor>500], masked_tsnr_acompcor[masked_tsnr_acompcor<-100] = 500, -100
        nib.save(nib.Nifti2Image(masked_tsnr_acompcor, affine = func_data.affine, header = func_data.header), BASEPATH + '{0}/TSNR_acompcor_{1}.nii.gz'.format(sub_id,space_identifier))

        gc.collect()
        
        #Combined AROMA and aCompCor TSNR map
        combined_aroma_acompcor = pd.concat([aroma_regressors, acompcor_regressors, sub_phys_3C4R1M_shuffled], axis = 1)
        fig = plot_design_matrix(combined_aroma_acompcor)
        plt.savefig(BASEPATH + '{0}/confounds_cleaning_AROMA+retro.png'.format(sub_id))
        func_data_aroma_acompcor_cleaned = nilearn.image.clean_img(func_data, standardize = False, detrend = False, confounds = combined_aroma_acompcor, t_r = 2.02)
        tsnr_matrix_aroma_acompcor = np.divide(np.mean(func_data_aroma_acompcor_cleaned.get_fdata(), axis = 3), np.std(func_data_aroma_acompcor_cleaned.get_fdata(),axis = 3))
        del func_data_aroma_acompcor_cleaned
        masked_tsnr_aroma_acompcor = ma.array(np.nan_to_num(tsnr_matrix_aroma_acompcor, neginf = 0, posinf = 0), mask = mask).filled(0)
        masked_tsnr_aroma_acompcor[masked_tsnr_aroma_acompcor > 500], masked_tsnr_aroma_acompcor[masked_tsnr_aroma_acompcor < -100]  = 500, -100
        nib.save(nib.Nifti2Image(masked_tsnr_aroma_acompcor, affine = func_data.affine, header = func_data.header), BASEPATH + '{0}/TSNR_aroma_acompcor_{1}.nii.gz'.format(sub_id, space_identifier))


        #Combined AROMA and aCompCor TSNR map
        combined_aroma_retro = pd.concat([sub_phys_3C4R1M, aroma_regressors, acompcor_regressors_shuffled], axis = 1)
        fig = plot_design_matrix(combined_aroma_retro)
        plt.savefig(BASEPATH + '{0}/confounds_cleaning_AROMA+retro.png'.format(sub_id))
        func_data_aroma_retro_cleaned = nilearn.image.clean_img(func_data, standardize = False, detrend = False, confounds = combined_aroma_retro, t_r = 2.02)
        tsnr_matrix_aroma_retro = np.divide(np.mean(func_data_aroma_retro_cleaned.get_fdata(), axis = 3), np.std(func_data_aroma_retro_cleaned.get_fdata(),axis = 3))
        del func_data_aroma_retro_cleaned
        masked_tsnr_aroma_retro = ma.array(np.nan_to_num(tsnr_matrix_aroma_retro, neginf = 0, posinf = 0), mask = mask).filled(0)
        masked_tsnr_aroma_retro[masked_tsnr_aroma_retro > 500], masked_tsnr_aroma_retro[masked_tsnr_aroma_retro < -100]  = 500, -100
        nib.save(nib.Nifti2Image(masked_tsnr_aroma_retro, affine = func_data.affine, header = func_data.header), BASEPATH + '{0}/TSNR_aroma_retro_{1}.nii.gz'.format(sub_id, space_identifier))


        # Combined AROMA, retroICOR and aCompCor TSNR map
        combined_regressors = pd.concat([sub_phys_3C4R1M, aroma_regressors, acompcor_regressors], axis = 1)
        fig = plot_design_matrix(combined_regressors)
        plt.savefig(BASEPATH + '{0}/confounds_cleaning_AROMA+retro.png'.format(sub_id))
        func_data_aroma_retro_acompcor_cleaned = nilearn.image.clean_img(func_data, standardize = False, detrend = False, confounds = combined_regressors, t_r = 2.02)
        tsnr_matrix_aroma_retro_acompcor = np.divide(np.mean(func_data_aroma_retro_acompcor_cleaned.get_fdata(), axis = 3), np.std(func_data_aroma_retro_acompcor_cleaned.get_fdata(),axis = 3))
        del func_data_aroma_retro_acompcor_cleaned
        masked_tsnr_aroma_retro_acompcor = ma.array(np.nan_to_num(tsnr_matrix_aroma_retro_acompcor, neginf = 0, posinf = 0), mask = mask).filled(0)
        masked_tsnr_aroma_retro_acompcor[masked_tsnr_aroma_retro_acompcor > 500], masked_tsnr_aroma_retro_acompcor[masked_tsnr_aroma_retro_acompcor < -100]  = 500, -100
        nib.save(nib.Nifti2Image(masked_tsnr_aroma_retro_acompcor, affine = func_data.affine, header = func_data.header), BASEPATH + '{0}/TSNR_aroma_retro_acompcor_{1}.nii.gz'.format(sub_id, space_identifier))


        # Difference aggreAroma to uncleaned
        difference_aroma_uncleaned =  masked_tsnr_aroma - masked_tsnr_uncleaned
        nib.save(nib.Nifti1Image(difference_aroma_uncleaned, affine = func_data.affine, header = func_data.header), BASEPATH + '{0}/TSNR_difference_aroma_to_uncleaned_{1}.nii.gz'.format(sub_id, space_identifier))

        # Difference retro to uncleaned
        difference_retro_uncleaned =  masked_tsnr_retro - masked_tsnr_uncleaned
        nib.save(nib.Nifti2Image(difference_retro_uncleaned, affine = func_data.affine, header = func_data.header), BASEPATH + '{0}/TSNR_difference_retro_to_uncleaned_{1}.nii.gz'.format(sub_id, space_identifier))

        # Difference retro to uncleaned
        difference_acompcor_uncleaned =  masked_tsnr_acompcor - masked_tsnr_uncleaned
        nib.save(nib.Nifti2Image(difference_acompcor_uncleaned, affine = func_data.affine, header = func_data.header), BASEPATH + '{0}/TSNR_difference_acompcor_to_uncleaned_{1}.nii.gz'.format(sub_id, space_identifier))

        
        # Difference aroma and retro to uncleaned
        difference_aroma_acompcor_uncleaned = masked_tsnr_aroma_acompcor - masked_tsnr_uncleaned
        nib.save(nib.Nifti2Image(difference_aroma_acompcor_uncleaned, affine = func_data.affine, header = func_data.header), BASEPATH + '{0}/TSNR_difference_aroma_acompcor_to_uncleaned_{1}.nii.gz'.format(sub_id, space_identifier))

        # Difference aroma and acompcor to uncleaned
        difference_aroma_retro_uncleaned = masked_tsnr_aroma_retro - masked_tsnr_uncleaned
        nib.save(nib.Nifti2Image(difference_aroma_retro_uncleaned, affine = func_data.affine, header = func_data.header), BASEPATH + '{0}/TSNR_difference_aromaretro_to_uncleaned_{1}.nii.gz'.format(sub_id, space_identifier))


        # Unqiue AROMA effect & Percent
        unique_tsnr_aroma_to_retro = masked_tsnr_aroma_retro - masked_tsnr_retro
        nib.save(nib.Nifti2Image(unique_tsnr_aroma_to_retro, affine = func_data.affine, header = func_data.header), BASEPATH + '{0}/TSNR_difference_unique_aroma_to_retro_{1}.nii.gz'.format(sub_id, space_identifier))

        # Unique retro effect to AROMA
        unique_tsnr_retro_to_aroma = masked_tsnr_aroma_retro - masked_tsnr_aroma
        nib.save(nib.Nifti2Image(unique_tsnr_retro_to_aroma, affine = func_data.affine, header = func_data.header), BASEPATH + '{0}/TSNR_difference_unique_retro_to_aroma_{1}.nii.gz'.format(sub_id, space_identifier))

        #Unique effect aCompCor to AROMA
        unique_tsnr_acompcor_to_aroma = masked_tsnr_aroma_acompcor - masked_tsnr_aroma
        nib.save(nib.Nifti2Image(unique_tsnr_acompcor_to_aroma, affine = func_data.affine, header = func_data.header), BASEPATH + '{0}/TSNR_difference_unique_acompcor_to_aroma_{1}.nii.gz'.format(sub_id, space_identifier))


        #Unique retro effect to aroma and acompcor
        unique_tsnr_retro_to_aroma_acompcor = masked_tsnr_aroma_retro_acompcor - (masked_tsnr_aroma + masked_tsnr_acompcor)
        nib.save(nib.Nifti2Image(unique_tsnr_retro_to_aroma_acompcor, affine = func_data.affine, header = func_data.header), BASEPATH + '{0}/TSNR_difference_unique_retro_to_aroma_acompcor_{1}.nii.gz'.format(sub_id, space_identifier))


        #Percentage of unique TSNR of retro to aroma
        percent_unique_tsnr_retro_to_aroma = ((((masked_tsnr_aroma_retro / masked_tsnr_uncleaned) - 1) * 100) - (
                    ((masked_tsnr_aroma / masked_tsnr_uncleaned) - 1) * 100))
        nib.save(nib.Nifti2Image(percent_unique_tsnr_retro_to_aroma, affine=func_data.affine, header=func_data.header),
                 BASEPATH + '{0}/TSNR_difference_unique_percent_retro_to_aroma_{1}.nii.gz'.format(sub_id,space_identifier))
                              
        
        #Percentage of unique TSNR of aroma to retro
        percent_unique_tsnr_aroma_to_retro = ((((masked_tsnr_aroma_retro / masked_tsnr_uncleaned) - 1) * 100) - (((masked_tsnr_retro / masked_tsnr_uncleaned) - 1) * 100))
        nib.save(nib.Nifti2Image(percent_unique_tsnr_aroma_to_retro, affine=func_data.affine, header=func_data.header),
                 BASEPATH + '{0}/TSNR_difference_percent_unqiue_aroma_to_retro_{1}.nii.gz'.format(sub_id, space_identifier))
        
        #Percentage of unique TSNR of acompcor to aroma
        percent_unique_tsnr_acompcor_to_aroma = ((((masked_tsnr_aroma_acompcor / masked_tsnr_uncleaned) - 1) * 100) - (
                    ((masked_tsnr_aroma / masked_tsnr_uncleaned) - 1) * 100))
        nib.save(nib.Nifti2Image(percent_unique_tsnr_acompcor_to_aroma, affine=func_data.affine, header=func_data.header),
                 BASEPATH + '{0}/TSNR_difference_unique_percent_acompcor_to_aroma_{1}.nii.gz'.format(sub_id,space_identifier))
        
        #Percentage of unique TSNR of retro to aroma and acompcor
        percent_unique_tsnr_retro_to_aroma_acompcor = ((((masked_tsnr_aroma_retro_acompcor / masked_tsnr_uncleaned) - 1) * 100) - (
                    (((masked_tsnr_aroma_acompcor) / masked_tsnr_uncleaned) - 1) * 100))
        nib.save(nib.Nifti2Image(percent_unique_tsnr_retro_to_aroma_acompcor, affine=func_data.affine, header=func_data.header),
                 BASEPATH + '{0}/TSNR_difference_unique_percent_retro_to_aroma_acompcor_{1}.nii.gz'.format(sub_id,space_identifier))               
        
        # Difference Percent retro
        percent_retro_uncleaned =  ((masked_tsnr_retro / masked_tsnr_uncleaned) - 1) * 100
        nib.save(nib.Nifti2Image(percent_retro_uncleaned, affine=func_data.affine, header=func_data.header),
                 BASEPATH + '{0}/TSNR_percent_retro_uncleaned_{1}.nii.gz'.format(sub_id, space_identifier))

        # Difference Percent AROMA
        percent_aroma_uncleaned = ((masked_tsnr_aroma / masked_tsnr_uncleaned) - 1) * 100
        nib.save(nib.Nifti2Image(percent_aroma_uncleaned, affine=func_data.affine, header=func_data.header),
                 BASEPATH + '{0}/TSNR_percent_aroma_uncleaned_{1}.nii.gz'.format(sub_id, space_identifier))
        
        # Difference percent acompcor
        percent_acompcor_uncleaned = ((masked_tsnr_acompcor / masked_tsnr_uncleaned) - 1) * 100
        nib.save(nib.Nifti2Image(percent_acompcor_uncleaned, affine=func_data.affine, header=func_data.header),
                 BASEPATH + '{0}/TSNR_percent_acompcor_uncleaned_{1}.nii.gz'.format(sub_id, space_identifier))        
        
        
        mask_list = [masked_tsnr_uncleaned, masked_tsnr_retro, masked_tsnr_aroma, masked_tsnr_acompcor, masked_tsnr_aroma_retro, masked_tsnr_aroma_acompcor, masked_tsnr_aroma_retro_acompcor,
                     unique_tsnr_aroma_to_retro, unique_tsnr_retro_to_aroma, unique_tsnr_acompcor_to_aroma, unique_tsnr_retro_to_aroma_acompcor, difference_aroma_uncleaned,
                     difference_aroma_retro_uncleaned, difference_retro_uncleaned,
                     percent_retro_uncleaned, percent_aroma_uncleaned, percent_acompcor_uncleaned, percent_unique_tsnr_aroma_to_retro,
                     percent_unique_tsnr_retro_to_aroma, percent_unique_tsnr_acompcor_to_aroma, percent_unique_tsnr_retro_to_aroma_acompcor]


        #Create Average TSNR images in MNI space for all comparisons
        if sub_id == part_list[0][-7:] and func_data_counter == 0:
            tsnr_noclean_MNI = masked_tsnr_uncleaned[:, :, :, np.newaxis]
            tsnr_retro_MNI = masked_tsnr_retro[:, :, :, np.newaxis]
            tsnr_aroma_MNI = masked_tsnr_aroma[:, :, :, np.newaxis]
            tsnr_acompcor_MNI = masked_tsnr_acompcor[:, :, :, np.newaxis]
            tsnr_aroma_retro_MNI = masked_tsnr_aroma_retro[:, :, :, np.newaxis]
            tsnr_aroma_acompcor_MNI = masked_tsnr_aroma_acompcor[:, :, :, np.newaxis]
            tsnr_aroma_retro_acompcor_MNI = masked_tsnr_aroma_retro_acompcor[:, :, :, np.newaxis]
            tsnr_unique_retro_to_aroma_MNI = unique_tsnr_retro_to_aroma[:, :, :, np.newaxis]
            tsnr_unique_aroma_to_retro_MNI = unique_tsnr_aroma_to_retro[:, :, :, np.newaxis]
            tsnr_unique_acomppcor_to_aroma_MNI = unique_tsnr_acompcor_to_aroma[:, :, :, np.newaxis]
            tsnr_unique_retro_to_aroma_acompcor_MNI = unique_tsnr_retro_to_aroma_acompcor[:, :, :, np.newaxis]
            tsnr_difference_aroma_uncleaned_MNI = difference_aroma_uncleaned[:, :, :, np.newaxis]
            tsnr_difference_aroma_retro_uncleaned_MNI = difference_aroma_retro_uncleaned[:, :, :, np.newaxis]
            tsnr_difference_retro_uncleaned_MNI = difference_retro_uncleaned[:, :, :, np.newaxis]
            tsnr_percent_retro_uncleaned_MNI = percent_retro_uncleaned[:, :, :, np.newaxis]
            tsnr_percent_aroma_uncleaned_MNI = percent_aroma_uncleaned[:, :, :, np.newaxis]
            tsnr_percent_acompcor_uncleaned_MNI = percent_acompcor_uncleaned[:, :, :, np.newaxis]
            tsnr_percent_unique_aroma_to_retro= percent_unique_tsnr_aroma_to_retro[:, :, :, np.newaxis]
            tsnr_percent_unique_retro_to_aroma = percent_unique_tsnr_retro_to_aroma[:, :, :, np.newaxis]
            tsnr_percent_unique_acompcor_to_aroma = percent_unique_tsnr_acompcor_to_aroma[:, :, :, np.newaxis]            
            tsnr_percent_unique_retro_to_aroma_acompcor = percent_unique_tsnr_retro_to_aroma_acompcor[:, :, :, np.newaxis] 
            
            output_list = [tsnr_noclean_MNI, tsnr_retro_MNI, tsnr_aroma_MNI, tsnr_acompcor_MNI, tsnr_aroma_retro_MNI, tsnr_aroma_acompcor_MNI, 
                           tsnr_aroma_retro_acompcor_MNI, tsnr_unique_aroma_to_retro_MNI, tsnr_unique_acomppcor_to_aroma_MNI,tsnr_unique_retro_to_aroma_acompcor_MNI,
                           tsnr_difference_aroma_uncleaned_MNI,
                           tsnr_difference_aroma_retro_uncleaned_MNI, tsnr_difference_retro_uncleaned_MNI,
                           tsnr_percent_retro_uncleaned_MNI,
                           tsnr_percent_aroma_uncleaned_MNI, tsnr_percent_acompcor_uncleaned_MNI, tsnr_percent_unique_aroma_to_retro,tsnr_percent_unique_retro_to_aroma, 
                           tsnr_percent_unique_acompcor_to_aroma, tsnr_percent_unique_retro_to_aroma_acompcor]

        elif sub_id != part_list[0][-7:] and func_data_counter == 0:
            for output_counter, output in enumerate(output_list):
                output_list[output_counter] = np.concatenate((output, mask_list[output_counter][:, :, :, np.newaxis]), axis = 3)

output_list_names = ['tsnr_noclean_MNI', 'tsnr_retro_MNI', 'tsnr_aroma_MNI', 'tsnr_aromaretro_MNI',
                           'tsnr_difference_aromaretro_retro_MNI', 'tsnr_difference_aromaretro_aroma_MNI',
                           'tsnr_difference_aroma_uncleaned_MNI',
                           'tsnr_difference_aromaretro_uncleaned_MNI', 'tsnr_difference_retro_uncleaned_MNI',
                           'tsnr_difference_retro_aroma_MNI', 'tsnr_percent_retro_uncleaned_MNI',
                     'tsnr_percent_AROMA_uncleaned_MNI', 'tsnr_percent_unique_AROMA',
                     'tsnr_percent_unique_retro']

for output_counter, output in enumerate(output_list):
    nib.save(nib.Nifti2Image(np.mean(output, axis = 3), affine = func_data.affine, header = func_data.header), BASEPATH + 'Overall_{0}.nii.gz'.format(output_list_names[output_counter]))

with open(BASEPATH + 'seed_dict.txt', 'w') as f:
    print(seed_dict, file=f)