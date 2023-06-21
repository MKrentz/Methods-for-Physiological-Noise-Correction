#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 20 15:30:33 2021

@author: markre

"""

import sys
import numpy as np
import nibabel as nib
import pandas as pd
from Subject_Class_new import Subject
from nilearn import glm
from nilearn.glm import threshold_stats_img

print(sys.argv[1])
sub_id = sys.argv[1]

BASEPATH = '/project/3013068.03/physio_revision/GLM_approach/'
"""part_list = glob.glob(BASEPATH + 'sub-*')
part_list.sort() 
part_list = [part_list[0]]
"""
# Indicating subject having the 'stress' condition during their FIRsT functional session
stress_list = ['sub-002', 'sub-003', 'sub-004', 'sub-007', 'sub-009', 'sub-013', 'sub-015',
               'sub-017', 'sub-021', 'sub-023', 'sub-025', 'sub-027', 'sub-029']
cor_ls = ['fdr', 'bonferroni']

# Invoke subject_Class to allow access to all necessary data
glm_path = '/project/3013068.03/physio_revision/GLM_approach/{0}/glm_output/'.format(sub_id)
sub = Subject(sub_id)

# Account for balancing in stress/control session order
ses_nr = 2 if sub_id in stress_list else 1

# Base scan settings for the used sequence
t_r = 2.02
n_scans = 240
constant = [1.0]*240
frame_times = np.arange(n_scans)*t_r

# standard MNI mask used for masking
mni_mask = sub.get_brainmask(MNI=True, session=ses_nr, run=2)

# GLM settings
melodic_GLM = glm.first_level.FirstLevelModel(t_r=2.02,
                                              slice_time_ref=None,
                                              smoothing_fwhm=6,
                                              drift_model=None,
                                              hrf_model=None,
                                              mask_img=mni_mask,
                                              verbose=1)

# Loading respective functional run as NII-img-like nibabel object
func_data = sub.get_func_data(session=ses_nr,
                              run=2,
                              task='RS',
                              MNI=True)

# All file-paths to the respective regressor files as created by 'Confound_file_creation_aroma_model.py'

retro_noise = sub.get_retroicor_confounds(session=ses_nr, run=2, task='RS')
aroma_noise = sub.get_aroma_confounds(session=ses_nr, run=2, task='RS')
acompcor_noise = sub.get_acompcor_confounds(session=ses_nr, run=2, task='RS', number_regressors=5)

# Create lists of data contained in the regressor files
regressors_glm6 = pd.concat([retro_noise, aroma_noise, acompcor_noise], axis=1)
regressors_glm6['constant'] = constant

contrast_length_retro = len(retro_noise.columns)
contrast_length_aroma = len(aroma_noise.columns)
contrast_length_acompcor = len(acompcor_noise.columns)

# Create design matrix as used by nilearn(v0.8) for retro unique variance. This is done because
# I could not figure out
# how to create a non-singular design matrix for the F-contrast

# GLM retroICOR
design_glm1 = retro_noise.copy()
design_glm1['constant'] = constant
design_glm1.index = frame_times
glm_output = melodic_GLM.fit(func_data, design_matrices=design_glm1)
contrast_matrix = np.eye(design_glm1.shape[1])
F_contrast_retro = contrast_matrix[:contrast_length_retro]
F_contrast_retro_output = glm_output.compute_contrast([F_contrast_retro],
                                                      stat_type='F',
                                                      output_type='all')
glm1_report = glm_output.generate_report(contrasts=[F_contrast_retro],
                                         title=f'{sub_id} GLM1',
                                         plot_type='glass')
glm1_report.save_as_html(f'{glm_path}/{sub_id}_report_glm1.html')

# save outputs
for output_type in F_contrast_retro_output.keys():
    F_contrast_retro_output[output_type].to_filename(glm_path + f'glm1_retro/retro_{output_type}.nii.gz')

# Threshold maps
for correction in cor_ls:
    thresholded_contrast = threshold_stats_img(F_contrast_retro_output['z_score'],
                                               alpha=.05,
                                               height_control=correction)
    nib.save(thresholded_contrast[0], glm_path + f'glm1_retro/retro_z_score_{correction}_corrected.nii.gz')

# GLM aroma
design_glm2 = aroma_noise.copy()
design_glm2['constant'] = constant
design_glm2.index = frame_times
glm_output = melodic_GLM.fit(func_data, design_matrices=design_glm2)
contrast_matrix = np.eye(design_glm2.shape[1])
F_contrast_aroma = contrast_matrix[:contrast_length_aroma]
F_contrast_aroma_output = glm_output.compute_contrast([F_contrast_aroma],
                                                      stat_type='F',
                                                      output_type='all')
glm2_report = glm_output.generate_report(contrasts=[F_contrast_aroma],
                                         title=f'{sub_id} GLM2',
                                         plot_type='glass')
glm2_report.save_as_html(f'{glm_path}/{sub_id}_report_glm2.html')
# save resulting z-maps (unthresholded)

for output_type in F_contrast_aroma_output.keys():
    F_contrast_aroma_output[output_type].to_filename(glm_path + f'glm2_aroma/aroma_{output_type}.nii.gz')

for correction in cor_ls:
    thresholded_contrast = threshold_stats_img(F_contrast_aroma_output['z_score'],
                                               alpha=.05,
                                               height_control=correction)
    nib.save(thresholded_contrast[0], glm_path + f'glm2_aroma/aroma_z_score_{correction}_corrected.nii.gz')

# GLM acompcor
design_glm3 = acompcor_noise.copy()
design_glm3['constant'] = constant
design_glm3.index = frame_times
glm_output = melodic_GLM.fit(func_data, design_matrices=design_glm3)
contrast_matrix = np.eye(design_glm3.shape[1])
F_contrast_acompcor = contrast_matrix[:contrast_length_acompcor]
F_contrast_acompcor_output = glm_output.compute_contrast([F_contrast_acompcor],
                                                         stat_type='F',
                                                         output_type='all')
glm3_report = glm_output.generate_report(contrasts=[F_contrast_acompcor],
                                         title=f'{sub_id} GLM3',
                                         plot_type='glass')
glm3_report.save_as_html(f'{glm_path}/{sub_id}_report_glm3.html')

for output_type in F_contrast_acompcor_output.keys():
    F_contrast_aroma_output[output_type].to_filename(glm_path + f'glm3_acompcor/acompcor_{output_type}.nii.gz')

for correction in cor_ls:
    thresholded_contrast = threshold_stats_img(F_contrast_aroma_output['z_score'],
                                               alpha=.05,
                                               height_control=correction)
    nib.save(thresholded_contrast[0], glm_path + f'glm3_acompcor/acompcor_z_score_{correction}_corrected.nii.gz')

# GLM acompcor + aroma
design_glm4 = pd.concat([acompcor_noise.copy(), aroma_noise.copy()], axis=1)
design_glm4['constant'] = constant
design_glm4.index = frame_times
contrast_matrix = np.eye(design_glm4.shape[1])
F_contrast_aroma_unique = contrast_matrix[:contrast_length_aroma]
F_contrast_acompcor_unique = contrast_matrix[contrast_length_aroma:-1]
F_contrast_shared = contrast_matrix[:contrast_length_aroma+contrast_length_acompcor]
glm_output = melodic_GLM.fit(func_data, design_matrices=design_glm4)
glm4_report = glm_output.generate_report(contrasts=[F_contrast_aroma_unique,
                                                    F_contrast_acompcor_unique,
                                                    F_contrast_shared],
                                         title=f'{sub_id} GLM4',
                                         plot_type='glass')
glm4_report.save_as_html(f'{glm_path}/{sub_id}_report_glm4.html')
# Create contrast matrix for F-tests

# Compute contrasts (for the shared contrast the choice of GLM is redundant)

glm4_contrast_dic = {'unique_aroma': glm_output.compute_contrast([F_contrast_aroma_unique],
                                                                 stat_type='F',
                                                                 output_type='all'),
                     'unique_acompcor': glm_output.compute_contrast([F_contrast_acompcor_unique],
                                                                    stat_type='F',
                                                                    output_type='all'),
                     'shared_aroma_acompcor': glm_output.compute_contrast([F_contrast_shared],
                                                                          stat_type='F',
                                                                          output_type='all')}

for contrast in glm4_contrast_dic.keys():
    for output_type in glm4_contrast_dic[contrast].keys():
        glm4_contrast_dic[contrast][output_type].to_filename(glm_path + f'glm4_aroma_acompcor/{contrast}'
                                                                        f'_{output_type}.nii.gz')
    for correction in cor_ls:
        thresholded_contrast = threshold_stats_img(glm4_contrast_dic[contrast]['z_score'],
                                                   alpha=.05,
                                                   height_control=correction)
        nib.save(thresholded_contrast[0],
                 glm_path + f'glm4_aroma_acompcor/{contrast}_z_score_{correction}_corrected.nii.gz')

# GLM retroICOR on top of aroma
design_glm5 = pd.concat([retro_noise.copy(), aroma_noise.copy()], axis=1)
design_glm5['constant'] = constant
design_glm5.index = frame_times
contrast_matrix = np.eye(design_glm5.shape[1])
F_contrast_retro_unique = contrast_matrix[:contrast_length_retro]
F_contrast_aroma_unique = contrast_matrix[contrast_length_retro:-1]
F_contrast_shared = contrast_matrix[:contrast_length_retro+contrast_length_aroma]

glm_output = melodic_GLM.fit(func_data, design_matrices=design_glm5)
glm5_report = glm_output.generate_report(contrasts=[F_contrast_retro_unique,
                                                    F_contrast_aroma_unique,
                                                    F_contrast_shared],
                                         title=f'{sub_id} GLM5',
                                         plot_type='glass')
glm5_report.save_as_html(f'{glm_path}/{sub_id}_report_glm5.html')

glm5_contrast_dic = {'unique_retro': glm_output.compute_contrast([F_contrast_retro_unique],
                                                                 stat_type='F',
                                                                 output_type='all'),
                     'unique_aroma': glm_output.compute_contrast([F_contrast_aroma_unique],
                                                                 stat_type='F',
                                                                 output_type='all'),
                     'shared_retro_aroma': glm_output.compute_contrast([F_contrast_shared],
                                                                       stat_type='F',
                                                                       output_type='all')}

for contrast in glm5_contrast_dic.keys():
    for output_type in glm5_contrast_dic[contrast].keys():
        glm5_contrast_dic[contrast][output_type].to_filename(glm_path + f'glm5_retro_aroma/{contrast}'
                                                                        f'_{output_type}.nii.gz')
    for correction in cor_ls:
        thresholded_contrast = threshold_stats_img(glm5_contrast_dic[contrast]['z_score'],
                                                   alpha=.05,
                                                   height_control=correction)
        nib.save(thresholded_contrast[0],
                 glm_path + f'glm5_retro_aroma/{contrast}_z_score_{correction}_corrected.nii.gz')

# GLM retroICOR on top of aroma and acompcor
design_glm6 = regressors_glm6.copy()
design_glm6.index = frame_times
glm_output = melodic_GLM.fit(func_data, design_matrices=design_glm6)
contrast_matrix = np.eye(design_glm6.shape[1])
F_contrast_retro_unique = contrast_matrix[:contrast_length_retro]
F_contrast_aroma_unique = contrast_matrix[contrast_length_retro:contrast_length_retro+contrast_length_aroma]
F_contrast_acompcor_unique = contrast_matrix[contrast_length_retro+contrast_length_aroma:-1]
F_contrast_shared = contrast_matrix[:contrast_length_retro+contrast_length_aroma+contrast_length_acompcor]
glm6_report = glm_output.generate_report(contrasts=[F_contrast_retro_unique,
                                                    F_contrast_aroma_unique,
                                                    F_contrast_acompcor_unique,
                                                    F_contrast_shared],
                                         title=f'{sub_id} GLM6',
                                         plot_type='glass')
glm6_report.save_as_html(f'{glm_path}/{sub_id}_report_glm6.html')

# Create contrast matrix for F-tests

glm6_contrast_dic = {'unique_retro': glm_output.compute_contrast([F_contrast_retro_unique],
                                                                 stat_type='F',
                                                                 output_type='all'),
                     'unique_aroma': glm_output.compute_contrast([F_contrast_aroma_unique],
                                                                 stat_type='F',
                                                                 output_type='all'),
                     'unique_acompcor': glm_output.compute_contrast([F_contrast_acompcor_unique],
                                                                    stat_type='F',
                                                                    output_type='all'),
                     'shared_retro_aroma_acompcor': glm_output.compute_contrast([F_contrast_shared],
                                                                                stat_type='F',
                                                                                output_type='all')}

for contrast in glm6_contrast_dic.keys():
    for output_type in glm6_contrast_dic[contrast].keys():
        glm6_contrast_dic[contrast][output_type].to_filename(glm_path + f'glm6_retro_aroma_acompcor/{contrast}_'
                                                                        f'{output_type}.nii.gz')

    for correction in cor_ls:
        thresholded_contrast = threshold_stats_img(glm6_contrast_dic[contrast]['z_score'],
                                                   alpha=.05,
                                                   height_control=correction)
        nib.save(thresholded_contrast[0],
                 glm_path + f'glm6_retro_aroma_acompcor/{contrast}_z_score_{correction}_corrected.nii.gz')

# ADD HR and RVT

full_physio = sub.get_physio(session=ses_nr, run=2, task='RS')
retro_addition_noise = full_physio[full_physio.columns[-5:]]
cl_retro_addition = np.shape(retro_addition_noise)[1]
design_glm7 = pd.concat([retro_noise.copy(),
                         retro_addition_noise.copy(),
                         aroma_noise.copy(),
                         acompcor_noise.copy()],
                        axis=1)
design_glm7['constant'] = constant
design_glm7.index = frame_times
glm_output = melodic_GLM.fit(func_data, design_matrices=design_glm7)
contrast_matrix = np.eye(design_glm7.shape[1])
F_contrast_retro_unique = contrast_matrix[:contrast_length_retro]
F_contrast_retro_addition_unique = contrast_matrix[contrast_length_retro:contrast_length_retro+cl_retro_addition]
F_contrast_retro_combined = contrast_matrix[:contrast_length_retro+cl_retro_addition]
F_contrast_aroma_unique = contrast_matrix[contrast_length_retro+cl_retro_addition:
                                          contrast_length_retro+cl_retro_addition+contrast_length_aroma]
F_contrast_acompcor_unique = contrast_matrix[contrast_length_retro+contrast_length_aroma+cl_retro_addition:-1]
F_contrast_shared = contrast_matrix[:-1]
glm7_report = glm_output.generate_report(contrasts=[F_contrast_retro_unique,
                                                    F_contrast_retro_addition_unique,
                                                    F_contrast_retro_combined,
                                                    F_contrast_aroma_unique,
                                                    F_contrast_acompcor_unique,
                                                    F_contrast_shared],
                                         title=f'{sub_id} GLM7',
                                         plot_type='glass')
glm7_report.save_as_html(f'{glm_path}/{sub_id}_report_glm7.html')

# contrast calculation
glm7_contrast_dic = {'unique_retro': glm_output.compute_contrast([F_contrast_retro_unique],
                                                                 stat_type='F',
                                                                 output_type='all'),
                     'unique_retro_addition': glm_output.compute_contrast([F_contrast_retro_addition_unique],
                                                                          stat_type='F',
                                                                          output_type='all'),
                     'unique_retro_combined': glm_output.compute_contrast([F_contrast_retro_combined],
                                                                          stat_type='F',
                                                                          output_type='all'),
                     'unique_aroma': glm_output.compute_contrast([F_contrast_aroma_unique],
                                                                 stat_type='F',
                                                                 output_type='all'),
                     'unique_acompcor': glm_output.compute_contrast([F_contrast_acompcor_unique],
                                                                    stat_type='F',
                                                                    output_type='all'),
                     'shared_retro_addition_aroma_acompcor': glm_output.compute_contrast([F_contrast_shared],
                                                                                         stat_type='F',
                                                                                         output_type='all')}
for contrast in glm7_contrast_dic.keys():
    for output_type in glm7_contrast_dic[contrast].keys():
        glm7_contrast_dic[contrast][output_type].to_filename(glm_path + f'glm7_retro_addition_aroma_acompcor/'
                                                                        f'{contrast}_{output_type}.nii.gz')

    for correction in cor_ls:
        thresholded_contrast = threshold_stats_img(glm7_contrast_dic[contrast]['z_score'],
                                                   alpha=.05,
                                                   height_control=correction)
        nib.save(thresholded_contrast[0],
                 glm_path + f'glm7_retro_addition_aroma_acompcor/{contrast}_z_score_{correction}_corrected.nii.gz')

