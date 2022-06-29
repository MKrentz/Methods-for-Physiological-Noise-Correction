# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import glob
import pandas as pd
import nibabel as nib
import numpy as np

#Set general path to GLM folder
BASEPATH = '/project/3013068.03/test/TSNR_approach/'

#Source all subjects within the folder
part_list = glob.glob(BASEPATH + 'sub-*')
part_list.sort() 

output_list_names = ['tsnr_noclean_MNI', 'tsnr_retro_MNI', 'tsnr_aroma_MNI', 'tsnr_acompcor_MNI', 'tsnr_aroma_retro_MNI', 'tsnr_aroma_acompcor_MNI', 'tsnr_aroma_retro_acompcor_MNI', 'tsnr_difference_unique_retro_to_aroma_MNI',
'tsnr_difference_unique_aroma_to_retro_MNI', 'tsnr_difference_unique_acompcor_to_aroma_MNI','tsnr_difference_unique_retro_to_aroma_acompcor_MNI',
'tsnr_difference_aroma_to_uncleaned_MNI',
'tsnr_difference_aroma_retro_to_uncleaned_MNI', 'tsnr_difference_retro_to_uncleaned_MNI',
'tsnr_difference_percent_retro_to_uncleaned_MNI',
'tsnr_difference_percent_aroma_to_uncleaned_MNI', 'tsnr_difference_percent_acompcor_to_uncleaned_MNI', 'tsnr_difference_percent_unique_aroma_to_retro_MNI' ,'tsnr_difference_percent_unique_retro_to_aroma_MNI', 
'tsnr_difference_percent_unique_acompcor_to_aroma_MNI', 'tsnr_difference_percent_unique_retro_to_aroma_acompcor_MNI']

overall_results = pd.DataFrame(index = [sub_id[-7:] for sub_id in part_list], columns = output_list_names)

for subs in part_list:
    sub_id = subs[-7:]
    sub_path = BASEPATH + '{}/glms/'.format(sub_id)
    for names in output_list_names:
        overall_results.at[sub_id, names] = glob.glob(sub_path + names + '.nii.gz')[0]

for glms in output_list_names:
    temp_list = []
    for subjects in overall_results[glms]:
        temp_list.append(nib.load(subjects).get_fdata())
    temp_mean = np.mean(temp_list, axis = 0)
    nib.save(nib.Nifti2Image(temp_mean, affine = nib.load(subjects).affine, header = nib.load(subjects).header), '/project/3013068.03/test/TSNR_approach/mean_TSNR/' + glms[:-4] + '.nii.gz')




glm_descriptive_names = ['Uncleaned', 'RETROICOR', 'AROMA', 'aCompCor', 'Combined tSNR RETROICOR+AROMA', 'Combined tSNR AROMA+aCompCor', 'Combined tSNR RETROICOR+AROMA+aCompCor', 'RETROICOR over AROMA | tSNR Difference',
'AROMA over RETROICOR | tSNR Difference', 'aCompCor over AROMA | tSNR Difference','RETROICOR over AROMA + aCompCor | tSNR Difference',
'AROMA over Uncleaned | tSNR Difference',
'Combined RETROICOR + AROMA over Uncleaned | tSNR Difference', 'RETROICOR over Uncleaned | tSNR Difference',
'RETROICOR over Uncleaned | tSNR Difference in Percent',
'AROMA over Uncleaned | tSNR Difference in Percent', 'aCompCor over Uncleaned | tSNR Difference in Percent', 'AROMA over RETROICOR | tSNR Difference in Percent' ,'RETROICOR over AROMA | tSNR Difference in Percent', 
'aCompCor over AROMA | tSNR Difference in Percent', 'RETROICOR over AROMA + aCompCor | tSNR Difference in Percent']

mean_list = glob.glob(BASEPATH + 'mean_TSNR/*')

for mean_counter, mean_name in enumerate(mean_list):
    nilearn.plotting.plot_img(nib.load(mean_name), 
                              display_mode = 'z',
                              threshold = 0.001,
                              black_bg = True,
                              colorbar = True,
                              cmap = 'hot',
                              cut_coords = 8,
                              vmin = 0,
                              output_file = BASEPATH + 'mean_TSNR/' + names[mean_counter])