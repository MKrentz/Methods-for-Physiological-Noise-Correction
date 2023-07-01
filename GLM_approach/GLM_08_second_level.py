import numpy as np
import pandas as pd
import glob
from nilearn.glm.second_level import SecondLevelModel
import gc
from nilearn.plotting import plot_design_matrix
from nilearn.glm.second_level import non_parametric_inference
import nibabel as nib
import sys


contrast = sys.argv[1]

basepath = '/project/3013068.03/physio_revision/GLM_approach/'
stress_list = ['sub-002', 'sub-003', 'sub-004', 'sub-007', 'sub-009', 'sub-013', 'sub-015', 'sub-017', 'sub-021',
               'sub-023', 'sub-025', 'sub-027', 'sub-029']
subject_list = [i[-7:] for i in glob.glob('/project/3013068.03/physio_revision/GLM_approach/sub*')]

# exclusion_list = ['sub-006', 'sub-026', 'sub-012', 'sub-024', 'sub-015']
# subject_list = [i for i in subject_list if i not in exclusion_list]

# Create base for second level design matrix
n_subjects = len(subject_list)

design_matrix = pd.DataFrame(
    np.hstack(([1] * n_subjects)),
    index=subject_list,
    columns=["Physio Contrast"]
)


contrast_list = ['glm1_retro/retro_effect_size',
                 'glm2_aroma/aroma_effect_size',
                 'glm3_acompcor/acompcor_effect_size',
                 'glm4_aroma_acompcor/unique_acompcor_effect_size',
                 'glm4_aroma_acompcor/unique_aroma_effect_size',
                 'glm4_aroma_acompcor/shared_aroma_acompcor_effect_size',
                 'glm5_retro_aroma/unique_retro_effect_size',
                 'glm5_retro_aroma/unique_aroma_effect_size',
                 'glm5_retro_aroma/shared_retro_aroma_effect_size',
                 'glm6_retro_aroma_acompcor/unique_retro_effect_size',
                 'glm6_retro_aroma_acompcor/unique_aroma_effect_size',
                 'glm6_retro_aroma_acompcor/unique_acompcor_effect_size',
                 'glm6_retro_aroma_acompcor/shared_retro_aroma_acompcor_effect_size',
                 'glm7_retro_addition_aroma_acompcor/unique_retro_effect_size',
                 'glm7_retro_addition_aroma_acompcor/unique_retro_addition_effect_size',
                 'glm7_retro_addition_aroma_acompcor/unique_retro_combined_effect_size',
                 'glm7_retro_addition_aroma_acompcor/unique_aroma_effect_size',
                 'glm7_retro_addition_aroma_acompcor/unique_acompcor_effect_size',
                 'glm7_retro_addition_aroma_acompcor/shared_retro_addition_aroma_acompcor_effect_size']

contrast_list_names = ['Overall RETROICOR Effect',
                       'Overall AROMA Effect',
                       'Overall ACOPMCOR Effect',
                       'Unique ACOMPCOR Effect with AROMA',
                       'Unique AROMA Effect with ACOMPCOR',
                       'Shared Effect of AROMA and ACOMPCOR',
                       'Unique RETROICOR Effect with AROMA',
                       'Unqiue AROMA Effect with RETROICOR',
                       'Shared Effect of RETROICOR and AROMA',
                       'Unique RETROICOR Effect with AROMA and ACOMPCOR',
                       'Unique AROMA Effect with RETROICOR and ACOMPCOR',
                       'Unique ACOMPCOR Effect with RETROICOR and AROMA',
                       'Shared EFFECT of RETROICOR, AROMA and ACOMPCOR',
                       'Unique RETROICOR Effect with HR, RVT, AROMA and ACOMPCOR',
                       'Unique HR&RVT Effect with RETROICOR, AROMA and ACOMPCOR',
                       'Combined Unique RETROICOR/HR/RVT Effect with AROMA and ACOMPCOR'
                       'Unique AROMA Effect with RETROICOR/HR/RVT and ACOMPCOR',
                       'Unique ACOMPCOR Effect with RETROICOR/HR/RVT and AROMA',
                       'Shared Effect of RETROICOR/HR/RVT, AROMA and ACOMPCOR']


# Create lists of to be compared runs
counter = contrast_list.index(contrast)
design_matrix.columns = [contrast_list_names[counter]]
image_list = []
for sub_id in subject_list:
    image_list.append(glob.glob(f'{basepath}{sub_id}/glm_output/{contrast}.nii.gz')[0])
second_level_input = image_list

# Visualise and save second level design matrix
plot_design_matrix(design_matrix,
                   rescale=False,
                   output_file=f'{basepath}/second_level/'
                               f'{contrast.split("/")[1]}_design_mat.png')

nr_permutations = 6000
print(f'Running non-parametric permutation test for {contrast_list_names[counter]}...')
out_dict = non_parametric_inference(
    second_level_input,
    second_level_contrast=[1],
    design_matrix=design_matrix,
    model_intercept=True,
    n_perm=nr_permutations,
    smoothing_fwhm=6.0,
    two_sided_test=False,
    n_jobs=-1,
    threshold=0.05,
    tfce=True
)

threshold = 1  # p < 0.1
vmax = -np.log10(1 / 10000)

for i in out_dict.keys():
    nib.save(out_dict[i], f'/project/3013068.03/physio_revision/GLM_approach/second_level/'
                          f'{contrast}_{i}.nii.gz')

print(f'Output saved for non-parametric permutation test for {contrast_list_names[counter]}...')

gc.collect()

