import nibabel as nib
from Subject_Class import Subject
from nilearn.image import resample_to_img
import glob
import numpy as np

BASEPATH = '/project/3013068.03/test/TSNR_approach/'

part_list = glob.glob(BASEPATH + 'sub-*')
part_list.sort()

#Mean Matrices for LC
stress_list = ['sub-002', 'sub-003', 'sub-004', 'sub-007', 'sub-009', 'sub-013', 'sub-015', 'sub-017', 'sub-021', 'sub-023', 'sub-025', 'sub-027', 'sub-029']

for subject_path in part_list:
    sub_id = subject_path[-7:]
    sub_obj = Subject(sub_id)

    # Account for balancing in stress/control session order
    ses_nr = 2 if sub_id in stress_list else 1
    sub_func = sub_obj.get_func_data(run=2, session=ses_nr)

    try:
        # LC mask resampling and binarising
        LC_mask = sub_obj.get_LC()
        LC_mask_native = resample_to_img(LC_mask, sub_func, interpolation='nearest')
        LC_mask_mat = LC_mask_native.get_fdata()
        LC_mask_mat = np.where((LC_mask_mat == 0) | (LC_mask_mat == 1), 1 - LC_mask_mat, LC_mask_mat)
        LC_mask_nii = nib.save(nib.Nifti2Image(LC_mask_mat, LC_mask.affine, LC_mask.header), BASEPATH + '{0}/masks/LC_mask_native.nii.gz'.format(sub_id))

    except:
        print('No LC mask for {}'.format(sub_id))

    # Load in, resample and binarise GM mask
    gm_mask = nib.load('/project/3013068.03/derivate/fmriprep/{0}/anat/{0}_desc-aparcaseg_dseg.nii.gz'.format(sub_id))
    gm_mask_mat = gm_mask.get_fdata()
    gm_mask_mat[gm_mask_mat < 1000], gm_mask_mat[gm_mask_mat >= 1000] = 1, 0
    gm_mask_native = nib.Nifti2Image(gm_mask_mat, gm_mask.affine, gm_mask.header)
    gm_mask_native = resample_to_img(gm_mask_native, sub_func, interpolation='nearest')
    nib.save(gm_mask_native, BASEPATH + '{0}/masks/gm_mask_native.nii.gz'.format(sub_id))
