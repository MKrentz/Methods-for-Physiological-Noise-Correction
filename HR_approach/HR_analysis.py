from scipy import io
import numpy as np
import glob
import pandas as pd

BASEPATH = '/project/3013068.03/RETROICOR/TSNR/'

# Load all available participants
part_list = glob.glob(BASEPATH + 'sub-*')
part_list.sort()

subs = []
for x in part_list:
    subs.append(x[-7:])

# Indicating subject having the 'stress' condition during their FIRST functional session
stress_list = ['sub-002', 'sub-003', 'sub-004', 'sub-007', 'sub-009', 'sub-013', 'sub-015', 'sub-017', 'sub-021', 'sub-023', 'sub-025', 'sub-027', 'sub-029']
rejection_df = pd.DataFrame(index=subs, columns=['HR Rejection Percentage', 'Session Number'])

for subject_long in part_list:
    # Subject space_identifier
    sub_id = subject_long[-7:]
    if sub_id != 'sub-017':
        # Account for balancing in stress/control session order
        ses_nr = 3 if sub_id in stress_list else 2

        try:
            hera = io.loadmat(glob.glob('/project/3013068.03/stats/HR_processing/{0}/ses-0{1}/sub_{2}_0{1}*run_4*hera.mat'.format(sub_id, str(ses_nr), sub_id[-3:]))[0])
        except:
            hera = io.loadmat(glob.glob('/project/3013068.03/stats/HR_processing/{0}/ses-0{1}/{0}*ses-0{1}*RS*run-2*hera.mat'.format(sub_id, str(ses_nr)))[0])

        hera_data = hera['matfile']

        rejections = []

        for counter, object in enumerate(hera_data[0][0][7][0]):
            rejections.append(object[0][1] - object[0][0])
            #else:
            #    null_intervals.append(object[0][1] - object[0][0])

        last_peak = float(hera_data[0][0][4][0][-1:])
        rejection_duration = np.sum(rejections)
        percentage_rejection = rejection_duration / last_peak * 100
        rejection_df['HR Rejection Percentage'][sub_id] = percentage_rejection
        rejection_df['Session Number'][sub_id] = ses_nr