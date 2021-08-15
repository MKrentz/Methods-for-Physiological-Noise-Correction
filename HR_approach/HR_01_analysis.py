#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 10:00:00 2021

This script implements the calculation of removed datapoints during cleaning of heart rate data.
The implementation here IS convoluted, as a consequence of HERA (the program used for data cleaning) allowing for overlapping
windows of rejection, leading to otherwise faulty caluations.

Output:
    A dataframe showing percentage of rejection per subject as well as presence of potential rejection overlap (which is considered during calculation)

@author: MKrentz
"""

from scipy import io
import glob
import numpy as np
import pandas as pd

BASEPATH = '/project/3013068.03/RETROICOR/TSNR/'

# Load all available participants
part_list = glob.glob(BASEPATH + 'sub-*')
part_list.sort()

subs = []
for participants in part_list:
    subs.append(participants[-7:])

# Indicating subject having the 'stress' condition during their FIRST functional session
stress_list = ['sub-002', 'sub-003', 'sub-004', 'sub-007', 'sub-009', 'sub-013', 'sub-015', 'sub-017', 'sub-021', 'sub-023', 'sub-025', 'sub-027', 'sub-029']

# Create DataFrame to be filled
rejection_df = pd.DataFrame(index=subs, columns=['HR Rejection Percentage', 'Session Number', 'Overlap'])

# Subject loop
for subject_long in part_list:

    # Subject space_identifier
    sub_id = subject_long[-7:]

    # Account for balancing in stress/control session order
    ses_nr = 3 if sub_id in stress_list else 2

    # Account for different naming conventions in HR data
    try:
        hera = io.loadmat(glob.glob('/project/3013068.03/stats/HR_processing/{0}/ses-0{1}/sub_{2}_0{1}*run_4*hera.mat'.format(sub_id, str(ses_nr), sub_id[-3:]))[0])
    except:
        hera = io.loadmat(glob.glob('/project/3013068.03/stats/HR_processing/{0}/ses-0{1}/{0}*ses-0{1}*RS*run-2*hera.mat'.format(sub_id, str(ses_nr)))[0])

    hera_data = hera['matfile']

    rejections = []
    overlap = 'No'

    #Loop over all rejection windows to calculate seconds rejected
    for counter, object in enumerate(hera_data[0][0][7][0]):

        # Identifier whether a rejection window overlap exists
        overlap_object = False

        # This loop accounts for rejection overlaps
        for timing_pairs in hera_data[0][0][7][0]:
            if timing_pairs[0][0] < object[0][1] and timing_pairs[0][1] > object[0][1]:
                overlap = 'Yes'
                overlap_object = True
                break

        # Depending on overlap the values are adjusted
        if overlap_object == True:
            overlap_calc = timing_pairs[0][0] - object[0][0]
            if overlap_calc < 0:
                continue
            rejections.append(overlap_calc)

        elif overlap_object == False:
            overlap_calc = object[0][1] - object[0][0]
            if overlap_calc < 0:
                continue
            rejections.append(overlap_calc)

    # Taking the last peak timestamp to substract from and calculate percentage
    last_peak = float(hera_data[0][0][4][0][-1:])
    rejection_duration = np.sum(rejections)
    percentage_rejection = rejection_duration / last_peak * 100

    #Create DataFrame and save
    rejection_df['HR Rejection Percentage'][sub_id] = percentage_rejection
    rejection_df['Session Number'][sub_id] = ses_nr
    rejection_df['Overlap'][sub_id] = overlap
    rejection_df.to_csv(BASEPATH + '/HR_rejections.txt')