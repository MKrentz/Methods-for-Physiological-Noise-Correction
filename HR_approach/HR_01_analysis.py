"""
Created on Thu Aug 12 10:00:00 2021
This script calculates the total duration of heart rate data points removed 
during the cleaning process in HERA. It specifically addresses the issue of 
overlapping rejection windows to ensure the final percentage is accurate.
"""

from scipy import io
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Define paths for the specific cluster environment
BASEPATH = '/project/3013068.03/RETROICOR/TSNR/'
SAVEPATH = '/project/3013068.03/RETROICOR/HR_approach/'

# Generate subject list from the existing directory structure
part_dirs = sorted(glob.glob(BASEPATH + 'sub-*'))
subs = [p[-7:] for p in part_dirs]

# Counterbalancing list: subjects who had the 'stress' condition in their first session
stress_first_list = [
    'sub-002', 'sub-003', 'sub-004', 'sub-007', 'sub-009', 'sub-013', 
    'sub-015', 'sub-017', 'sub-021', 'sub-023', 'sub-025', 'sub-027', 'sub-029'
]

# Prepare results container
rejection_df = pd.DataFrame(index=subs, columns=['HR Rejection Percentage', 'Session Number', 'Overlap'])

for subject_dir in part_dirs:
    sub_id = subject_dir[-7:]
    
    # Identify the correct session number based on condition balancing
    # Stress session is 3 for the 'stress_first' group, otherwise session 2
    session_nr = 3 if sub_id in stress_first_list else 2

    # HERA output naming can vary; we attempt both known patterns
    try:
        pattern1 = '/project/3013068.03/stats/HR_processing/{0}/ses-0{1}/sub_{2}_0{1}*run_4*hera.mat'.format(sub_id, str(session_nr), sub_id[-3:])
        hera_file = glob.glob(pattern1)[0]
        hera = io.loadmat(hera_file)
    except IndexError:
        # Fallback to alternative naming convention (e.g., run-2)
        pattern2 = '/project/3013068.03/stats/HR_processing/{0}/ses-0{1}/{0}*ses-0{1}*RS*run-2*hera.mat'.format(sub_id, str(session_nr))
        hera_file = glob.glob(pattern2)[0]
        hera = io.loadmat(hera_file)

    # Access the MATLAB data structure
    hera_mat = hera['matfile']
    
    # Extract the timing pairs for rejected segments
    # The structure follows: [7] for rejection windows, [0] for the data array
    raw_intervals = []
    for entry in hera_mat[0][0][7][0]:
        start_time, end_time = entry[0][0], entry[0][1]
        if end_time > start_time:
            raw_intervals.append([start_time, end_time])

    # Algorithm to merge overlapping intervals to prevent double-counting
    rejection_seconds = 0
    overlap_detected = 'No'
    
    if raw_intervals:
        # Sort intervals by start time for the merging algorithm
        raw_intervals.sort(key=lambda x: x[0])
        merged_intervals = [raw_intervals[0]]
        
        for current in raw_intervals[1:]:
            prev_start, prev_end = merged_intervals[-1]
            curr_start, curr_end = current
            
            if curr_start < prev_end:
                # If current start is before previous end, windows overlap
                overlap_detected = 'Yes'
                # Update the previous end time to the maximum of both
                merged_intervals[-1][1] = max(prev_end, curr_end)
            else:
                merged_intervals.append(current)
        
        rejection_seconds = sum(m[1] - m[0] for m in merged_intervals)

    # Total session length is derived from the final R-peak timestamp
    total_duration = float(hera_mat[0][0][4][0][-1:])
    perc_rejected = (rejection_seconds / total_duration) * 100

    # Fill the dataframe using .at for efficient scalar assignment
    rejection_df.at[sub_id, 'HR Rejection Percentage'] = perc_rejected
    rejection_df.at[sub_id, 'Session Number'] = session_nr
    rejection_df.at[sub_id, 'Overlap'] = overlap_detected

# Save the final table
rejection_df.to_csv(SAVEPATH + 'HR_rejections.tsv', sep='\t')

# Visualization of rejection distribution
fig, ax = plt.subplots(figsize=(5, 7))
# Ensure data is numeric for plotting
numeric_data = pd.to_numeric(rejection_df['HR Rejection Percentage'])

# Generate boxplot and identify outliers (fliers)
bp = ax.boxplot(numeric_data, widths=0.4, patch_artist=True,
                boxprops=dict(facecolor='#d1e5f0', color='#000000'))

# Label outliers with subject ID for quality control review
outlier_values = bp['fliers'][0].get_ydata()
for val in outlier_values:
    # Retrieve the subject ID associated with the outlier value
    subj_name = rejection_df[rejection_df['HR Rejection Percentage'] == val].index[0]
    ax.text(1.1, val, f'{subj_name} ({val:.1f}%)', va='center', color='red', fontsize=9)

# Plot aesthetics
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_ylabel('Data Rejected (%)', fontsize=10)
ax.set_title('Heart Rate Cleaning: Percentage Rejected', fontsize=12, pad=15)
plt.xticks([])
plt.tight_layout()
plt.savefig(SAVEPATH + 'HR_rejections_boxplot.png', dpi=300)
