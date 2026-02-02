"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
@author: markre
"""

import glob
import numpy as np
import nibabel as nib
import pandas as pd
from nilearn import plotting
import matplotlib.pyplot as plt
import scipy
import seaborn as sns

# Path to data
BASEPATH = '/project/3013068.03/physio_revision_fixed/GLM_approach/'

#Path to FDR corrected model of unique RETROICOR variance
participant_list = glob.glob(BASEPATH + 'sub-*/glm_output/glm5_retro_aroma/'
                             'unique_retro_z_score_fdr_corrected.nii.gz')
part_list = glob.glob(BASEPATH + 'sub-*')
part_list.sort()

count_df = pd.DataFrame(index=[sub[sub.find('sub-'):sub.find('sub-') + 7] for sub in participant_list],
                       columns=['z_mean'])
sum_df = pd.DataFrame(index=[sub[sub.find('sub-'):sub.find('sub-') + 7] for sub in participant_list],
                       columns=['z_mean'])
mean_df = pd.DataFrame(index=[sub[sub.find('sub-'):sub.find('sub-') + 7] for sub in participant_list],
                       columns=['z_mean'])

for sub in participant_list:
    sub_id = sub[sub.find('sub-'):sub.find('sub-') + 7]
    z_map = nib.load(sub).get_fdata()
    count_df.loc[sub_id] = len(z_map[z_map > 0])
    sum_df.loc[sub_id] = z_map[z_map > 0].sum()
    mean_df.loc[sub_id] = z_map.mean()

total_df = pd.DataFrame(columns=['Subject', 'Goodness of Fit', 'Component Number'])
for subs in part_list:
    sub_id = subs[-7:]
    try:
        summary = glob.glob(BASEPATH + f'{sub_id}/melodic_glms_output/*summary.txt')[0]
    except:
        continue
    fit_matrix = pd.read_csv(summary, index_col=0)
    fit_matrix.sort_values('Goodness of Fit', ascending=False, kind='stable', inplace=True)
    for row in fit_matrix.iterrows():
        if row[1]['Goodness of Fit'] > 0.75 and row[1]['Component Classification'] == 'Signal':
            total_df = pd.concat([total_df, pd.DataFrame([[sub_id, row[1]['Goodness of Fit'], row[0] + 1]],
                                                         columns=['Subject', 'Goodness of Fit', 'Component Number'])],
                                 ignore_index=True)

z_change = pd.read_csv(BASEPATH + 'misclassification_zchange_overview.txt')
z_change_median = z_change.groupby('Subject').median()['Z Change']
z_change_median.drop(index=['sub-012', 'sub-029'], inplace=True)
#z_change_median.drop(index=['sub-012'], inplace=True)
z_change_sum = z_change.groupby('Subject').sum()['Z Change']
z_change_sum.drop(index=['sub-012', 'sub-029'], inplace=True)
#z_change_sum.drop(index=['sub-012'], inplace=True)

subjects_not_in_z_change = sum_df.index.difference(z_change['Subject'])
sum_df.drop(index=subjects_not_in_z_change, inplace=True)
sum_df.drop(index=['sub-012', 'sub-029'], inplace=True)
#sum_df.drop(index=['sub-012'], inplace=True)

sum_df = sum_df.astype('float')
sum_df = sum_df.sort_index()

count_df.drop(index=subjects_not_in_z_change, inplace=True)
count_df.drop(index=['sub-012', 'sub-029'], inplace=True)
#count_df.drop(index=['sub-012'], inplace=True)

count_df = count_df.astype('float')
count_df = count_df.sort_index()

mean_df.drop(index=subjects_not_in_z_change,inplace=True)
mean_df.drop(index=['sub-012', 'sub-029'], inplace=True)
#mean_df.drop(index=['sub-012'], inplace=True)

mean_df = mean_df.sort_index()
mean_df = count_df.astype('float')

total_df.rename(columns={'Component Number': 'Melodic Component'}, inplace=True)
z_change.sort_values(['Subject', 'Melodic Component'], inplace=True)
total_df.sort_values(['Subject', 'Melodic Component'], inplace=True)

z_change = pd.merge(
    z_change,
    total_df[['Subject', 'Melodic Component', 'Goodness of Fit']],
    on=['Subject', 'Melodic Component'],
    how='left'
)

correlation_dic = {
    'Goodness of Fit with Voxel Count': scipy.stats.spearmanr(z_change['Goodness of Fit'], z_change['Voxel Number'], nan_policy='omit'),
    'Goodness of Fit with Z Change': scipy.stats.spearmanr(z_change['Goodness of Fit'], z_change['Z Change'], nan_policy='omit'),
    'Count of Voxels with Median Z Change': scipy.stats.spearmanr(count_df, z_change_median, nan_policy='omit'),
    'Count of Voxels with Median Z Change Pearson': scipy.stats.pearsonr(count_df['z_mean'], z_change_median),
    'Count of Voxels with Sum Z Change': scipy.stats.spearmanr(count_df, z_change_sum, nan_policy='omit'),
    'Count of Voxels with Sum Z Change Pearson': scipy.stats.pearsonr(count_df['z_mean'], z_change_sum)
}


x_temp = 'Count of Suprathreshold RETROICOR Voxels per Subject'
y_temp = 'Sum of Z-Change per Subject'

ax = sns.regplot(x=count_df,
                 y=z_change_sum,
                 robust=True,
                 line_kws={'label': f"r = {correlation_dic['Count of Voxels with Sum Z Change'][0]: .3f},"
                                    f" p = {correlation_dic['Count of Voxels with Sum Z Change'][1]: .3f}"})
ax.set_xlabel(x_temp)
ax.set_ylabel(y_temp)
ax.set_title('Impact of Misclassification ')
#ax.legend()

plt.savefig(f'{BASEPATH}correlation_misclassification_RETRO.svg')
plt.close()

import glob
import numpy as np
import nibabel as nib
import pandas as pd
from nilearn import plotting
import matplotlib.pyplot as plt
import scipy
import seaborn as sns

BASEPATH = '/project/3013068.03/physio_revision_fixed/GLM_approach/'

# Load subject-level results from unified model
participant_list = glob.glob(BASEPATH + 'sub-*/full_model_RETROICOR_FWE.nii.gz')
sub_ids = [sub.split('/')[-2] for sub in participant_list]  # Extract subject IDs

# Initialize DataFrames
results = pd.DataFrame(index=sub_ids, columns=['voxel_count', 'z_sum', 'z_mean', 'z_change'])

# Load unified model results
for sub_path in participant_list:
    sub_id = sub_path.split('/')[-2]

    # Load thresholded Z-map
    z_map = nib.load(sub_path).get_fdata()
    mask = z_map != 0  # Non-zero voxels after thresholding

    # Store metrics
    results.loc[sub_id, 'voxel_count'] = mask.sum()
    results.loc[sub_id, 'z_sum'] = z_map[mask].sum()
    results.loc[sub_id, 'z_mean'] = z_map[mask].mean()

    # Load direct Z-change from unified model (single value per subject)
    z_change_path = f"{BASEPATH}{sub_id}/full_model_z_change.txt"
    results.loc[sub_id, 'z_change'] = pd.read_csv(z_change_path, header=None).values[0][0]

# Clean data (adapt exclusion criteria as needed)
results = results.dropna()
results = results.astype(float)
results = results.sort_index()

# Calculate correlations
correlations = {
    'Voxel Count vs Z-Change (Spearman)': scipy.stats.spearmanr(results['voxel_count'], results['z_change']),
    'Voxel Count vs Z-Change (Pearson)': scipy.stats.pearsonr(results['voxel_count'], results['z_change']),
    'Z-Sum vs Z-Change (Spearman)': scipy.stats.spearmanr(results['z_sum'], results['z_change'])
}

# Visualization
plt.figure(figsize=(10, 6))
ax = sns.regplot(x='voxel_count', y='z_change', data=results,
                 scatter_kws={'s': 100, 'alpha': 0.7},
                 line_kws={'color': 'red', 'lw': 2})

ax.set(xlabel='Count of Significant RETROICOR Voxels',
       ylabel='Unified Model Z-Change',
       title='Impact of Component Misclassification (Unified Model)')

# Add correlation annotation
corr_coef, p_val = correlations['Voxel Count vs Z-Change (Spearman)'][:]
plt.text(0.05, 0.9, f'Spearman ρ = {corr_coef:.2f}\np = {p_val:.3f}',
         transform=ax.transAxes)

plt.savefig(f'{BASEPATH}unified_model_correlation.svg', bbox_inches='tight')
plt.close()