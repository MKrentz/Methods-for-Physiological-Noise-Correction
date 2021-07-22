#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 12:02:43 2021

@author: markre
"""

import glob


part_list = glob.glob('/project/3013068.03/RETROICOR/Example_Visualisation/sub-*') 
part_list.sort()

## Currently_adjusted

for par in part_list:
    feat_file = open("/project/3013068.03/RETROICOR/Example_Visualisation/{}/RETRO_vs_AROMA.fsf".format(par[-7:]), "r")
    feat_lines = feat_file.readlines()	
    for line_counter, lines in enumerate(feat_lines):
        if 'set fmri(confoundevs)' in lines:
            feat_lines[line_counter] = 'set fmri(confoundevs) 1\n\nset confoundev_files(1) "/project/3013068.03/RETROICOR/Example_Visualisation/{0}/3C4R1M_vs_AROMA/Resting_State_confounds_all_{0}_ses-03_run-02_nonaggrAROMA.txt"\n'.format(par[-7:])
        if 'set fmri(outputdir)' in lines:
            feat_lines[line_counter] = 'set fmri(outputdir) "/project/3013068.03/RETROICOR/Example_Visualisation/{0}/RETRO_vs_nonaggrAROMA.feat"\n'.format(par[-7:])
    new_file = open("/project/3013068.03/RETROICOR/Example_Visualisation/{0}/RETRO_vs_nonaggrAROMA.fsf".format(par[-7:]), "w")
    new_file.writelines(feat_lines)
    new_file.close()
    feat_file.close()
