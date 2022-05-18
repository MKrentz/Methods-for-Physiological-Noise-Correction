#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun 15 08 11:56:23 2021

@author: markre
"""

import os
import glob

BASEPATH = '/project/3013068.03/test/'
sub_list = ['sub-001', 'sub-002','sub-003','sub-004','sub-005', 'sub-006', 'sub-007', 'sub-008', 'sub-009', 'sub-010',
            'sub-012', 'sub-013', 'sub-014', 'sub-015', 'sub-016', 'sub-017', 'sub-018', 'sub-020', 'sub-021', 'sub-022',
            'sub-023', 'sub-024', 'sub-025', 'sub-027', 'sub-028', 'sub-029', 'sub-030']
glm_list = ['glm1_retro', 'glm2_aroma', 'glm3_acompcor', 'glm4_aroma_acompcor', 'glm5_retro_aroma', 'glm6_retro_aroma_acompcor']


if not glob.glob(BASEPATH):
    os.mkdir(BASEPATH)
    os.mkdir(os.path.join(BASEPATH, 'GLM_approach'))
    os.mkdir(os.path.join(BASEPATH, 'TSNR_approach'))
    os.mkdir(os.path.join(BASEPATH, 'HR_approach'))
    for sub in sub_list:
        os.mkdir(os.path.join(BASEPATH, 'GLM_approach', sub))
        os.mkdir(os.path.join(BASEPATH, 'GLM_approach', sub, 'confounds'))
        os.mkdir(os.path.join(BASEPATH, 'GLM_approach', sub, 'glm_output'))
        for glms in glm_list:
            os.mkdir(os.path.join(BASEPATH, 'GLM_approach', sub, 'glm_output', glms))
        os.mkdir(os.path.join(BASEPATH, 'GLM_approach', sub, 'melodic_glms_output'))
        os.mkdir(os.path.join(BASEPATH, 'GLM_approach', sub, 'melodic_misclassifications'))
        os.mkdir(os.path.join(BASEPATH, 'TSNR_approach', sub))