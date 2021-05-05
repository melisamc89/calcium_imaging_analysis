#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 11:50:40 2019

@author: Melisa
"""

import os
# Define the steps in the pipeline (in order)
steps = [
        'decoding', 
        'cropping', # spatial borders that are unusable (due to microenscope border 
        # or blood clot) are removed
        'motion_correction', # individual trial movies (5 min) are rigidly or 
        # piecewise rigidly motion corrected
        'alignment', # Multiple videos (e.g. all trials of a session, 210 min) are
        # rigid motion corrected to each other, resulting in a long aligned video
        'equalization',
        'source_extraction', # neural activity is deconvolved from the videos
        # trial-wise or session-wise
        'component_evaluation',
        'registration',
        'model'
        ]

# Paths
#for running the pipeline
#analysis_states_database_path = os.environ['PROJECT_DIR'] + 'references/analysis/calcium_imaging_data_base_trial_wise_analysis.xlsx'
# Paths
#for running the pipeline day based
#analysis_states_database_path = os.environ['PROJECT_DIR'] + 'references/analysis/calcium_imaging_data_base_day_wise_analysis.xlsx'
#for running day wise 2021
#analysis_states_database_path = os.environ['PROJECT_DIR'] + 'references/analysis/calcium_data_base_2021.xlsx'
analysis_states_database_path = '/scratch/melisa/calcium_imaging_analysis/references/analysis/calcium_data_base_2021.xlsx'
##for parameters selection
#analysis_states_database_path = os.environ['PROJECT_DIR'] + 'references/analysis/data_base_templete_control3.xlsx'
#for alejandra alonsos data
#analysis_states_database_path = os.environ['PROJECT_DIR'] + 'references/analysis/calcium_imaging_data_base_alonso2.xlsx'
#analysis_states_database_path = os.environ['PROJECT_DIR'] + 'references/analysis/calcium_imaging_data_base_alonso2_day_wise.xlsx'

### auxiliary for neuro_course
#analysis_states_database_path = os.environ['PROJECT_DIR'] + 'references/analysis/auxiliary_data_base.xlsx'

backup_path = os.environ['PROJECT_DIR']  + 'references/analysis/backup/'
parameters_path = os.environ['PROJECT_DIR']  + 'references/analysis/parameters_database.xlsx'

# Multi Index Structure
data_structure = ['mouse', 'session', 'trial', 'is_rest']
analysis_structure = [f'{step}_v' for step in steps]
multi_index_structure = data_structure + analysis_structure

# Columns
columns = data_structure + ['experiment_parameters',
            'experiment_comments',
            'raw_output',
            'raw_comments']
# for each step, add a 'v' (version), 'parameters', 'output' and 'comments' columns
for step in steps:
    columns += [f'{step}_{idx}' for idx in ['v','parameters','output','comments']]
