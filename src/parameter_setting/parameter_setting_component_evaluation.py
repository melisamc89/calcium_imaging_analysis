#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 09:19:20 2019

@author: Melisa
"""

import os
import sys
import psutil
import logging
import datetime
import numpy as np
import pylab as pl
import pandas as pd
# This should be in another file. Let's leave it here for now


import matplotlib.pyplot as plt
import src.configuration
import caiman as cm
import src.data_base_manipulation as db
from src.steps.decoding import run_decoder as main_decoding
from src.steps.cropping import run_cropper as main_cropping
from src.steps.cropping import cropping_interval
from src.analysis.figures import plot_movie_frame, plot_movie_frame_cropped, get_fig_gSig_filt_vals

from src.steps.motion_correction import run_motion_correction as main_motion_correction
from src.steps.source_extraction import run_source_extraction as main_source_extraction
from src.steps.component_evaluation import run_component_evaluation as main_component_evaluation
import src.analysis_files_manipulation as fm
import src.analysis.metrics as metrics
from caiman.source_extraction.cnmf.cnmf import load_CNMF
import src.analysis.figures as figures
import src.analysis.metrics as metrics

# Paths
analysis_states_database_path = os.environ['PROJECT_DIR'] + 'references/analysis/calcium_imaging_data_base_server_new.xlsx'
backup_path = 'references/analysis/backup/'

states_df = db.open_analysis_states_database()
mouse = 51565
session = 1
trial = 1
is_rest = None
cropping_version = 1
motion_correction_version = 1
source_extraction_version =1


#%% COMPONENT EVALUATION RUNNING ON MULTIPLE VERSIONS AND PARAMETERS

source_extraction_v_array = np.arange(1,30) # (this par shoul related to the length of the general selection of source extracted
#files)

min_SNR_array =np.arange(1,7,1)
r_values_min_array = [0.75,0.80,0.85,0.90,0.95]

#%%
for kk in range(len(source_extraction_v_array)):
    for ii in range(len(min_SNR_array)):
        for jj in range(len(r_values_min_array)):
            min_SNR = min_SNR_array[ii]
            r_values_min = r_values_min_array[jj]
            selected_row= db.select(states_df, 'component_evaluation', mouse = mouse, session = session, trial = trial,
                                    is_rest = is_rest, cropping_v=cropping_version, motion_correction_v=motion_correction_version,
                                 source_extraction_v= source_extraction_v_array[kk])
            mouse_row = selected_row.iloc[0]
            parameters_component_evaluation = {'min_SNR': min_SNR,
                                               'rval_thr': r_values_min,
                                               'use_cnn': False}
            mouse_row_new = main_component_evaluation(mouse_row, parameters_component_evaluation)
            states_df = db.append_to_or_merge_with_states_df(states_df, mouse_row_new)
            db.save_analysis_states_database(states_df, path=analysis_states_database_path, backup_path = backup_path)



#%% Plotting the result of selected and unselected cells for each parameter selection (contours and traces)

component_evaluation_v_array = np.arange(0,len(min_SNR_array)*len(r_values_min_array))

for kk in range(1,len(source_extraction_v_array)):
    for ll in range(len(component_evaluation_v_array)):
        selected_row = db.select(states_df, 'component_evaluation', mouse = mouse, session = session, trial = trial,
                                 is_rest = is_rest, cropping_v=cropping_version,
                                 motion_correction_v=motion_correction_version,
                                 source_extraction_v= source_extraction_v_array[kk],
                                 component_evaluation_v = component_evaluation_v_array[ll])
        mouse_row = selected_row.iloc[0]
        figures.plot_contours_evaluated(mouse_row)
        figures.plot_traces_multiple_evaluated(mouse_row)

