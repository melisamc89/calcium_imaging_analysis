#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Melisa
Created on Tue Jan 28 12.00.00 2020
"""

import os
import sys
import numpy as np
import pandas as pd
import pickle
import math
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt

import src.configuration
import src.paths as paths
import caiman as cm
from caiman.base.rois import com
import src.data_base_manipulation as db
from caiman.source_extraction.cnmf.cnmf import load_CNMF
import src.analysis.figures as figures
from caiman.base.rois import register_multisession
from src.steps.registering import run_registration as main_registration
import src.steps.normalized_traces as normalization

# Paths
analysis_states_database_path = paths.analysis_states_database_path
backup_path = 'references/analysis/backup/'
states_df = db.open_analysis_states_database(path = analysis_states_database_path)

mouse_number = 56165
sessions = [1,2,4]
is_rest = None

decoding_version = 1
motion_correction = 20
alignment_version = 3
equalization_version = 0
source_extraction_version = 1
component_evaluation_version = 1
cropping_number = [1,2,3,4]

for session in sessions:
    for cropping_version in cropping_number:
        selected_rows = db.select(states_df, 'registration', mouse=mouse_number, session=session, is_rest=is_rest,
                                  decoding_v=decoding_version,
                                  cropping_v=cropping_version,
                                  motion_correction_v=motion_correction,
                                  alignment_v=alignment_version,
                                  equalization_v=equalization_version,
                                  source_extraction_v=source_extraction_version,
                                  component_evaluation_v= component_evaluation_version)
        parameters_registration = {'session_wise': False, 'model_method': False, 'cost_threshold': 0.9, 'max_dist': 15,
                      'min_cell_size': 10, 'max_cell_size': 25, 'scramble': False, 'normalization': True, 'day_wise' : True}
        if parameters_registration['scramble']:
            shuffle_selected_rows = selected_rows.sample(frac = 1)
        else:
            shuffle_selected_rows = selected_rows.copy()
        new_selected_rows = main_registration(shuffle_selected_rows, parameters_registration)
        states_df = db.append_to_or_merge_with_states_df(states_df, new_selected_rows)
        db.save_analysis_states_database(states_df, analysis_states_database_path, backup_path)


concateneted_files_dir = os.environ['DATA_DIR'] + 'data/interim/reconstruction/trial_wise/'
time_sf = 10
registration_version = 2
for session in sessions:
    calcium_trace = []
    calcium_trace_shape = []
    for cropping_version in cropping_number:
        selected_rows = db.select(states_df, 'registration', mouse=mouse_number, session=session, is_rest=is_rest,
                                  decoding_v=decoding_version,
                                  cropping_v=cropping_version,
                                  motion_correction_v=motion_correction,
                                  alignment_v=alignment_version,
                                  equalization_v=equalization_version,
                                  source_extraction_v=source_extraction_version,
                                  component_evaluation_v= component_evaluation_version,
                                  registration_v= registration_version)

        row = selected_rows.iloc[0]
        registration_output = eval(row['registration_output'])['main']
        registration_time_order = eval(row['registration_parameters'])['trials_order']
        timeline_file = eval(row['alignment_output'])['meta']['timeline']
        timeline_data = pickle.load(open(timeline_file, "rb"))
        cnm_result = pickle.load( open(registration_output, "rb" ))
        #reorder_cnm_result = np.zeros_like(cnm_result.C)

        #timeline = []
        #for i in range(len(timeline_data)):
        #    timeline.append(timeline_data[i][1])
        #timeline.append(cnm_result.C.shape[1])
        #time_length = np.diff(timeline)
        #for i in range(len(registration_time_order)):
        #    auxiliar = cnm_result.C[:, timeline[i]: timeline[i] +time_length[registration_time_order[i]-1]]
        #    reorder_cnm_result[:,timeline[registration_time_order[i]-1]:timeline[registration_time_order[i]-1]+auxiliar.shape[1]] = auxiliar

        #calcium_trace.append(reorder_cnm_result)
        calcium_trace.append(cnm_result.C)
        calcium_trace_shape.append(cnm_result.C.shape)

    time = np.arange(0,(calcium_trace_shape[0])[1])/time_sf
    n_neurons = 0
    for i in range(len(cropping_number)):
        n_neurons = n_neurons + (calcium_trace_shape[i])[0]

    min_time = 5000000000
    for i in range(len(calcium_trace_shape)):
        if calcium_trace_shape[i][1] < min_time:
            min_time = calcium_trace_shape[i][1]

    activity_matrix = np.zeros((n_neurons+1,min_time))
    activity_matrix[0,:] = time[:min_time]
    init = 1
    finish = (calcium_trace_shape[0])[0]+1
    for i in range(len(calcium_trace)-1):
        activity_matrix[init:finish,:] = calcium_trace[i][:,:min_time]
        init = init + (calcium_trace_shape[i])[0]
        finish = finish + (calcium_trace_shape[i+1])[0]
    activity_matrix[init:finish,:] = calcium_trace[len(cropping_number)-1][:,:min_time]
    output_file_name = db.create_file_name(7, row.name)
    np.save(concateneted_files_dir + output_file_name, activity_matrix)

