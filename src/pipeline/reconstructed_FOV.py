#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Melisa
Created on Wed 26 Aug 2020
"""

import os
import sys
import numpy as np
import pickle

# This should be in another file. Let's leave it here for now
sys.path.append('/home/sebastian/Documents/Melisa/calcium_imaging_analysis/src/')
sys.path.remove('/home/sebastian/Documents/calcium_imaging_analysis')

import src.configuration
import src.paths as paths
import caiman as cm
from caiman.base.rois import com
import src.data_base_manipulation as db
from caiman.source_extraction.cnmf.cnmf import load_CNMF
import src.steps.normalized_traces as normalization


# Paths
analysis_states_database_path = paths.analysis_states_database_path
backup_path = 'references/analysis/backup/'
states_df = db.open_analysis_states_database(path = analysis_states_database_path)

mouse_number = 32365
sessions = [2,3]
is_rest = None

decoding_version = 1
motion_correction = 20
alignment_version = 3
equalization_version = 0
source_extraction_version = 1
component_evaluation_version = 1
cropping_number = [1,2,3,4]

#day_wise for registration version = 0 (do a day by day analysis)

concateneted_files_dir = os.environ['DATA_DIR'] + 'data/interim/reconstruction/day_wise/'
time_sf = 10

for session in sessions:
    for trial in [1,6,11,16,21]:
        calcium_trace = []
        calcium_trace_shape = []
        for cropping_version in cropping_number:
            selected_rows = db.select(states_df, 'registration', mouse=mouse_number, session=session, trial = trial, is_rest=is_rest,
                                      decoding_v=decoding_version,
                                      cropping_v=cropping_version,
                                      motion_correction_v=motion_correction,
                                      alignment_v=alignment_version,
                                      equalization_v=equalization_version,
                                      source_extraction_v=source_extraction_version,
                                      component_evaluation_v= component_evaluation_version)

            row = selected_rows.iloc[0]
            component_evaluation_hdf5_file_path = eval(row['component_evaluation_output'])['main']
            #registration_time_order = eval(row['component_evaluation_parameters'])['trials_order']
            timeline_file = eval(row['alignment_output'])['meta']['timeline']
            timeline_data = pickle.load(open(timeline_file, "rb"))
            #cnm_result = pickle.load( open(component_evaluation_output, "rb" ))
            cnm= load_CNMF(component_evaluation_hdf5_file_path)
            if cnm.estimates.bl is None:
                raw_normed, cnm_normed, res_normed, s_normed, noise_levels = normalization.normalize_traces(
                    cnm.estimates.C,
                    cnm.estimates.YrA,
                    cnm.estimates.S,
                    1,
                    offset_method="denoised_floor")
            else:
                raw_normed, cnm_normed, res_normed, s_normed, noise_levels = normalization.normalize_traces(
                    cnm.estimates.C - cnm.estimates.bl[:, np.newaxis],
                    cnm.estimates.YrA,
                    cnm.estimates.S,
                    1,
                    offset_method="denoised_floor")

            #cnm_result = cnm.estimates.C[cnm.estimates.idx_components, :]
            cnm_result = cnm_normed[cnm.estimates.idx_components, :]
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
            calcium_trace.append(cnm_result)
            calcium_trace_shape.append(cnm_result.shape)

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
