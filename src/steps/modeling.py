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
from src.steps.model_reconstruction import run_model_reconstruction as main_model
import src.steps.normalized_traces as normalization

# Paths
analysis_states_database_path = paths.analysis_states_database_path
backup_path = 'references/analysis/backup/'
states_df = db.open_analysis_states_database(path = analysis_states_database_path)

mouse_number = 56165
sessions = [1,2,4]
is_rest = None

decoding_version = 1
motion_correction = 100
alignment_version = 1
equalization_version = 0
source_extraction_version = 1
component_evaluation_version = 1
registration_version = 0
cropping_number = [1,2,3,4]

parameters_model = {'session_wise': False, 'gSig_filt': (5, 5), 'downsample_rate': 10}

for session in sessions:
    for cropping_version in cropping_number:
        selected_rows = db.select(states_df, 'registration', mouse=mouse_number, session=session, is_rest=is_rest,
                                  decoding_v=decoding_version,
                                  cropping_v=cropping_version,
                                  motion_correction_v=motion_correction,
                                  alignment_v=alignment_version,
                                  equalization_v=equalization_version,
                                  source_extraction_v=source_extraction_version,
                                  component_evaluation_v= component_evaluation_version,
                                  registration_v = registration_version)
        new_selected_rows = main_model(selected_rows, parameters_model)
        states_df = db.append_to_or_merge_with_states_df(states_df, new_selected_rows)
        db.save_analysis_states_database(states_df, analysis_states_database_path, backup_path)




concateneted_files_dir = os.environ['DATA_DIR'] + 'data/interim/reconstructed_videos/trial_wise/'
time_sf = 10
re_sf = 10

for session in sessions:
    video_list = []
    video_shape_list = []
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
        model_output = eval(row['model_output'])['main']

        model_result = cm.load(model_output)
        video_list.append(model_result)
        video_shape_list.append(model_result.shape)

    video_size = min(video_shape_list)
    #new_video_size = ( int(video_size[0]/re_sf), video_size[1]*2, video_size[2]*2)

    video_list_flatten = []
    image_size = 0
    for i in range(len(video_list)):
        video_flatten = np.zeros((video_shape_list[i][0], video_shape_list[i][1]*video_shape_list[i][2]))
        image_size = image_size + video_shape_list[i][1]*video_shape_list[i][2]
        for j in range(video_shape_list[i][0]):
                video_flatten[j,:] = video_list[i][j,:,:].flatten()
        video_flatten = video_flatten.T
        reshape_video_flatten = np.reshape(video_flatten[:, :int(int(video_size[0] / re_sf) * re_sf)],
                                             (video_flatten.shape[0], int(video_size[0]/ re_sf), re_sf))
        resample_video= np.mean(reshape_video_flatten, axis=2)
        video_list_flatten.append(resample_video)

    reconstructed_frame = np.zeros((image_size, video_list_flatten[0].shape[1]))
    initial = 0
    for i in range(len(video_list_flatten)):
        reconstructed_frame[initial:initial + video_list_flatten[i].shape[0],:] = video_list_flatten[i]
        initial = initial + video_list_flatten[i].shape[0]

    output_file_name = db.create_file_name(7, row.name)
    np.save(concateneted_files_dir + output_file_name, reconstructed_frame)


