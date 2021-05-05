#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Melisa
"""

import os
import psutil
import numpy as np

import src.configuration
import caiman as cm
import src.data_base_manipulation as db
from src.steps.decoding import run_decoder as main_decoding
from src.steps.cropping import run_cropper as main_cropping
from src.steps.equalizer import  run_equalizer as main_equalizing
from src.steps.cropping import cropping_interval, cropping_segmentation
from src.analysis.figures import plot_movie_frame
from src.steps.motion_correction import run_motion_correction as main_motion_correction
from src.steps.alignment2 import run_alignmnet as main_alignment
from src.steps.source_extraction import run_source_extraction as main_source_extraction
from src.steps.component_evaluation import run_component_evaluation as main_component_evaluation
import src.paths as paths

# Paths
analysis_states_database_path = paths.analysis_states_database_path
backup_path = os.environ['PROJECT_DIR'] + 'references/analysis/backup/'
states_df = db.open_analysis_states_database(path=analysis_states_database_path)

mouse_number = 401714
sessions = [2]
init_trial = 1
end_trial = 22
is_rest = None

#  Select first data
selected_rows = db.select(states_df, 'decoding', mouse=mouse_number, is_rest=is_rest, decoding_v= 1)
mouse_row = selected_rows.iloc[0]
#mouse_row = main_decoding(mouse_row)
plot_movie_frame(mouse_row)

# select cropping parameters
parameters_cropping = cropping_interval()  # check whether it is better to do it like this or to use the functions get
# and set parameters from the data_base_manipulation file
parameters_cropping['segmentation'] = False
parameters_cropping_list = cropping_segmentation(parameters_cropping)

parameters_motion_correction = {'motion_correct': True, 'pw_rigid': True, 'save_movie_rig': False,
                                'gSig_filt': (5, 5), 'max_shifts': (25, 25), 'niter_rig': 1,
                                'strides': (48, 48),
                                'overlaps': (96, 96), 'upsample_factor_grid': 2, 'num_frames_split': 80,
                                'max_deviation_rigid': 15,
                                'shifts_opencv': True, 'use_cuda': False, 'nonneg_movie': True, 'border_nan': 'copy'}

parameters_alignment = {'make_template_from_trial': '1', 'gSig_filt': (5, 5), 'max_shifts': (25, 25), 'niter_rig': 1,
                        'strides': (48, 48), 'overlaps': (96, 96), 'upsample_factor_grid': 2, 'num_frames_split': 80,
                        'max_deviation_rigid': 15, 'shifts_opencv': True, 'use_cuda': False, 'nonneg_movie': True,
                        'border_nan': 'copy'}


n_processes = psutil.cpu_count()
#cm.cluster.stop_server()
# Start a new cluster
c, dview, n_processes = cm.cluster.setup_cluster(backend='local',
                                                 n_processes=n_processes,
                                                 single_thread=False)


for session in sessions:
    print(session)
    # Run decoding for group of data tha have the same cropping parameters (same mouse)
    selection = selected_rows.query('(session ==' + f'{session}' + ')')
    #for i in range(init_trial,end_trial):
    #    print(i)
     #   selection = selected_rows.query('(trial ==' + f'{i}' + ')')
     #   for j in range(len(selection)):
     #       mouse_row = selection.iloc[j]
     #       mouse_row = main_decoding(mouse_row)
     #       states_df = db.append_to_or_merge_with_states_df(states_df, mouse_row)
     #       db.save_analysis_states_database(states_df, analysis_states_database_path, backup_path)

    decoding_version = mouse_row.name[4]
    # Run cropping for the already decoded group

    for parameters_cropping in parameters_cropping_list:
        selected_rows = db.select(states_df,'cropping',mouse = mouse_number,session=session, is_rest=is_rest,
                                  decoding_v = decoding_version,
                                  cropping_v= 0 )

        for i in range(init_trial,end_trial):
            selection = selected_rows.query('(trial ==' + f'{i}' + ')')
            for j in range(len(selection)):
                mouse_row = selection.iloc[j]
                mouse_row = main_cropping(mouse_row, parameters_cropping)
                states_df = db.append_to_or_merge_with_states_df(states_df, mouse_row)
                db.save_analysis_states_database(states_df, analysis_states_database_path, backup_path)

        cropping_version = mouse_row.name[5] # set the cropping version to the one currently used
        # Select rows to be motion corrected using current version of cropping, define motion correction parameters
        # (refer to parameter_setting_motion_correction)
        selected_rows = db.select(states_df,'motion_correction',mouse = mouse_number,session=session, is_rest=is_rest,
                                  decoding_v = decoding_version,
                                  cropping_v= cropping_version,
                                  motion_correction_v= 0)
        for i in range(init_trial,end_trial):
            print(i)
            selection = selected_rows.query('(trial ==' + f'{i}' + ')')
            for j in range(len(selection)):
                mouse_row = selection.iloc[j]
                mouse_row = main_motion_correction(mouse_row, parameters_motion_correction,dview)
                states_df = db.append_to_or_merge_with_states_df(states_df, mouse_row)
                db.save_analysis_states_database(states_df, analysis_states_database_path, backup_path)

        motion_correction_version = mouse_row.name[6]
        # alignment
        selected_rows = db.select(states_df,'alignment',mouse = mouse_number, session = session, is_rest= is_rest,
                                  decoding_v= decoding_version,
                                  cropping_v = cropping_version,
                                  motion_correction_v = motion_correction_version,
                                  alignment_v= 0)

