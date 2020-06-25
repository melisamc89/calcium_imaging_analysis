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
from src.steps.equalizer import run_equalizer as main_equalizing
from src.steps.cropping import cropping_interval, cropping_segmentation
from src.analysis.figures import plot_movie_frame
from src.steps.motion_correction import run_motion_correction as main_motion_correction
from src.steps.alignment2 import run_alignmnet as main_alignment
from src.steps.alignment2 import run_alignmnet_2 as main_alignment_2
from src.steps.source_extraction import run_source_extraction as main_source_extraction
from src.steps.component_evaluation import run_component_evaluation as main_component_evaluation
import src.paths as paths

# Paths
analysis_states_database_path = paths.analysis_states_database_path
backup_path = os.environ['PROJECT_DIR'] + 'references/analysis/backup/'
states_df = db.open_analysis_states_database(path=analysis_states_database_path)

mouse_number = 56165
sessions = [1, 2, 4]
#trials separated by days
init_trial = [1,6,11,16,21]
end_trial = [5,10,15,20,21]
is_rest = None

#  Select first data
selected_rows = db.select(states_df, 'decoding', mouse=mouse_number, decoding_v=1)
mouse_row = selected_rows.iloc[0]
# mouse_row = main_decoding(mouse_row)
plot_movie_frame(mouse_row)

# select cropping parameters
#parameters_cropping = cropping_interval()  # check whether it is better to do it like this or to use the functions get
# and set parameters from the data_base_manipulation file
#parameters_cropping['segmentation'] = False
#parameters_cropping_list = cropping_segmentation(parameters_cropping)

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

h_step = 10
parameters_equalizer = {'make_template_from_trial': '0', 'equalizer': 'histogram_matching', 'histogram_step': h_step}

gSig = 5
gSiz = 4 * gSig + 1
min_corr = 0.6
min_pnr = 4
##trial_wise_parameters
parameters_source_extraction = {'equalization': False, 'session_wise': True, 'fr': 10, 'decay_time': 0.1,
                                'min_corr': min_corr,
                                'min_pnr': min_pnr, 'p': 1, 'K': None, 'gSig': (gSig, gSig),
                                'gSiz': (gSiz, gSiz),
                                'merge_thr': 0.7, 'rf': 60, 'stride': 30, 'tsub': 1, 'ssub': 2, 'p_tsub': 1,
                                'p_ssub': 2, 'low_rank_background': None, 'nb': 0, 'nb_patch': 0,
                                'ssub_B': 2,
                                'init_iter': 2, 'ring_size_factor': 1.4, 'method_init': 'corr_pnr',
                                'method_deconvolution': 'oasis', 'update_background_components': True,
                                'center_psf': True, 'border_pix': 0, 'normalize_init': False,
                                'del_duplicates': True, 'only_init': True}

parameters_component_evaluation = {'min_SNR': 2.5,
                                   'rval_thr': 0.6,
                                   'use_cnn': False}

n_processes = psutil.cpu_count()
# cm.cluster.stop_server()
# Start a new cluster
c, dview, n_processes = cm.cluster.setup_cluster(backend='local',
                                                 n_processes=n_processes,
                                                 single_thread=False)


# %%
for session in sessions:
    print(session)
    selection = selected_rows.query('(session ==' + f'{session}' + ')')
    decoding_version = mouse_row.name[4]
    #for parameters_cropping in parameters_cropping_list:
    for cropping_version in [1,2,3,4]:

        # Run alignment
        alignment_version = 0
        selected_rows = db.select(states_df, 'alignment', mouse=mouse_number, session=session,
                                  decoding_v=decoding_version,
                                  cropping_v=cropping_version,
                                  motion_correction_v=1,
                                  alignment_v=alignment_version)
        for k in range(len(init_trial)): # run alignment day by day

            #days selection by trial ID
            selection = selected_rows.query('(trial >=' + f'{init_trial[k]}' + ')')
            selection = selection.query('(trial <=' + f'{end_trial[k]}' + ')')

            #run alignment
            parameters_alignment = {'make_template_from_trial': f'{init_trial[k]}', 'gSig_filt': (5, 5), 'max_shifts': (25, 25),
                                    'niter_rig': 1,
                                    'strides': (48, 48), 'overlaps': (96, 96), 'upsample_factor_grid': 2,
                                    'num_frames_split': 80,
                                    'max_deviation_rigid': 15, 'shifts_opencv': True, 'use_cuda': False,
                                    'nonneg_movie': True,
                                    'border_nan': 'copy'}
            new_selected_rows = main_alignment(selection, parameters_alignment, dview)

            #set version and save for first trial in a day
            new_index = db.replace_at_index1(new_selected_rows.iloc[0].name, 4 + 3, 20)
            row_new = new_selected_rows.iloc[0].copy()
            row_new.name = new_index
            new_index = db.replace_at_index1(row_new.name, 4 + 2, 10)
            row_new.name = new_index
            states_df = db.append_to_or_merge_with_states_df(states_df, row_new)
            db.save_analysis_states_database(states_df, analysis_states_database_path, backup_path)

            #set version and save the rest of the trials (this might not be necessary)
            for i in range(len(new_selected_rows)):
                new_index = db.replace_at_index1(new_selected_rows.iloc[i].name, 4 + 3, 2)
                row_new = new_selected_rows.iloc[i].copy()
                row_new.name = new_index
                new_index = db.replace_at_index1(row_new.name, 4 + 2, 10)
                row_new.name = new_index
                states_df = db.append_to_or_merge_with_states_df(states_df, row_new)
                db.save_analysis_states_database(states_df, analysis_states_database_path, backup_path)
            alignment_version = row_new.name[7]

        selected_rows = db.select(states_df, 'alignment', mouse=mouse_number, session=session,
                                  decoding_v=decoding_version,
                                  cropping_v=cropping_version,
                                  motion_correction_v=10,
                                  alignment_v=20)

        # run alignment
        parameters_alignment = {'make_template_from_trial': f'{1}', 'gSig_filt': (5, 5),
                                'max_shifts': (25, 25),
                                'niter_rig': 1,
                                'strides': (48, 48), 'overlaps': (96, 96), 'upsample_factor_grid': 2,
                                'num_frames_split': 80,
                                'max_deviation_rigid': 15, 'shifts_opencv': True, 'use_cuda': False,
                                'nonneg_movie': True,
                                'border_nan': 'copy'}

        new_selected_rows = main_alignment_2(selected_rows, parameters_alignment, dview)
        for i in range(len(new_selected_rows)):
            new_index = db.replace_at_index1(new_selected_rows.iloc[i].name, 4 + 3, 3)
            row_new = new_selected_rows.iloc[i].copy()
            row_new.name = new_index
            new_index = db.replace_at_index1(row_new.name, 4 + 2, 20)
            row_new.name = new_index
            states_df = db.append_to_or_merge_with_states_df(states_df, row_new)
            db.save_analysis_states_database(states_df, analysis_states_database_path, backup_path)
        alignment_version = row_new.name[7]

        selected_rows = db.select(states_df, 'source_extraction', mouse=mouse_number, session=session, is_rest=is_rest,
                                  decoding_v=decoding_version,
                                  cropping_v=cropping_version,
                                  motion_correction_v=20,
                                  alignment_v=3,
                                  source_extraction_v=0)
        for i in range(len(selected_rows)):
            selection = selected_rows.query('(trial ==' + f'{init_trial[i]}' + ')')
            mouse_row = selection.iloc[0]
            mouse_row_new = main_source_extraction(mouse_row, parameters_source_extraction, dview, session_wise= True)
            states_df = db.append_to_or_merge_with_states_df(states_df, mouse_row_new)
            db.save_analysis_states_database(states_df, path=analysis_states_database_path, backup_path=backup_path)
        source_extraction_version = mouse_row_new.name[9]

# %% run separately component evaluation

# evaluation
dview.terminate()
decoding_version = 1
for session in sessions:
    print(session)
    # Run decoding for group of data tha have the same cropping parameters (same mouse)
    selection = selected_rows.query('(session ==' + f'{session}' + ')')
    for cropping_version in [1, 2, 3, 4]:
        print(cropping_version)
        selected_rows = db.select(states_df, 'component_evaluation', mouse=mouse_number, session=session,
                                  is_rest=is_rest,
                                  decoding_v=decoding_version,
                                  cropping_v=cropping_version,
                                  motion_correction_v=20,
                                  alignment_v=3,
                                  equalization_v=0,
                                  source_extraction_v=1)
        for i in init_trial:
            selection = selected_rows.query('(trial ==' + f'{i}' + ')')
            mouse_row = selection.iloc[0]
            mouse_row_new = main_component_evaluation(mouse_row, parameters_component_evaluation,session_wise=True)
            states_df = db.append_to_or_merge_with_states_df(states_df, mouse_row_new)
            db.save_analysis_states_database(states_df, path=analysis_states_database_path, backup_path=backup_path)
