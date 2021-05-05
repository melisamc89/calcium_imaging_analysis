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
init_trial = 20
end_trial = 22
is_rest = None

#  Select first data
selected_rows = db.select(states_df, 'decoding', mouse=mouse_number,  decoding_v = 0)
mouse_row = selected_rows.iloc[0]
#mouse_row = main_decoding(mouse_row)
#plot_movie_frame(mouse_row)


for session in sessions:
    print(session)
    # Run decoding for group of data tha have the same cropping parameters (same mouse)
    selection1 = selected_rows.query('(session ==' + f'{session}' + ')')
    for i in range(init_trial,end_trial):
        print(i)
        selection = selection1.query('(trial ==' + f'{i}' + ')')
        for j in range(len(selection)):
            mouse_row = selection.iloc[j]
            mouse_row = main_decoding(mouse_row)
            states_df = db.append_to_or_merge_with_states_df(states_df, mouse_row)
            db.save_analysis_states_database(states_df, analysis_states_database_path, backup_path)

    decoding_version = mouse_row.name[4]
    # Run cropping for the already decoded group
