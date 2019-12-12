#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 09:19:20 2019

@author: Melisa

This script is designed to test the impact of different motion correction parameters in the pre-processed images.

The two main parameters that are explored are:

    gSig - > size of the gaussian filter that is applied to the image before motion corrected.
    strides_vector ->

    All can be run in a pw_rigid / rigid mode.

gSig sizes are selected in a np.array, and it's values can be changed in line: gSig_filters = 2*np.arange(0,5)+3
The effect of this filters are first ploted using the function in src.analysis.figures.get_fig_gSig_filt_vals.

After running motion correction in all the selected gSig values, the data base is updates with the new versions of
motion correction. If the metrics are computed (using function  metrics.get_metrics_motion_correction) the data base is
also updated with the quality metrics of motion correction (in this case crispness of the summaty images of mean image
and correlation image).

After selection of a gSig value, strides can be testes as a relevant parameter, and the same data base and metrics
saving applies here.

At the end of the script all crispness can be compare. Take into account that crispness (as a Frobenious norm that is not
normalized now) is dependent on the size of the summary image, so only videos that have the same cropping size can be
compared.

Last part of the script is for setting in the parameters data base the selected values for this particular mouse, session,
trial and resting condition. After running a few conditions check whether the same can be used for all the mice.

"""

import sys
import psutil
import logging
import numpy as np
# This should be in another file. Let's leave it here for now
sys.path.append('/home/sebastian/Documents/Melisa/calcium_imaging_analysis/src/')
sys.path.remove('/home/sebastian/Documents/calcium_imaging_analysis')

import src.configuration
import caiman as cm
import src.data_base_manipulation as db
from src.analysis.figures import get_fig_gSig_filt_vals

from src.steps.motion_correction import run_motion_correction as main_motion_correction
import src.analysis.metrics as metrics
import src.analysis.figures as figures

analysis_states_database_path = 'references/analysis/analysis_states_database.xlsx'
backup_path = 'references/analysis/backup/'

## Open the data base with all data
states_df = db.open_analysis_states_database()
#detail the selected mouse
mouse = 32364
session = 1
trial = 1
is_rest = 1
cropping_version = 1

# Select rows from the data base fo the next analysis step motion correction
selected_rows = db.select(states_df,'motion_correction', mouse = mouse, session = session, trial = trial, is_rest = is_rest,
                          cropping_v = cropping_version)

#For visualization: plot different filter sizes
mouse_row = selected_rows.iloc[0]
gSig_filters = 2*np.arange(0,5)+3
get_fig_gSig_filt_vals(mouse_row,gSig_filters)


#start a cluster
n_processes = psutil.cpu_count()
cm.cluster.stop_server()
# Start a new cluster
c, dview, n_processes = cm.cluster.setup_cluster(backend='local',
                                                 n_processes=n_processes,  # number of process to use, if you go out of memory try to reduce this one
                                                single_thread=False)

logging.info(f'Starting cluster. n_processes = {n_processes}.')

# Run all the motion correction steps but changing the filter size parameter
for gSig in gSig_filters:
    print(gSig)
    parameters_motion_correction = {'motion_correct': True, 'pw_rigid': True, 'save_movie_rig': False,
                                    'gSig_filt': (gSig, gSig), 'max_shifts': (25, 25), 'niter_rig': 1, 'strides': (96, 96),
                                    'overlaps': (48, 48), 'upsample_factor_grid': 2, 'num_frames_split': 80, 'max_deviation_rigid': 15,
                                    'shifts_opencv': True, 'use_cuda': False, 'nonneg_movie': True, 'border_nan': 'copy'}

    mouse_row_new=main_motion_correction(mouse_row,parameters_motion_correction,dview)
    # Compute metrics for quality assessment in motion corrected movies'crispness'
    mouse_row_new = metrics.get_metrics_motion_correction(mouse_row_new, crispness = True)
    states_df = db.append_to_or_merge_with_states_df(states_df, mouse_row_new)
    db.save_analysis_states_database(states_df, path=analysis_states_database_path, backup_path = backup_path)

#%%
#
# Choose filter size=5
# Run all the motion correction steps but changing the strides size explore the same as in the paper
# (24/48), (48/48), (96,48) , (128,48)
# Select rows from the data base fo the next analysis step motion correction
selected_rows = db.select(states_df,'motion_correction',56165, mouse = mouse, session = session, trial = trial, is_rest = is_rest,
                          cropping_v = cropping_version)

mouse_row = selected_rows.iloc[0]
strides_vector=[24,48,96,128]
for strides in strides_vector:
    print(strides)
    parameters_motion_correction = {'motion_correct': True, 'pw_rigid': True, 'save_movie_rig': False,
                                    'gSig_filt': (5, 5), 'max_shifts': (25, 25), 'niter_rig': 1, 'strides': (strides, strides),
                                    'overlaps': (48, 48), 'upsample_factor_grid': 2, 'num_frames_split': 80, 'max_deviation_rigid': 15,
                                    'shifts_opencv': True, 'use_cuda': False, 'nonneg_movie': True, 'border_nan': 'copy'}

    mouse_row_new=main_motion_correction(mouse_row,parameters_motion_correction,dview)
    # Compute metrics for quality assessment in motion corrected movies'crispness'
    mouse_row_new = metrics.get_metrics_motion_correction(mouse_row_new, crispness = True)
    states_df = db.append_to_or_merge_with_states_df(states_df, mouse_row_new)
    db.save_analysis_states_database(states_df, path=analysis_states_database_path, backup_path = backup_path)

#save state version index to comparare in next step
index = mouse_row.name
cm.stop_server(dview=dview)

#%%
# Now compare the result using crispness for the used analyzed parameters

#choose all states corresponding to this particular analysis state
# with the parameter max_version = False, all the analysis states with the same previous parameters that the current one
#are selected

selected_rows = db.select(states_df,'motion_correction',mouse = index[0], session = index[1],trial = index[2],
                          is_rest = index [3], decoding_v= index[4] , cropping_v = index[5],max_version=False)

#choose only the ones that explores filter size (let's think if this make sense...we can just select the best in all)

# Choose the best one using crispness measurement in summary image (mean or corr_image, both values are save)

#visualizarion
crispness_mean_original, crispness_corr_original, crispness_mean, crispness_corr = metrics.compare_crispness(
    selected_rows)
crispness_figure = figures.plot_crispness_for_parameters(selected_rows)
selected_version = np.argmin(crispness_mean)
selected_parameters = eval(selected_rows.iloc[selected_version]['motion_correction_parameters'])
print('Motion Correction selected parameters = ')
print(selected_parameters)

#%% Here add parameter setting for all trials and sessions of one mouse, and update the parameters data base


