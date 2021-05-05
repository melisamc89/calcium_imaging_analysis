
"""
Created on Tue Mar 17 13:43

@author: Melisa

From decoded files, run the rest of the pipeline and use multiple source extraction values
"""

import os
import psutil
import logging
import numpy as np

import src.configuration
import caiman as cm
import src.data_base_manipulation as db
import src.paths as paths
from src.steps.decoding import run_decoder as main_decoding
from src.steps.cropping import run_cropper as main_cropping
from src.steps.cropping import cropping_interval, cropping_segmentation
from src.analysis.figures import plot_movie_frame, plot_movie_frame_cropped, get_fig_gSig_filt_vals
from src.steps.motion_correction import run_motion_correction as main_motion_correction
from src.steps.source_extraction import run_source_extraction as main_source_extraction
from src.steps.component_evaluation import run_component_evaluation as main_component_evaluation
import src.analysis_files_manipulation as fm
import src.analysis.metrics as metrics
from caiman.source_extraction.cnmf.cnmf import load_CNMF


n_processes = psutil.cpu_count()
cm.cluster.stop_server()
# Start a new cluster
c, dview, n_processes = cm.cluster.setup_cluster(backend='local',
                                                 n_processes=n_processes,  # number of process to use, if you go out of memory try to reduce this one
                                                single_thread=False)

logging.info(f'Starting cluster. n_processes = {n_processes}.')

#%%
# Paths
analysis_states_database_path = paths.analysis_states_database_path
backup_path = os.environ['PROJECT_DIR'] +  'references/analysis/backup/'
#parameters_path = 'references/analysis/parameters_database.xlsx'

## Open thw data base with all data
states_df = db.open_analysis_states_database(path = analysis_states_database_path)

mouse_number = 341776
sessions = [1,2,3]
init_trial = 1
end_trial = 22
is_rest = None
session = 5

#%% CROPPING
# Select the rows for cropping
selected_rows = db.select(states_df,'cropping',mouse_number, session = 5)
mouse_row = selected_rows.iloc[0]

plot_movie_frame(mouse_row)
#%%
parameters_cropping = cropping_interval() #check whether it is better to do it like this or to use the functions get
# and set parameters from the data_base_manipulation file
# and set parameters from the data_base_manipulation file
parameters_cropping['segmentation'] = False
parameters_cropping_list = cropping_segmentation(parameters_cropping)

mouse_row = main_cropping(mouse_row, parameters_cropping)
plot_movie_frame_cropped(mouse_row) # verify that the cropping is the desired one
# Now cropping parameters had been selected. Next step is selection version analysis.
states_df = db.append_to_or_merge_with_states_df(states_df, mouse_row)
db.save_analysis_states_database(states_df, path=analysis_states_database_path, backup_path = backup_path)
# upload_to_server_cropped_movie(index,row)

#%% MOTION CORRECTION
# Select rows from the data base fo the next analysis step motion correction
selected_rows = db.select(states_df,'motion_correction',mouse_number)
mouse_row = selected_rows.iloc[0]
parameters_motion_correction = {'motion_correct': True, 'pw_rigid': True, 'save_movie_rig': False,
                                'gSig_filt': (5, 5), 'max_shifts': (25, 25), 'niter_rig': 1,
                                'strides': (48, 48),
                                'overlaps': (96, 96), 'upsample_factor_grid': 2, 'num_frames_split': 80,
                                'max_deviation_rigid': 15,
                                'shifts_opencv': True, 'use_cuda': False, 'nonneg_movie': True, 'border_nan': 'copy'}
mouse_row_new=main_motion_correction(mouse_row,parameters_motion_correction,dview)
states_df = db.append_to_or_merge_with_states_df(states_df, mouse_row_new)
db.save_analysis_states_database(states_df, path=analysis_states_database_path, backup_path = backup_path)


#%% SOURCE EXTRACTION
selected_rows = db.select(states_df,'source_extraction', mouse_number)
mouse_row = selected_rows.iloc[0]
## select a set of parameters and plot the binary corr, pnr and combined image to explore visualy different seed selections.
corr_limits = np.linspace(0.3, 0.6, 10)
pnr_limits = np.linspace(3, 7, 10)
gSig = 5
gSiz = 4 * gSig + 1
version = np.zeros((len(corr_limits)*len(pnr_limits)))
for ii in range(corr_limits.shape[0]):
    for jj in range(pnr_limits.shape[0]):
        parameters_source_extraction ={'session_wise': False,  'equalization': False, 'fr': 10, 'decay_time': 0.1, 'min_corr': corr_limits[ii],
                                       'min_pnr': pnr_limits[jj],'p': 1, 'K': None, 'gSig': (gSig, gSig), 'gSiz': (gSiz, gSiz),
                                       'merge_thr': 0.7, 'rf': 60, 'stride': 30, 'tsub': 1, 'ssub': 2, 'p_tsub': 1,
                                       'p_ssub': 2, 'low_rank_background': None, 'nb': 0, 'nb_patch': 0, 'ssub_B': 2,
                                       'init_iter': 2, 'ring_size_factor': 1.4, 'method_init': 'corr_pnr',
                                       'method_deconvolution': 'oasis', 'update_background_components': True,
                                        'center_psf': True, 'border_pix': 0, 'normalize_init': False,
                                        'del_duplicates': True, 'only_init': True}
        mouse_row_new = main_source_extraction(mouse_row, parameters_source_extraction, dview)
        states_df = db.append_to_or_merge_with_states_df(states_df, mouse_row_new)
        print(mouse_row_new.name)
        db.save_analysis_states_database(states_df, path=analysis_states_database_path, backup_path = backup_path)
        #save the version number for furhter plotting
        version[ii*len(pnr_limits)+jj] = mouse_row_new.name[8]

#%%
## plottin. This funcion  shoul be improved to be given a list of corr and pnr values and search in the data base that
#specific values, insted of going all over the versions values...
selected_rows = db.select(states_df,'component_evaluation',mouse = mouse, session = session, is_rest= is_rest,
                          cropping_v = cropping_version,
                          motion_correction_v = motion_correction_version, alignment_v= 0,max_version=False )
figures.plot_multiple_contours(selected_rows, version, corr_limits, pnr_limits)
figures.plot_traces_multiple(selected_rows, version , corr_limits, pnr_limits)

cm.stop_server(dview=dview)
