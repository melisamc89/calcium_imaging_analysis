#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 09:19:20 2019

@author: Melisa

This script is created for selection a cropping section from the videos and creates new files where only the region of
interest(ROI) is considered.

For testing different parameters in the upcoming parts of the pipeline, cropping parameters can be chosen to select
only a small region, in a way that this will accelerate the time requirements for running different tests.

Once the new cropping parameters are selected, the data base is update. If the cropping parameters are new and had never
been use for a particular mouse, session, trial, resting condition, a new line will be added to the data base and the
cropping version will be the previous maximum plus one. If the parameters where already used, the cropping version will not
be increased, but the cropping files will be rewritten.

Cropping parameters should be always the same for a mouse, so after selection the last part of this script can be use
to set the cropping parameters in the parameters data base for all the sessions and trials of a mouse.


"""
import sys
# This should be in another file. Let's leave it here for now
sys.path.append('/home/sebastian/Documents/Melisa/calcium_imaging_analysis/src/')
sys.path.remove('/home/sebastian/Documents/calcium_imaging_analysis')

import src.configuration # leave this here for users configuration
import src.data_base_manipulation as db
from src.steps.cropping import run_cropper as main_cropping
from src.steps.cropping import cropping_interval
from src.analysis.figures import plot_movie_frame, plot_movie_frame_cropped, get_fig_gSig_filt_vals

analysis_states_database_path = 'references/analysis/analysis_states_database.xlsx'
backup_path = 'references/analysis/backup/'
parameters_path = 'references/analysis/parameters_database.xlsx'

## Open the data base with all data
states_df = db.open_analysis_states_database()

#selection of data to crop. Take into account that cropping is more or less the same for every session in one mouse.
mouse = 32364
session = None
trial = None
is_rest = None
# CROPPING
# Select the rows for cropping
selected_rows = db.select(states_df,'cropping',mouse=mouse,session=session, trial= trial, is_rest=is_rest)
mouse_row = selected_rows.iloc[0]
#shows one frame of the movie so the cropping region can be choosen.
plot_movie_frame(mouse_row)

#%%
#manualy load the cropping region of interest
parameters_cropping = cropping_interval() #check whether it is better to do it like this or to use the functions get
# and set parameters from the data_base_manipulation file

mouse_row = main_cropping(mouse_row, parameters_cropping) #run cropping

plot_movie_frame_cropped(mouse_row) # verify that the cropping is the desired one
# Now cropping parameters had been selected. Next step is selection version analysis.

states_df = db.append_to_or_merge_with_states_df(states_df, mouse_row) #merge the new state with the previous data base
db.save_analysis_states_database(states_df, path=analysis_states_database_path, backup_path = backup_path) #save data base
# upload_to_server_cropped_movie(index,row)

#%% Here add parameter setting for all trials and sessions of one mouse, and update the parameters data base

for i in range(len(selected_rows)):
    index = selected_rows.iloc[i].name
    db.set_parameters('cropping',parameters_cropping, mouse = mouse, session = index[1], trial = index[2] , is_rest = index[3])
