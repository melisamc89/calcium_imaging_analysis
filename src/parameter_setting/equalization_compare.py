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

# This should be in another file. Let's leave it here for now
sys.path.append('/home/sebastian/Documents/Melisa/calcium_imaging_analysis/src/')
sys.path.remove('/home/sebastian/Documents/calcium_imaging_analysis')

import src.configuration
import caiman as cm
from caiman.base.rois import com
import src.data_base_manipulation as db
from caiman.source_extraction.cnmf.cnmf import load_CNMF
import src.analysis.figures as figures
from caiman.base.rois import register_multisession
from src.steps.registering import run_registration as main_registration

# Paths
analysis_states_database_path = 'references/analysis/calcium_imaging_data_base_trial_wise_analysis.xlsx'
backup_path = 'references/analysis/backup/'
states_df = db.open_analysis_states_database(path = analysis_states_database_path)

mouse_number = 56165
is_rest = None

decoding_version = 1
motion_correction = 100
alignment_version = 1
source_extraction_version = 1
component_evaluation_version = 1

parameters_registration = {'session_wise': False, 'model_method': False, 'cost_threshold': 0.9, 'max_dist': 15,
                           'min_cell_size': 10, 'max_cell_size': 25}
for session in [1,2,4]:
    for cropping_version in [1,3,4,2]:
        selected_rows = db.select(states_df, 'registration', mouse=mouse_number, session=session, is_rest=is_rest,
                                  decoding_v=decoding_version,
                                  cropping_v=cropping_version,
                                  motion_correction_v=motion_correction,
                                  alignment_v=alignment_version,
                                  equalization_v=0,
                                  source_extraction_v=source_extraction_version,
                                  component_evaluation_v= component_evaluation_version)
        evaluated_trials = []
        resting_condition = []
        for i in range(len(selected_rows)):
            mouse_row = selected_rows.iloc[i]
            trial = mouse_row.name[2]
            rest = mouse_row.name[3]
            evaluated_trials.append(trial)
            resting_condition.append(rest)
        for trial , rest in evaluated_trials,resting_condition:
            trial_selection = db.select(states_df, 'registration', mouse=mouse_number, session=session, is_rest=rest,
                                        trial = trial,
                                      decoding_v=decoding_version,
                                      cropping_v=cropping_version,
                                      motion_correction_v=motion_correction,
                                      alignment_v=alignment_version,
                                      #equalization_v=0, # no equalization selection
                                      source_extraction_v=source_extraction_version,
                                      component_evaluation_v=component_evaluation_version,
                                      max_version=False)

            A_list = []  ## list for contour matrix on multiple trials
            # A_size = []  ## list for the size of A (just to verify it is always the same size)
            FOV_size = []  ## list for the cn filter dim (to verify it is always the same dims)
            A_number_components = []  ## list with the total number of components extracted for each trial
            C_dims = []  ## dimension of C, to keep track of timeline
            C_list = []  ## list with traces for each trial
            for i in range(len(trial_selection)):
                row = trial_selection.iloc[i]
                component_evaluation_hdf5_file_path = eval(row['component_evaluation_output'])['main']
                corr_path = eval(row['source_extraction_output'])['meta']['corr']['main']
                cnm = load_CNMF(component_evaluation_hdf5_file_path)
                cn_filter = np.load(db.get_file(corr_path))

                FOV_size.append(cn_filter.shape)
                # A_size.append(cnm.estimates.A.shape[0])
                A_number_components.append(cnm.estimates.idx_components.shape[0])
                A_list.append(cnm.estimates.A[:, cnm.estimates.idx_components])
                C_dims.append(cnm.estimates.C.shape)
                size = cnm.estimates.A[:, cnm.estimates.idx_components].sum(axis=0)
                if cnm.estimates.bl is None:
                    C_list.append(cnm.estimates.C[cnm.estimates.idx_components, :])
                else:
                    C_list.append(cnm.estimates.C[cnm.estimates.idx_components, :] - cnm.estimates.bl[
                        cnm.estimates.idx_components, np.newaxis])

            ## add a size restriction on the neurons that will further be processed. This restriction boundary
            # decision is based in the histogram of typical neuronal sizes
            min_size = parameters_registration['min_cell_size']
            max_size = parameters_registration['max_cell_size']
            new_A_list = []
            new_C_list = []
            A_components = []
            C_dims_new = []
            new_evaluated_trials = []
            new_evaluated_session = []
            for i in range(len(A_list)):
                accepted_size = []
                size = A_list[i].sum(axis=0)
                for j in range(size.shape[1]):
                    if size[0, j] > 10 and size[0, j] < 25:
                        accepted_size.append(j)
                if len(accepted_size) > 1:
                    new_A_list.append(A_list[i][:, accepted_size])
                    new_C_list.append(C_list[i][accepted_size, :])
                    A_components.append(A_number_components[i])
                    C_dims_new.append(new_C_list[-1].shape)
            A_list = new_A_list
            C_list = new_C_list

            ## run CaImAn registration rutine that use the Hungarian matching algorithm in the contours list
            spatial_union, assignments, match = register_multisession(A=A_list, dims=FOV_size[0],
                                                                      thresh_cost=parameters_registration['cost_threshold'],
                                                                      max_dist=parameters_registration['max_dist'])


figure, axes = plt.subplots(1,2)
A = A_list[0]
corr_path = eval(row['source_extraction_output'])['meta']['corr']['main']
cn_filter = np.load(db.get_file(corr_path))
coordinates = cm.utils.visualization.get_contours(A, np.shape(cn_filter), 0.2, 'max')
for c in coordinates:
    v = c['coordinates']
    c['bbox'] = [np.floor(np.nanmin(v[:, 1])), np.ceil(np.nanmax(v[:, 1])),
                 np.floor(np.nanmin(v[:, 0])), np.ceil(np.nanmax(v[:, 0]))]
    axes[0].plot(*v.T, c='r')

A = A_list[1]
corr_path = eval(row['source_extraction_output'])['meta']['corr']['main']
cn_filter = np.load(db.get_file(corr_path))
coordinates = cm.utils.visualization.get_contours(A, np.shape(cn_filter), 0.2, 'max')
for c in coordinates:
    v = c['coordinates']
    c['bbox'] = [np.floor(np.nanmin(v[:, 1])), np.ceil(np.nanmax(v[:, 1])),
                 np.floor(np.nanmin(v[:, 0])), np.ceil(np.nanmax(v[:, 0]))]
    axes[1].plot(*v.T, c='r')




figure, axes = plt.subplots(1)
C_0 = C_list[0].copy()
C_1 = C_list[1].copy()
C_0[0] += C_0[0].min()
for i in range(1, len(C_list[0])):
    C_0[i] += C_0[i].min() + C_0[:i].max()
    C_1[i] += C_1[i].min() + C_0[:i].max()
    axes.plot(C_0[i])
    axes.plot(C_1[i])
axes.set_xlabel('t [frames]')
axes.set_yticks([])
#axes.vlines(timeline[1],0, 150000, color = 'k')
axes.set_ylabel('activity')
figure.set_size_inches([50., .5 * len(C_0)])