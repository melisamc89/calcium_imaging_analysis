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

figure_path = '/mnt/Data01/data/calcium_imaging_analysis/data/interim/component_evaluation/trial_wise/meta/figures/'

data_path = '/home/sebastian/Documents/Melisa/neural_analysis/data/calcium_traces_concatenation/'
typical_size = []
component_evaluation_version = 1
session = 1
cropping_version = 1

for session in [1, 2, 4]:
    for cropping_version in [1,3,4,2]:
        selected_rows = db.select(states_df, 'registration', mouse=mouse_number, session=session, is_rest=is_rest,
                                  decoding_v=decoding_version,
                                  cropping_v=cropping_version,
                                  motion_correction_v=motion_correction,
                                  alignment_v=alignment_version,
                                  equalization_v=0,
                                  source_extraction_v=source_extraction_version,
                                  component_evaluation_v= component_evaluation_version)
        parameters_registration = {'session_wise': False, 'model_method': False, 'cost_threshold': 0.9, 'max_dist': 15,
                      'min_cell_size': 10, 'max_cell_size': 25}
        new_selected_rows = main_registration(selected_rows, parameters_registration)





figure, axes = plt.subplots(1)
C_0 = C_matrix.copy()
C_0[0] += C_0[0].min()
for i in range(1, len(C_matrix)):
    C_0[i] += C_0[i].min() + C_0[:i].max()
    axes.plot(C_0[i])
axes.set_xlabel('t [frames]')
axes.set_yticks([])
#axes.vlines(timeline[1],0, 150000, color = 'k')
axes.set_ylabel('activity')
figure.set_size_inches([50., .5 * len(C_0)])
figure.savefig('/mnt/Data01/data/calcium_imaging_analysis/data/interim/component_evaluation/trial_wise/meta/figures/mouse_56165_cropping_4.png')






figure, axes = plt.subplots(1)
axes.hist(typical_size, bins = 25)


### this part computes center of mass and distance between center of mass of the contours
A_center_of_mass_list_x = []
A_center_of_mass_list_y = []
A_template = []
trial_belonging = []
for i in range(len(A_list)):
    for j in range(A_list[i].shape[1]):
        cm_coordinates = com(A_list[i][:, j],FOV_size[i][0], FOV_size[i][1])
        A_center_of_mass_list_x.append(cm_coordinates[0][0])
        A_center_of_mass_list_y.append(cm_coordinates[0][1])
        trial_belonging.append(i+1)
        A_template.append(A_list[i][:,j])

distance_list = []
for i in range(len(A_center_of_mass_list_x)):
    for j in range(i+1,len(A_center_of_mass_list_x)):
        x1 = A_center_of_mass_list_x[i]
        x2 = A_center_of_mass_list_x[j]
        y1 = A_center_of_mass_list_y[i]
        y2 = A_center_of_mass_list_y[j]
        distance = math.sqrt((x1-x2)**2 + (y1-y2)**2)
        distance_list.append(distance)

figure, axes = plt.subplots(1)
axes.hist(distance_list, bins = 50)
axes.set_xlabel('Distance [pixels]', fontsize = 12)
axes.set_ylabel('Number of pairs', fontsize = 12)
figure.suptitle('Distance between center of mass', fontsize = 20)
figure_name = db.create_file_name(5,row.name)
figure.savefig( figure_path + figure_name + '.png')

figure1, axes = plt.subplots(1)
for i in range(1,126):
    new_vector_x=[]
    new_vector_y=[]
    for j in range(len(A_center_of_mass_list_y)):
        if trial_belonging[j] == i:
            new_vector_x.append(A_center_of_mass_list_x[j])
            new_vector_y.append(A_center_of_mass_list_y[j])
    axes.scatter(new_vector_y,new_vector_x)


## this part computes the pcc between contours templates for the entire session

pcc_list = []
for i in range(len(A_template)):
    for j in range(i+1,len(A_template)):
        pcc =np.corrcoef(np.array((A_template[i].todense())).flatten(),np.array((A_template[j].todense())).flatten())
        pcc_list.append(pcc[0,1])


figure, axes = plt.subplots(1)
axes.hist(pcc_list, bins = 50)
axes.set_xlabel('PCC', fontsize = 12)
axes.set_ylabel('Number of pairs', fontsize = 12)
axes.set_yscale('log')
figure.suptitle('PCC between contours template', fontsize = 20)
figure.savefig( '/mnt/Data01/data/calcium_imaging_analysis/data/interim/component_evaluation/trial_wise/meta/figures/mouse_56165_session_1_pcc_contours.png')



figure, axes = plt.subplots(5,5)
i = 0
for n in range(5):
    for m in range(5):
        i = n * 5 + m + 1
        row = selected_rows.iloc[i]
        component_evaluation_hdf5_file_path = eval(row['component_evaluation_output'])['main']
        corr_path = eval(row['source_extraction_output'])['meta']['corr']['main']
        cnm = load_CNMF(component_evaluation_hdf5_file_path)
        cn_filter = np.load(db.get_file(corr_path))

        coordinates = cm.utils.visualization.get_contours(A_list[i], np.shape(cn_filter), 0.2, 'max')
        axes[n, m].imshow(cn_filter)
        for c in coordinates:
            v = c['coordinates']
            c['bbox'] = [np.floor(np.nanmin(v[:, 1])), np.ceil(np.nanmax(v[:, 1])),
                        np.floor(np.nanmin(v[:, 0])), np.ceil(np.nanmax(v[:, 0]))]
            axes[n, m].plot(*v.T, c='w')


spatial_union, assignments, match = register_multisession(A=A_list[0:41], dims=FOV_size[0], thresh_cost=0.9, max_dist= 5)

coordinates = cm.utils.visualization.get_contours(spatial_union, np.shape(cn_filter), 0.2, 'max')
for c in coordinates:
    v = c['coordinates']
    c['bbox'] = [np.floor(np.nanmin(v[:, 1])), np.ceil(np.nanmax(v[:, 1])),
                 np.floor(np.nanmin(v[:, 0])), np.ceil(np.nanmax(v[:, 0]))]
    axes.plot(*v.T, c='r')



time=0
timeline=[0]
for i in range(len(C_dims)):
    time = time + C_dims[i][1]
    timeline.append(timeline[i]+C_dims[i][1])
C_matrix = np.zeros((spatial_union.shape[1],time))

for i in range(spatial_union.shape[1]):
    for j in range(assignments.shape[1]):
        if math.isnan(assignments[i,j]) == False:
            C_matrix[i][timeline[j]:timeline[j+1]] = (C_list[j])[int(assignments[i,j]),:]

figure, axes = plt.subplots(1)
C = C_matrix.copy()
C[0] += C[0].min()
for i in range(1, len(C)):
    C[i] += C[i].min() + C[:i].max()
    axes.plot(C[i])
axes.set_xlabel('t [frames]')
axes.set_yticks([])
axes.vlines(timeline,0, 50000, color = 'k')
axes.set_ylabel('activity')
figure.set_size_inches([50., .5 * len(C)])
figure.savefig('/mnt/Data01/data/calcium_imaging_analysis/data/interim/component_evaluation/trial_wise/meta/figures/mouse_56165_session_1_v1.2.100.0.1.1_thres=0.9_dist=15.png')


figure, axes = plt.subplots(1)
i = 0
corr_name = 'meta/corr/mouse_56165_session_1_trial_1_v1.1.100.0_gSig_5.npy'
corr_path = directory_path1 + corr_name
cn_filter = np.load(db.get_file(corr_path))
axes.imshow(cn_filter)
for n in range(5):
    for m in range(5):
        i = n * 5 + m + 1
        file_name = 'main/mouse_56165_session_1_trial_' + f'{i}' + '_v1.1.100.0.1.1.hdf5'
        input_hdf5_file_path = directory_path2 + file_name
        cnm = load_CNMF(input_hdf5_file_path)
        coordinates = cm.utils.visualization.get_contours(A_list[i],np.shape(cn_filter), 0.2, 'max')
        for c in coordinates:
            v = c['coordinates']
            c['bbox'] = [np.floor(np.nanmin(v[:, 1])), np.ceil(np.nanmax(v[:, 1])),
                         np.floor(np.nanmin(v[:, 0])), np.ceil(np.nanmax(v[:, 0]))]
            axes.plot(*v.T, c='w')
        file_name = 'main/mouse_56165_session_1_trial_' + f'{i}' + '_R_v1.1.100.0.1.1.hdf5'
        input_hdf5_file_path = directory_path2 + file_name
        cnm = load_CNMF(input_hdf5_file_path)
        coordinates = cm.utils.visualization.get_contours(A_list[2*i],np.shape(cn_filter), 0.2, 'max')
        for c in coordinates:
            v = c['coordinates']
            c['bbox'] = [np.floor(np.nanmin(v[:, 1])), np.ceil(np.nanmax(v[:, 1])),
                         np.floor(np.nanmin(v[:, 0])), np.ceil(np.nanmax(v[:, 0]))]
            axes.plot(*v.T, c='w')
coordinates = cm.utils.visualization.get_contours(spatial_union, np.shape(cn_filter), 0.2, 'max')
for c in coordinates:
    v = c['coordinates']
    c['bbox'] = [np.floor(np.nanmin(v[:, 1])), np.ceil(np.nanmax(v[:, 1])),
                 np.floor(np.nanmin(v[:, 0])), np.ceil(np.nanmax(v[:, 0]))]
    axes.plot(*v.T, c='r')


i = 0
general_coordinates = cm.utils.visualization.get_contours(spatial_union, np.shape(cn_filter), 0.2, 'max')
for n in range(5):
    for m in range(5):
        i = n * 5 + m + 1
        corr_name = 'meta/corr/mouse_56165_session_1_trial_'+ f'{i}' + '_v1.1.100.0_gSig_5.npy'
        corr_path = directory_path1 + corr_name
        cn_filter = np.load(db.get_file(corr_path))
        figure, axes = plt.subplots(1)
        axes.imshow(cn_filter/np.max(cn_filter))
        for c in general_coordinates:
            v = c['coordinates']
            c['bbox'] = [np.floor(np.nanmin(v[:, 1])), np.ceil(np.nanmax(v[:, 1])),
                         np.floor(np.nanmin(v[:, 0])), np.ceil(np.nanmax(v[:, 0]))]
            axes.plot(*v.T, c='r')
        coordinates = cm.utils.visualization.get_contours(A_list[i], np.shape(cn_filter), 0.2, 'max')
        for c in coordinates:
            v = c['coordinates']
            c['bbox'] = [np.floor(np.nanmin(v[:, 1])), np.ceil(np.nanmax(v[:, 1])),
                         np.floor(np.nanmin(v[:, 0])), np.ceil(np.nanmax(v[:, 0]))]
            axes.plot(*v.T, c='w')
        figure.suptitle('Mouse 56165 session 1 trial ' + f'{i}')
        figure_name = 'contours/mouse_56165_session_1_trial_'+ f'{i}' + '_v1.1.100.0.1.1.png'
        figure.savefig(figure_path + figure_name)

        corr_name = 'meta/corr/mouse_56165_session_1_trial_'+ f'{i}' + '_R_v1.1.100.0_gSig_5.npy'
        corr_path = directory_path1 + corr_name
        cn_filter = np.load(db.get_file(corr_path))
        figure, axes = plt.subplots(1)
        axes.imshow(cn_filter/np.max(cn_filter))
        for c in general_coordinates:
            v = c['coordinates']
            c['bbox'] = [np.floor(np.nanmin(v[:, 1])), np.ceil(np.nanmax(v[:, 1])),
                         np.floor(np.nanmin(v[:, 0])), np.ceil(np.nanmax(v[:, 0]))]
            axes.plot(*v.T, c='r')
        coordinates = cm.utils.visualization.get_contours(A_list[i*2], np.shape(cn_filter), 0.2, 'max')
        for c in coordinates:
            v = c['coordinates']
            c['bbox'] = [np.floor(np.nanmin(v[:, 1])), np.ceil(np.nanmax(v[:, 1])),
                         np.floor(np.nanmin(v[:, 0])), np.ceil(np.nanmax(v[:, 0]))]
            axes.plot(*v.T, c='w')
        figure.suptitle('Mouse 56165 session 1 trial ' + f'{i}')
        figure_name = 'contours/mouse_56165_session_1_trial_'+ f'{i}' + '_R_v1.1.100.0.1.1.png'
        figure.savefig(figure_path + figure_name)

