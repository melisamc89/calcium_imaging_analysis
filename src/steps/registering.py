# -*- coding: utf-8 -*-
"""

Created on Tue Feb  4 14:47:00 2020

@author: Melisa
"""

import os
import logging
import datetime
import numpy as np
import pickle
import math
import scipy
import scipy.stats

from scipy.sparse import csr_matrix

import caiman as cm
from caiman.base.rois import com
from caiman.source_extraction.cnmf.cnmf import load_CNMF
from caiman.base.rois import register_multisession

import src.data_base_manipulation as db
import src.paths as paths
from random import randint


#Method posibilities (model method): registration (True) or matching (False)
# cost_threshold: threshold for cost in matching with Hungarian matching algorithm.
# max_dist : maximum distance between centroids to allow a matching.
# max_cell_size and min_cell size should be taken from the distribution of typical sizes (use function typical size)
parameters = { 'model_method': False, 'cost_threshold' : 0.9 , 'max_dist' : 15 , 'min_cell_size' : 10, 'max_cell_size' : 25}


'''
This is the main idea for typical size funcion

    ## add a size restriction on the neurons that will further be processed. This restriction boundary
    # decision is based in the histogram of typical neuronal sizes
    min_size = np.mean(typical_size) - 2*np.std(typical_size)
    max_size = np.mean(typical_size) + 2*np.std(typical_size)
    new_A_list = []
    new_C_list = []
    for i in range(len(A_list)):
        accepted_size = []
        size = A_list[i].sum(axis=0)
        for j in range(size.shape[1]):
            if size[0, j] > min_size and size[0, j] < max_size:
                accepted_size.append(j)
        new_A_list.append(A_list[i][:, accepted_size])
        new_C_list.append(C_list[i][accepted_size, :])
    A_list = new_A_list
    C_list = new_C_list

'''

def run_registration(selected_rows,parameters):

    '''
    This is the main registering function. Is is supposed to be run after trial wise component evaluation.
    Regsitration takes over different contours of trial wise source extracted contours and do a matching between cells.
    It can use two different methods: Hungarian matching algorithm (RegisterMulti) (as implement in Giovannucci, et al.
    2019) or cell registration (CellReg)using centroids distance and spatial correlation (as implemented in Sheintuch, et al. 2017).
    Default method is registration with no modeling of distributions of centroids and spatial correlation.

    :param row: state of the data base containing all the trials that were aligned and source extracted.
    :param paramerts: dictionary containing parameters (work in process)
    :return: row: dictionary
    '''

    step_index = 7
    # Sort the dataframe correctly
    df = selected_rows.copy()
    df = df.sort_values(by=paths.multi_index_structure)
    try:
        df.reset_index()[['session','trial', 'is_rest']].set_index(['session','trial', 'is_rest'], verify_integrity=True)
    except ValueError:
        logging.error('You passed multiple of the same trial in the dataframe df')
        return df

    ##create the dictionary with metadata information
    output = {
        'meta': {
            'analysis': {
                'analyst': os.environ['ANALYST'],
                'date': datetime.datetime.today().strftime("%m-%d-%Y"),
                'time': datetime.datetime.today().strftime("%H:%M:%S")
            },
            'duration': {}
        }
    }

    first_row = df.iloc[0]
    alignmnet_output = eval(first_row['alignment_output'])
    alignment_timeline_file = alignmnet_output['meta']['timeline']


    A_list = []  ## list for contour matrix on multiple trials
    #A_size = []  ## list for the size of A (just to verify it is always the same size)
    FOV_size = []  ## list for the cn filter dim (to verify it is always the same dims)
    A_number_components = []  ## list with the total number of components extracted for each trial
    C_dims = []  ## dimension of C, to keep track of timeline
    C_list = []  ## list with traces for each trial
    evaluated_trials = []
    typical_size = []
    for i in range(len(df)):
        row = df.iloc[i]
        component_evaluation_hdf5_file_path = eval(row['component_evaluation_output'])['main']
        corr_path = eval(row['source_extraction_output'])['meta']['corr']['main']
        cnm = load_CNMF(component_evaluation_hdf5_file_path)
        cn_filter = np.load(db.get_file(corr_path))

        FOV_size.append(cn_filter.shape)
        #A_size.append(cnm.estimates.A.shape[0])
        A_number_components.append(cnm.estimates.idx_components.shape[0])
        A_list.append(cnm.estimates.A[:, cnm.estimates.idx_components])
        C_dims.append(cnm.estimates.C.shape)
        size = cnm.estimates.A[:, cnm.estimates.idx_components].sum(axis=0)
        for j in range(len(cnm.estimates.idx_components)):
            typical_size.append(size[0, j])
        C_list.append(cnm.estimates.C[cnm.estimates.idx_components, :])
        evaluated_trials.append((selected_rows.iloc[i].name[2] - 1) * 2 + selected_rows.iloc[i].name[3] + 1) ## number that goes from 0 to 42

    ## add a size restriction on the neurons that will further be processed. This restriction boundary
    # decision is based in the histogram of typical neuronal sizes
    min_size = parameters['min_cell_size']
    max_size = parameters['max_cell_size']
    new_A_list = []
    new_C_list = []
    for i in range(len(A_list)):
        accepted_size = []
        size = A_list[i].sum(axis=0)
        for j in range(size.shape[1]):
            if size[0, j] > min_size and size[0, j] < max_size:
                accepted_size.append(j)
        if len(accepted_size) > 0:
            new_A_list.append(A_list[i][:, accepted_size])
            new_C_list.append(C_list[i][accepted_size, :])
        else:
            evaluated_trials.remove(i)
    A_list = new_A_list
    C_list = new_C_list

    spatial_union, assignments, match = register_multisession(A=A_list, dims=FOV_size[0], thresh_cost=parameters['cost_threshold'], max_dist=parameters['max_dist'])

    with open(alignment_timeline_file, 'rb') as f:
        timeline = pickle.load(f)
    total_time = timeline[len(timeline) - 1][1] + C_list[len(C_list)-1].shape[1]
    timeline.append(['End',total_time])
    C_matrix = np.zeros((spatial_union.shape[1], total_time))

    new_assignments = np.zeros_like(assignments)
    ##rename the assignments to the correct trial number
    for cell in range(assignments.shape[0]):
        for trial in range(len(evaluated_trials)):
            positions =np.where(assignments[cell]==trial)[0]
            new_assignments[cell][positions]= np.ones_like(positions) * evaluated_trials[trial]

    for i in range(spatial_union.shape[1]):
        for j in range(1,new_assignments.shape[1]):
            trial = evaluated_trials[j-1]
            if new_assignments[i, j] != 0:
                C_matrix[i][timeline[trial][1]:timeline[trial+1][1]] = (C_list[j])[int(new_assignments[i, j]-1), :]


    return C_matrix, spatial_union