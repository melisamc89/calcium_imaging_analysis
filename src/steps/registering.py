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
import copy

#Method posibilities (model method): registration (True) or matching (False)
# cost_threshold: threshold for cost in matching with Hungarian matching algorithm.
# max_dist : maximum distance between centroids to allow a matching.
# max_cell_size and min_cell size should be taken from the distribution of typical sizes (use function typical size)
#parameters = { 'session_wise': False,'model_method': False, 'cost_threshold' : 0.9 , 'max_dist' : 15 ,
#               'min_cell_size' : 10, 'max_cell_size' : 25}

class estimates(object):
    def __init__(self , A = None, C = None):
        self.A = A
        self.C = C

def run_registration(selected_rows,parameters):

    '''
    This is the main registering function. Is is supposed to be run after trial wise component evaluation.
    Registration takes over different contours of trial wise source extracted contours and do a matching between cells.
    It can use two different methods: Hungarian matching algorithm (RegisterMulti) (as implement in Giovannucci, et al.
    2019) or cell registration (CellReg)using centroids distance and spatial correlation (as implemented in Sheintuch, et al. 2017).
    Default method is registration with no modeling of distributions of centroids and spatial correlation.

    :param row: state of the data base containing all the trials that were aligned and source extracted.
    :param paramerts: dictionary containing parameters (work in process)
    :return: row: dictionary
    '''

    ## define registration version
    step_index = 7
    # Sort the dataframe correctly
    df = selected_rows.copy()
    df = df.sort_values(by=paths.multi_index_structure)
    for i in range(len(df)):
        index = df.iloc[i].name
        row_new = db.set_version_analysis('registration',df.iloc[i].copy())
        df = db.append_to_or_merge_with_states_df(df, row_new)
    df = df.query('registration_v == ' + f'{row_new.name[4+step_index]}')
    df = df.sort_values(by=paths.multi_index_structure)

    try:
        df.reset_index()[['session','trial', 'is_rest']].set_index(['session','trial', 'is_rest'], verify_integrity=True)
    except ValueError:
        logging.error('You passed multiple of the same trial in the dataframe df')
        return df


    if parameters['session_wise'] == False:
        data_dir = os.environ['DATA_DIR'] + 'data/interim/registration/trial_wise/main/'
    else:
        data_dir = os.environ['DATA_DIR'] + 'data/interim/registration/session_wise/main/'

    file_name = db.create_file_name(step_index, row_new.name)
    output_file_path =  data_dir + f'{file_name}.pkl'

    ##create the dictionary with metadata information
    output = {
        'main': output_file_path,
        'meta': {
            'analysis': {
                'analyst': os.environ['ANALYST'],
                'date': datetime.datetime.today().strftime("%m-%d-%Y"),
                'time': datetime.datetime.today().strftime("%H:%M:%S")
            },
            'duration': {}
        }
    }

    ## take alignment data for the timeline of alingment
    first_row = df.iloc[0]
    alignmnet_output = eval(first_row['alignment_output'])
    alignment_timeline_file = alignmnet_output['meta']['timeline']


    ## multiple list created to append the relevant information for the registration and creation of a unique time trace
    ## matrix (cnm.estimates.A  and cnm.estimates.C ) both taken after component evaluation
    A_list = []  ## list for contour matrix on multiple trials
    #A_size = []  ## list for the size of A (just to verify it is always the same size)
    FOV_size = []  ## list for the cn filter dim (to verify it is always the same dims)
    A_number_components = []  ## list with the total number of components extracted for each trial
    C_dims = []  ## dimension of C, to keep track of timeline
    C_list = []  ## list with traces for each trial
    evaluated_trials = []
    evaluated_session  = []
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
        if cnm.estimates.bl is None:
            C_list.append(cnm.estimates.C[cnm.estimates.idx_components, :])
        else:
            C_list.append(cnm.estimates.C[cnm.estimates.idx_components, :]-cnm.estimates.bl[cnm.estimates.idx_components,np.newaxis])
        evaluated_trials.append( (df.iloc[i].name[2] -1) * 2 + df.iloc[i].name[3]) ## number that goes from 0 to 42
        evaluated_session.append(df.iloc[i].name[1])

    ## add a size restriction on the neurons that will further be processed. This restriction boundary
    # decision is based in the histogram of typical neuronal sizes
    min_size = parameters['min_cell_size']
    max_size = parameters['max_cell_size']
    new_A_list = []
    new_C_list = []
    A_components = []
    C_dims_new = []
    new_evaluated_trials= []
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
            new_evaluated_trials.append(evaluated_trials[i])
            new_evaluated_session.append(evaluated_session[i])
    A_list = new_A_list
    C_list = new_C_list

    ## run CaImAn registration rutine that use the Hungarian matching algorithm in the contours list
    spatial_union, assignments, match = register_multisession(A=A_list, dims=FOV_size[0], thresh_cost=parameters['cost_threshold'], max_dist=parameters['max_dist'])

    ## open the timeline and create the new traces matrix C_matrix
    with open(alignment_timeline_file, 'rb') as f:
        timeline = pickle.load(f)
    total_time = timeline[len(timeline) - 1][1] + C_list[len(C_list)-1].shape[1]
    timeline.append(['End',total_time])
    C_matrix = np.zeros((spatial_union.shape[1], total_time))

    #new_assignments = np.zeros_like(assignments)
    ##rename the assignments to the correct trial number
    #for cell in range(assignments.shape[0]):
    #    for trial in range(len(evaluated_trials)):
    #        positions =np.where(assignments[cell]==trial)[0]
    #        new_assignments[cell][positions]= np.ones_like(positions) * evaluated_trials[trial]

    new_assignments = np.zeros((spatial_union.shape[1],len(timeline)))
    for i in range(spatial_union.shape[1]):
        for j in range(assignments.shape[1]):
            trial = new_evaluated_trials[j]
            if math.isnan(assignments[i, j]) == False:
                new_assignments[i][trial] = assignments[i, j]+1

    unique_session = []
    for x in evaluated_session:
        if x not in unique_session:
            unique_session.append(x)
    session_vector = np.arange(0,len(unique_session))
    final_evaluated_session = []
    for i in range(assignments.shape[1]):
        for j in range(len(unique_session)):
            if new_evaluated_session[i] == unique_session[j]:
                final_evaluated_session.append(session_vector[j])


    for i in range(spatial_union.shape[1]):
        for j in range(assignments.shape[1]):
            trial = (final_evaluated_session[j]+1) * new_evaluated_trials[j]
            print(trial)
            if math.isnan(assignments[i, j]) == False:
                C_matrix[i][timeline[trial][1]:timeline[trial][1]+C_dims_new[j][1]] =  (C_list[j])[int(assignments[i, j]), :]
                #C_matrix[i][timeline[trial][1]:timeline[trial+1][1]] = (C_list[j])[int(assignments[i, j]), :]

    #for i in range(new_assignments.shape[0]):
    #    for j in range(new_assignments.shape[1]):
    #        if new_assignments[i,j] != 0 and j in evaluated_trials:
    #            list_value = evaluated_trials.index(j)
    #            C_matrix[i][timeline[j][1]:timeline[j+1][1]] = (C_list[list_value])[int(new_assignments[i, j]), :]


    cnm_registration = estimates(A = spatial_union,C = C_matrix)
    with open(output_file_path, 'wb') as output_file:
        pickle.dump(cnm_registration, output_file, pickle.HIGHEST_PROTOCOL)

    for idx, row in df.iterrows():
        df.loc[idx, 'registration_output'] = str(output)
        df.loc[idx, 'registration_parameters'] = str(parameters)

    return df