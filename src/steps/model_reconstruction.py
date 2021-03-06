"""

Created on Thrus Jun  11 13:29:00 2020

@author: Melisa


This is the first function to make a video reconstruction from the model.
This file produces one video with the modeled data for all trials in one session
and saves it (preferably in the sparse representation)

"""

import os
import logging
import datetime
import numpy as np
import pickle
import math

import caiman as cm
import cv2
from caiman.base.rois import com
from caiman.source_extraction.cnmf.cnmf import load_CNMF
import src.data_base_manipulation as db
import src.steps.normalized_traces as normalization
from caiman.motion_correction import high_pass_filter_space


def run_model_reconstruction(selected_rows, parameters):

    '''
    This is the model reconstruction function. Is is supposed to be run after trial wise component evaluation.
    Model reconstruction takes place after component evaluation. It produces one big representation of the entire trials
    that are selected by the input selected rows.

    :param selected_rows: state of the data base containing all the trials that were aligned and source extracted.
    '''

    step_index = 7
    df = selected_rows.copy()
    gSig_filt = parameters['gSig_filt']
    re_sf = parameters['downsample_rate']

    #for i in range(len(df)):
    #    index = df.iloc[i].name
    #    row_new = db.set_version_analysis('model',df.iloc[i].copy())
    #    df = db.append_to_or_merge_with_states_df(df, row_new)
    #df = df.query('model_v == ' + f'{1}')

    #try:
    #    df.reset_index()[['session','trial', 'is_rest']].set_index(['session','trial', 'is_rest'], verify_integrity=True)
    #except ValueError:
    #    logging.error('You passed multiple of the same trial in the dataframe df')
    #   return df

    if parameters['session_wise'] == False:
        data_dir = os.environ['DATA_DIR'] + 'data/interim/model_reconstruction/trial_wise/main/'
    else:
        data_dir = os.environ['DATA_DIR'] + 'data/interim/model_reconstruction/session_wise/main/'

    row = df.iloc[0]
    file_name = db.create_file_name(step_index, row.name)
    output_tif_file_path = data_dir + f"{file_name}.tif"

    ##create the dictionary with metadata information
    output = {
        'main': output_tif_file_path,
        'meta': {
            'analysis': {
                'analyst': os.environ['ANALYST'],
                'date': datetime.datetime.today().strftime("%m-%d-%Y"),
                'time': datetime.datetime.today().strftime("%H:%M:%S")
            },
            'duration': {}
        }
    }

    ## create a list of modeled videos
    model_list = []  ## list of model videos containing A*C matrix for each trial
    evaluated_trials = []
    for i in range(len(df)):
        row = df.iloc[i]
        if type(row['source_extraction_output'] ) == str:
            evaluated_trials.append((df.iloc[i].name[2] - 1) * 2 + df.iloc[i].name[3])  ## number that goes from 0 to 42

            component_evaluation_hdf5_file_path = eval(row['source_extraction_output'])['main']   # cnmf path
            corr_path = eval(row['source_extraction_output'])['meta']['corr']['main']                # corr image path
            cnm = load_CNMF(component_evaluation_hdf5_file_path)                                     # load cnmf object
            cn_filter = np.load(db.get_file(corr_path))                                              # load corr image

            # normalize calcium traces for every trial, and substract negative baseline
            if cnm.estimates.bl is None:
                raw_normed, cnm_normed, res_normed, s_normed, noise_levels =  normalization.normalize_traces(cnm.estimates.C,
                                                                                                            cnm.estimates.YrA,
                                                                                                            cnm.estimates.S,
                                                                                                            1,
                                                                                                            offset_method="denoised_floor")
            else:
                raw_normed, cnm_normed, res_normed, s_normed, noise_levels =  normalization.normalize_traces(cnm.estimates.C - cnm.estimates.bl[:,np.newaxis],
                                                                                                            cnm.estimates.YrA,
                                                                                                            cnm.estimates.S,
                                                                                                            1,
                                                                                                            offset_method="denoised_floor")
            # recreate video with model
            dims = cn_filter.shape
            Y_rec = cnm.estimates.A.dot(cnm_normed)
            Y_rec = Y_rec.reshape(dims + (-1,), order='F')
            Y_rec = Y_rec.transpose([2, 0, 1])

            # gaussian filter the videos
            Y_rec_filtered = high_pass_filter_space(Y_rec, gSig_filt)
            # downsample video
            img = Y_rec_filtered[0, :, :]
            scale_percent = 10  # percent of original size
            width = int(img.shape[1] * scale_percent / 100)
            height = int(img.shape[0] * scale_percent / 100)
            dim = (width, height)
            Y_rec_downsample = np.zeros([Y_rec_filtered.shape[0],dim[1],dim[0]])
            for i in range(Y_rec_filtered.shape[0]):
                Y_rec_downsample[i,:,:] =  cv2.resize(Y_rec_filtered[i,:,:], dim, interpolation=cv2.INTER_AREA)

            # add trial video to a list
            model_list.append(cm.movie(Y_rec_downsample))

    # use cn.concatenate to create one temporal video.
    complete_model = cm.concatenate(model_list, axis=0)

    complete_model.save(output_tif_file_path)

    for idx, row in df.iterrows():
        df.loc[idx, 'model_output'] = str(output)

    return df

