#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


@author: Melisa Maidana


Functions in this python file are related to plotting different stages of the calcium imaging analysis pipeline.

Most of the save the result in the corresponding folder of the particular step.

"""


import pylab as pl
import caiman as cm
import matplotlib.pyplot as plt
import math
import numpy as np
from caiman.motion_correction import high_pass_filter_space
from caiman.source_extraction.cnmf.cnmf import load_CNMF
import src.data_base_manipulation as db
import src.analysis.metrics as metrics
import logging
import os
import datetime
import src.analysis_files_manipulation as fm
from caiman.source_extraction.cnmf.initialization import downscale
import logging
from random import randrange

def plot_movie_frame(row):
    '''
    This function creates an image for visual inspection of cropping points.
    :param row: dictionary with all relevant information about state of analysis
    :return: none
    '''
    output = row['decoding_output']
    decoded_file = eval(output)['main']
    m = cm.load(decoded_file)
    #print(m.shape)
    pl.imshow(m[0,:,:],cmap='gray')
    return

def plot_movie_frame_cropped(row):
    '''
    This function creates an image for visual inspections of cropped frame
    :param row: dictionary with all relevant information about state of analysis
    :return: none
    '''
    output = row['cropping_output']
    cropped_file = eval(output)['main']
    m = cm.load(cropped_file)
    #print(m.shape)
    pl.imshow(m[0,:,:],cmap='gray')
    return


def plot_temporal_evolution(row,session_wise = False):
    '''
    After decoding this plots the time evolution of some pixel values in the ROI, the histogram if pixel values and
    the ROI with the mark of the position for the randomly selected pixels
    If non specified it uses the trial video, if not it uses the aligned version of the videos.
    If alignement version == 2, then it uses the equalized version

    '''
    if session_wise:
        output = row['alignment_output']
    else:
        output = row['decoding_output']
    decoded_file = eval(output)['main']
    if row.name[7] == 2:
        decoded_file = eval(output)['equalization']['main']
    movie_original = cm.load(decoded_file)

    figure = plt.figure(constrained_layout=True)
    gs = figure.add_gridspec(5, 6)

    figure_ax1 = figure.add_subplot(gs[0:2, 0:3])
    figure_ax1.set_title('ROI: ' + f"mouse_{row.name[0]}", fontsize = 15)
    figure_ax1.set_yticks([])
    figure_ax1.set_xticks([])

    figure_ax2 = figure.add_subplot(gs[2:5, 0:3])
    figure_ax2.set_xlabel('Time [frames]', fontsize = 15)
    figure_ax2.set_ylabel('Pixel value', fontsize = 15)
    figure_ax2.set_title('Temporal Evolution', fontsize = 15)

    figure_ax1.imshow(movie_original[0,:,:], cmap = 'gray')
    color = ['b', 'r' , 'g', 'c', 'm']
    for i in range(5):
        x = randrange(movie_original.shape[1]-5)+5
        y = randrange(movie_original.shape[2]-5)+5
        [x_, _x, y_, _y] = [x-5,x+5,y-5,y+5]
        rect = Rectangle((y_, x_), _y - y_, _x - x_, fill=False, color=color[i], linestyle='-', linewidth=2)
        figure_ax1.add_patch(rect)
        figure_ax2.plot(np.arange(0,movie_original.shape[0],), movie_original[:,x,y], color = color[i])

        figure_ax_i = figure.add_subplot(gs[i, 4:])
        figure_ax_i.hist(movie_original[:,x,y],20, color = color[i])
        figure_ax_i.set_xlim((500,1200))
        figure_ax_i.set_ylabel('#')
        figure_ax_i.set_xlabel('Pixel value')

    path = '/home/sebastian/Documents/Melisa/calcium_imaging_analysis/data/interim/decoding/meta/'
    name = db.create_file_name(1,row.name)
    figure.savefig(path + name + '.png')

    return

def funcitontocreate:

    '''
    figure, axes = plt.subplots(3,2)
    axes[0,0].imshow(movie_original[0,:,:], cmap = 'gray')
    axes[0,0].set_title('ROI')

    [x_, _x, y_, _y] = [15,25,15,25]
    rect = Rectangle((y_, x_), _y - y_, _x - x_, fill=False, color='r', linestyle='-', linewidth=2)
    axes[0, 0].add_patch(rect)

    for i in range(len(m_list)):
        axes[0,1].hist(m_list[i][:,20,20],bins = np.arange(600,1000))
    axes[0,1].set_xlim((500,1000))
    axes[0,1].set_ylabel('#')
    axes[0,1].set_xlabel('Pixel value')
    axes[0,1].set_title('RED SQUARE')
    axes[0,1].legend(['1','2','3','4','5'])

    [x_, _x, y_, _y] = [15,25,135,145]
    rect = Rectangle((y_, x_), _y - y_, _x - x_, fill=False, color='b', linestyle='-', linewidth=2)
    axes[0, 0].add_patch(rect)

    for i in range(len(m_list)):
        axes[1,0].hist(m_list[i][:,20,140],bins = np.arange(600,1000))
    axes[1,0].set_xlim((500,1000))
    axes[1,0].set_ylabel('#')
    axes[1,0].set_xlabel('Pixel value')
    axes[1,0].set_title('BLUE SQUARE')
    axes[1,0].legend(['1','2','3','4','5'])



    [x_, _x, y_, _y] = [135,145,135,145]
    rect = Rectangle((y_, x_), _y - y_, _x - x_, fill=False, color='g', linestyle='-', linewidth=2)
    axes[0, 0].add_patch(rect)
    for i in range(len(m_list)):
        axes[1,1].hist(m_list[i][:,140,140],bins = np.arange(600,1000))
    axes[1,1].set_xlim((500,1000))
    axes[1,1].set_ylabel('#')
    axes[1,1].set_xlabel('Pixel value')
    axes[1,1].set_title('GREEN SQUARE')
    axes[1,1].legend(['1','2','3','4','5'])

    [x_, _x, y_, _y] = [85,95,25,35]
    rect = Rectangle((y_, x_), _y - y_, _x - x_, fill=False, color='m', linestyle='-', linewidth=2)
    axes[0, 0].add_patch(rect)
    for i in range(len(m_list)):
        axes[2,0].hist(m_list[i][:,90,30],bins = np.arange(600,1000))
    axes[2,0].set_xlim((500,1000))
    axes[2,0].set_ylabel('#')
    axes[2,0].set_xlabel('Pixel value')
    axes[2,0].set_title('MAGENTA SQUARE')
    axes[2,0].legend(['1','2','3','4','5'])


    [x_, _x, y_, _y] = [25,35,85,95]
    rect = Rectangle((y_, x_), _y - y_, _x - x_, fill=False, color='c', linestyle='-', linewidth=2)
    axes[0, 0].add_patch(rect)
    for i in range(len(m_list)):
        axes[2,1].hist(m_list[i][:,30,90],bins = np.arange(600,1000))
    axes[2,1].set_xlim((500,1000))
    axes[2,1].set_ylabel('#')
    axes[2,1].set_xlabel('Pixel value')
    axes[2,1].set_title('CYAN SQUARE')
    axes[2,1].legend(['1','2','3','4','5'])
    '''

    return

def get_fig_gSig_filt_vals(row, gSig_filt_vals):
    '''
    Plot original cropped frame and several versions of spatial filtering for comparison
    :param row: analisis state row for which the filtering is computed
    :param gSig_filt_vals: array containing size of spatial filters that will be applyed
    :return: figure
    '''
    output = row['cropping_output']
    cropped_file = eval(output)['main']
    m = cm.load(cropped_file)
    temp = cm.motion_correction.bin_median(m)
    N = len(gSig_filt_vals)
    fig, axes = plt.subplots(int(math.ceil((N + 1) / 2)), 2)
    axes[0, 0].imshow(temp, cmap='gray')
    axes[0, 0].set_title('unfiltered')
    axes[0, 0].axis('off')
    for i in range(0, N):
        gSig_filt = gSig_filt_vals[i]
        m_filt = [high_pass_filter_space(m_, (gSig_filt, gSig_filt)) for m_ in m]
        temp_filt = cm.motion_correction.bin_median(m_filt)
        axes.flatten()[i + 1].imshow(temp_filt, cmap='gray')
        axes.flatten()[i + 1].set_title(f'gSig_filt = {gSig_filt}')
        axes.flatten()[i + 1].axis('off')
    if N + 1 != axes.size:
        for i in range(N + 1, axes.size):
            axes.flatten()[i].axis('off')

    # Get output file paths
    index = row.name
    data_dir = 'data/interim/motion_correction/'
    step_index = db.get_step_index('motion_correction')
    file_name = db.create_file_name(step_index, index)
    output_meta_gSig_filt = data_dir + f'meta/figures/frame_gSig_filt/{file_name}.png'

    fig.savefig(output_meta_gSig_filt)

    return fig

def plot_crispness_for_parameters(selected_rows = None):
    '''
    This function plots crispness for all the selected rows motion correction states. The idea is to compare crispness results
    :param selected_rows: analysis states for which crispness is required to be ploted
    :return: figure that is also saved
    '''
    crispness_mean_original,crispness_corr_original, crispness_mean, crispness_corr = metrics.compare_crispness(selected_rows)
    total_states_number = len(selected_rows)

    fig, axes = plt.subplots(1,2)
    axes[0].set_title('Summary image = Mean')
    axes[0].plot(np.arange(1,total_states_number,1),crispness_mean_original)
    axes[0].plot(np.arange(1,total_states_number,1),crispness_mean)
    axes[0].legend(('Original', 'Motion_corrected'))
    axes[0].set_ylabel('Crispness')
    #axes[0].set_xlabel('#')

    axes[1].set_title('Summary image = Corr')
    axes[1].plot(np.arange(1,total_states_number,1),crispness_corr_original)
    axes[1].plot(np.arange(1,total_states_number,1),crispness_corr)
    axes[1].legend(('Original', 'Motion_corrected'))
    axes[1].set_ylabel('Crispness')
    #axes[0].set_xlabel('#')

    # Get output file paths
    index = selected_rows.iloc[0].name
    data_dir = 'data/interim/motion_correction/'
    step_index = db.get_step_index('motion_correction')
    file_name = db.create_file_name(step_index, index)
    output_meta_crispness = data_dir + f'meta/figures/crispness/{file_name}.png'

    fig.savefig(output_meta_crispness)
    return fig

def plot_corr_pnr(mouse_row, parameters_source_extraction):
    '''
    Plots the summary images correlation and pnr. Also the pointwise product between them (used in Caiman paper Zhou
    et al 2018)
    :param mouse_row:
    :param parameters_source_extraction: parameters that will be used for source
    extraction. the relevant parameter here are min_corr and min_pnr because the source extraction algorithm is
    initialized (initial cell templates) in all values that surpasses that threshold
    :return:  figure
    '''

    input_mmap_file_path = eval(mouse_row.loc['motion_correction_output'])['main']

    # Load memory mappable input file
    if os.path.isfile(input_mmap_file_path):
        Yr, dims, T = cm.load_memmap(input_mmap_file_path)
        #        logging.debug(f'{index} Loaded movie. dims = {dims}, T = {T}.')
        images = Yr.T.reshape((T,) + dims, order='F')
    else:
        logging.warning(f'{mouse_row.name} .mmap file does not exist. Cancelling')

    # Determine output paths
    step_index = db.get_step_index('motion_correction')
    data_dir = 'data/interim/source_extraction/trial_wise/'

    # Check if the summary images are already there
    gSig = parameters_source_extraction['gSig'][0]
    corr_npy_file_path, pnr_npy_file_path = fm.get_corr_pnr_path(mouse_row.name, gSig_abs=(gSig, gSig))

    if corr_npy_file_path != None and os.path.isfile(corr_npy_file_path):
        # Already computed summary images
        logging.info(f'{mouse_row.name} Already computed summary images')
        cn_filter = np.load(corr_npy_file_path)
        pnr = np.load(pnr_npy_file_path)
    else:
        # Compute summary images
        t0 = datetime.datetime.today()
        logging.info(f'{mouse_row.name} Computing summary images')
        cn_filter, pnr = cm.summary_images.correlation_pnr(images[::1], gSig=parameters_source_extraction['gSig'][0],
                                                           swap_dim=False)
        # Saving summary images as npy files
        corr_npy_file_path = data_dir + f'meta/corr/{db.create_file_name(3, mouse_row.name)}_gSig_{gSig}.npy'
        pnr_npy_file_path = data_dir + f'meta/pnr/{db.create_file_name(3, mouse_row.name)}_gSig_{gSig}.npy'
        with open(corr_npy_file_path, 'wb') as f:
            np.save(f, cn_filter)
        with open(pnr_npy_file_path, 'wb') as f:
            np.save(f, pnr)

    fig = plt.figure(figsize=(15, 15))
    min_corr = round(parameters_source_extraction['min_corr'], 2)
    min_pnr = round(parameters_source_extraction['min_pnr'], 1)
    max_corr = round(cn_filter.max(), 2)
    max_pnr= 20

    # continuous
    cmap = 'viridis'
    fig, axes = plt.subplots(1, 3, sharex=True)

    corr_fig = axes[0].imshow(np.clip(cn_filter,min_corr,max_corr), cmap=cmap)
    axes[0].set_title('Correlation')
    fig.colorbar(corr_fig, ax=axes[0])
    pnr_fig = axes[1].imshow(np.clip(pnr,min_pnr,max_pnr), cmap=cmap)
    axes[1].set_title('PNR')
    fig.colorbar(pnr_fig, ax=axes[1])
    combined = cn_filter*pnr
    max_combined = 10
    min_combined = np.min(combined)
    corr_pnr_fig = axes[2].imshow(np.clip(cn_filter*pnr,min_combined,max_combined), cmap=cmap)
    axes[2].set_title('Corr * PNR')
    fig.colorbar(corr_pnr_fig, ax=axes[2])

    fig_dir = 'data/interim/source_extraction/trial_wise/meta/'
    fig_name= fig_dir + f'figures/corr_pnr/{db.create_file_name(3, mouse_row.name)}_gSig_{gSig}.png'
    fig.savefig(fig_name)

    return fig

def plot_corr_pnr_binary(mouse_row, corr_limits, pnr_limits, parameters_source_extraction, session_wise = False,
                         alignment = False, equalization = False):
    '''
    Plot 2 matrix of binary selected and not selected seeds for different corr_min and pnr_min
    :param mouse_row: analysis states data
    :param corr_limits: array of multiple values of corr_min to test
    :param pnr_limits: arrey of multiple values of pnr_min to test
    :param parameters_source_extraction: dictionary with parameters
    :return: figure pointer
    '''

    input_mmap_file_path = eval(mouse_row.loc['motion_correction_output'])['main']
    if alignment:
        input_mmap_file_path = eval(mouse_row.loc['alignment_output'])['main']
    if equalization:
        input_mmap_file_path = eval(mouse_row['alignment_output'])['equalizing_output']['main']

    # Load memory mappable input file
    if os.path.isfile(input_mmap_file_path):
        Yr, dims, T = cm.load_memmap(input_mmap_file_path)
        #        logging.debug(f'{index} Loaded movie. dims = {dims}, T = {T}.')
        images = Yr.T.reshape((T,) + dims, order='F')
    else:
        logging.warning(f'{mouse_row.name} .mmap file does not exist. Cancelling')

    # Determine output paths
    step_index = db.get_step_index('motion_correction')
    data_dir = 'data/interim/source_extraction/trial_wise/'

    # Check if the summary images are already there
    gSig = parameters_source_extraction['gSig'][0]
    corr_npy_file_path, pnr_npy_file_path = fm.get_corr_pnr_path(mouse_row.name, gSig_abs=(gSig, gSig))

    if corr_npy_file_path != None and os.path.isfile(corr_npy_file_path):
        # Already computed summary images
        logging.info(f'{mouse_row.name} Already computed summary images')
        cn_filter = np.load(corr_npy_file_path)
        pnr = np.load(pnr_npy_file_path)
    else:
        # Compute summary images
        t0 = datetime.datetime.today()
        logging.info(f'{mouse_row.name} Computing summary images')
        cn_filter, pnr = cm.summary_images.correlation_pnr(images[::1], gSig=parameters_source_extraction['gSig'][0],
                                                           swap_dim=False)
        # Saving summary images as npy files
        corr_npy_file_path = data_dir + f'meta/corr/{db.create_file_name(3, mouse_row.name)}_gSig_{gSig}.npy'
        pnr_npy_file_path = data_dir + f'meta/pnr/{db.create_file_name(3, mouse_row.name)}_gSig_{gSig}.npy'
        with open(corr_npy_file_path, 'wb') as f:
            np.save(f, cn_filter)
        with open(pnr_npy_file_path, 'wb') as f:
            np.save(f, pnr)

    fig1 = plt.figure(figsize=(50, 50))

    combined_image = cn_filter * pnr
    fig1, axes1 = plt.subplots(len(corr_limits), len(pnr_limits), sharex=True)
    fig2, axes2 = plt.subplots(len(corr_limits), len(pnr_limits), sharex=True)
    fig3, axes3 = plt.subplots(len(corr_limits), len(pnr_limits), sharex=True)

    ii=0
    for min_corr in corr_limits:
        min_corr = round(min_corr,2)
        jj=0
        for min_pnr in pnr_limits:
            min_pnr = round(min_pnr,2)
            # binary
            limit = min_corr * min_pnr
            axes1[ii, jj].imshow(combined_image> limit, cmap='binary')
            axes1[ii, jj].set_title(f'{min_corr}')
            axes1[ii, jj].set_ylabel(f'{min_pnr}')
            axes2[ii, jj].imshow(cn_filter > min_corr, cmap='binary')
            axes2[ii, jj].set_title(f'{min_corr}')
            axes2[ii, jj].set_ylabel(f'{min_pnr}')
            axes3[ii, jj].imshow(pnr> min_pnr, cmap='binary')
            axes3[ii, jj].set_title(f'{min_corr}')
            axes3[ii, jj].set_ylabel(f'{min_pnr}')
            jj=jj+1
        ii=ii+1

    fig_dir = 'data/interim/source_extraction/trial_wise/meta/'
    if session_wise:
        fig_dir = 'data/interim/source_extraction/session_wise/meta/'
    fig_name= fig_dir + f'figures/min_corr_pnr/{db.create_file_name(3, mouse_row.name)}_gSig_{gSig}_comb.png'
    fig1.savefig(fig_name)


    fig_name= fig_dir + f'figures/min_corr_pnr/{db.create_file_name(3, mouse_row.name)}_gSig_{gSig}_corr.png'
    fig2.savefig(fig_name)

    fig_name= fig_dir + f'figures/min_corr_pnr/{db.create_file_name(3, mouse_row.name)}_gSig_{gSig}_pnr.png'
    fig3.savefig(fig_name)

    return fig1,fig2,fig3



def plot_histogram(position, value , title = 'title', xlabel = 'x_label', ylabel = 'y_label'):
    '''
    This function plots a histogram for...
    :param position: x marks
    :param value: y marks
    :param title:
    :param xlabel:
    :param ylabel:
    :return:
    '''

    fig, axes = plt.subplots(1, 1, sharex=True)

    normalization = sum(value)
    axes.plot(position, value / normalization)
    axes.set_title(title)
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_ylim(0, np.max(value/normalization) + 0.01 *np.max(value/normalization))

    return fig

def create_video(row, time_cropping, session_wise = False):

    '''
    This fuction creates a complete video with raw movie (motion corrected), source extracted cells and source extraction + background.
    :param row: pandas dataframe containing the desired processing information to create the video. It can use the session_wise or trial_wise video.
    :return:
    '''

    if session_wise:
        input_mmap_file_path = eval(row.loc['alignment_output'])['main']
    else:
        input_mmap_file_path = eval(row.loc['motion_correction_output'])['main']

    #load the mmap file
    Yr, dims, T = cm.load_memmap(input_mmap_file_path)
    logging.debug(f'{row.name} Loaded movie. dims = {dims}, T = {T}.')
    #create a caiman movie with the mmap file
    images = Yr.T.reshape((T,) + dims, order='F')
    images = cm.movie(images)

    #load source extraction result
    output = eval(row.loc['source_extraction_output'])
    cnm_file_path = output['main']
    cnm = load_CNMF(db.get_file(cnm_file_path))

    #estimate the background from the extraction
    W, b0 = cm.source_extraction.cnmf.initialization.compute_W(Yr, cnm.estimates.A.toarray(), cnm.estimates.C,
                                                               cnm.estimates.dims, 1.4 * 5, ssub=2)
    cnm.estimates.W = W
    cnm.estimates.b0 = b0
    # this part could be use with the lastest caiman version
    # movie_dir = '/home/sebastian/Documents/Melisa/calcium_imaging_analysis/data/processed/movies/'
    # file_name = db.create_file_name(5,row.name)
    # cnm.estimates.play_movie(cnm.estimates, images, movie_name= movie_dir + file_name + '.avi')

    frame_range = slice(None, None, None)
    # create a movie with the model : estimated A and C matrix
    Y_rec = cnm.estimates.A.dot(cnm.estimates.C[:, frame_range])
    Y_rec = Y_rec.reshape(dims + (-1,), order='F')
    Y_rec = Y_rec.transpose([2, 0, 1])
    # convert the variable to a caiman movie type
    Y_rec = cm.movie(Y_rec)

    ## this part of the function is a copy from a caiman version
    ssub_B = int(round(np.sqrt(np.prod(dims) / W.shape[0])))
    B = images[frame_range].reshape((-1, np.prod(dims)), order='F').T - \
        cnm.estimates.A.dot(cnm.estimates.C[:, frame_range])
    if ssub_B == 1:
        B = b0[:, None] + W.dot(B - b0[:, None])
    else:
        B = b0[:, None] + (np.repeat(np.repeat(W.dot(
            downscale(B.reshape(dims + (B.shape[-1],), order='F'),
                      (ssub_B, ssub_B, 1)).reshape((-1, B.shape[-1]), order='F') -
            downscale(b0.reshape(dims, order='F'),
                      (ssub_B, ssub_B)).reshape((-1, 1), order='F'))
            .reshape(
            ((dims[0] - 1) // ssub_B + 1, (dims[1] - 1) // ssub_B + 1, -1), order='F'),
            ssub_B, 0), ssub_B, 1)[:dims[0], :dims[1]].reshape(
            (-1, B.shape[-1]), order='F'))
    B = B.reshape(dims + (-1,), order='F').transpose([2, 0, 1])

    Y_rec_2 = Y_rec + B
    Y_res = images[frame_range] - Y_rec - B

    images_np = np.zeros((time_cropping[1]-time_cropping[0],images.shape[1],images.shape[2]))
    images_np = images[time_cropping[0]:time_cropping[1],:,:]
    images_np = images_np / np.max(images_np)
    images_np = cm.movie(images_np)

    Y_rec_np = np.zeros((time_cropping[1]-time_cropping[0],images.shape[1],images.shape[2]))
    Y_rec_np = Y_rec[time_cropping[0]:time_cropping[1],:,:]
    Y_rec_np = Y_rec_np / np.max(Y_rec_np)
    Y_rec_np = cm.movie(Y_rec_np)

    Y_res_np = np.zeros((time_cropping[1]-time_cropping[0],images.shape[1],images.shape[2]))
    Y_res_np = Y_res[time_cropping[0]:time_cropping[1],:,:]
    Y_res_np = Y_res_np / np.max(Y_res_np)
    Y_res_np = cm.movie(Y_res_np)

    B_np = np.zeros((time_cropping[1]-time_cropping[0],images.shape[1],images.shape[2]))
    B_np = B[time_cropping[0]:time_cropping[1],:,:]
    B_np = B_np / np.max(B_np)
    B_np = cm.movie(B_np)

    mov1 = cm.concatenate((images_np, Y_rec_np), axis=2)

    mov2 = cm.concatenate((B_np, Y_res_np), axis=2)

    mov = cm.concatenate((mov1, mov2), axis=1)

    figure_path = '/home/sebastian/Documents/Melisa/calcium_imaging_analysis/data/interim/movies/'
    figure_name = db.create_file_name(5,row.name)
    #mov.save(figure_path+figure_name+'.tif')
    mov.save(figure_path+figure_name+'_'+f'{time_cropping[0]}' + '_' + f'{time_cropping[1]}'+'.tif')

    return

def plot_source_extraction_result(mouse_row_new):

    '''
    Generates and saves a contour plot and a trace plot for the specific mouse_row
    '''
    corr_min = round(eval(mouse_row_new['source_extraction_parameters'])['min_corr'], 1)
    pnr_min = round(eval(mouse_row_new['source_extraction_parameters'])['min_pnr'], 1)

    output_source_extraction = eval(mouse_row_new.loc['source_extraction_output'])
    corr_path = output_source_extraction['meta']['corr']['main']
    cn_filter = np.load(db.get_file(corr_path))

    cnm_file_path = output_source_extraction['main']
    cnm = load_CNMF(db.get_file(cnm_file_path))

    figure, axes = plt.subplots(1)
    axes.imshow(cn_filter)
    coordinates = cm.utils.visualization.get_contours(cnm.estimates.A, np.shape(cn_filter), 0.2, 'max')
    for c in coordinates:
        v = c['coordinates']
        c['bbox'] = [np.floor(np.nanmin(v[:, 1])), np.ceil(np.nanmax(v[:, 1])),
                     np.floor(np.nanmin(v[:, 0])), np.ceil(np.nanmax(v[:, 0]))]
        axes.plot(*v.T, c='w')
    axes.set_title('min_corr = ' + f'{corr_min}')
    axes.set_ylabel('min_pnr = ' + f'{pnr_min}')

    fig_dir = 'data/interim/source_extraction/session_wise/meta/figures/contours/'
    file_name = db.create_file_name(3, mouse_row_new.name)
    figure.savefig(fig_dir + file_name + '.png')
    ## up to here

    fig, ax = plt.subplots(1)
    C = cnm.estimates.C
    C[0] += C[0].min()
    for i in range(1, len(C)):
        C[i] += C[i].min() + C[:i].max()
        ax.plot(C[i])
    ax.set_xlabel('t [frames]')
    ax.set_yticks([])
    ax.set_ylabel('activity')
    fig.set_size_inches([10., .3 * len(C)])

    fig_dir = 'data/interim/source_extraction/session_wise/meta/figures/traces/'
    fig_name = fig_dir + db.create_file_name(3, mouse_row_new.name) + '.png'
    fig.savefig(fig_name)

    return


def plot_source_extraction_result_specific_cell(mouse_row_new, cell_number):

    '''
    (Still need to be finished) !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    THERE IS AN ERROR IN THE
    In the first plot shows correlation image and contour of the selected neurons.
    In the second plot shows the traces for the selected neurons.
    :param mouse_row_new: data base row
    :param cell_number: array with the cells that are selected to be ploted
    :return: None
    '''
    corr_min = round(eval(mouse_row_new['source_extraction_parameters'])['min_corr'], 1)
    pnr_min = round(eval(mouse_row_new['source_extraction_parameters'])['min_pnr'], 1)

    output_source_extraction = eval(mouse_row_new.loc['source_extraction_output'])
    corr_path = output_source_extraction['meta']['corr']['main']
    cn_filter = np.load(db.get_file(corr_path))

    cnm_file_path = output_source_extraction['main']
    cnm = load_CNMF(db.get_file(cnm_file_path))

    f, (a0, a1) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]})
    a0.imshow(cn_filter)
    coordinates = cm.utils.visualization.get_contours(cnm.estimates.A, np.shape(cn_filter), 0.2, 'max')
    for i in cell_number:
        v = coordinates[i]['coordinates']
        coordinates[i]['bbox'] = [np.floor(np.nanmin(v[:, 1])), np.ceil(np.nanmax(v[:, 1])),
                     np.floor(np.nanmin(v[:, 0])), np.ceil(np.nanmax(v[:, 0]))]
        a0.plot(*v.T, c='w')
    a0.set_title('Contour Plot')

    fig, ax = plt.subplots(1)
    C = cnm.estimates.C
    C[0] += C[0].min()
    for i in range(cell_number):
        C[i] += C[i].min() + C[:i].max()
        a1.plot(C[i])
    a1.set_xlabel('t [frames]')
    a1.set_yticks([])
    a1.set_title('Calcium Traces')
    fig.set_size_inches([10., .3 * len(C)])

    fig_dir = 'data/interim/source_extraction/session_wise/meta/figures/'
    fig_name = fig_dir + db.create_file_name(3, mouse_row_new.name) + '_example.png'
    f.savefig(fig_name)

    return


def plot_multiple_contours(rows, version = None , corr_array = None, pnr_array = None,session_wise = False):
    '''
    Plots different versions of contour images that change the initialization parameters for source extraction.
    The idea is to see the impact of different seed selection in the final source extraction result.
    :param row: one analysis state row
    :param version: array containing the version numbers of source extraction that will be plotted
    :param corr_array: array of the same length of version and pnr_array containing the min_corr values for those versions
    :param pnr_array: array of the same length of version and corr_array containing the min_pnr values for those versions
    :return: figure
    '''


    figure, axes = plt.subplots(len(corr_array), len(pnr_array), figsize=(15, 15))

    for ii in range(corr_array.shape[0]):
        for jj in range(pnr_array.shape[0]):
            version_number = ii *corr_array.shape[0] + jj + 1
            if version_number in version:
                new_row = rows.query('(source_extraction_v == ' + f'{version_number}' + ')')
                new_row = new_row.iloc[0]
                output = eval(new_row.loc['source_extraction_output'])
                cnm_file_path = output['main']
                cnm = load_CNMF(db.get_file(cnm_file_path))
                corr_path = output['meta']['corr']['main']
                cn_filter = np.load(db.get_file(corr_path))
                axes[ii, jj].imshow(cn_filter)
                coordinates = cm.utils.visualization.get_contours(cnm.estimates.A, np.shape(cn_filter), 0.2, 'max')
                for c in coordinates:
                    v = c['coordinates']
                    c['bbox'] = [np.floor(np.nanmin(v[:, 1])), np.ceil(np.nanmax(v[:, 1])),
                                 np.floor(np.nanmin(v[:, 0])), np.ceil(np.nanmax(v[:, 0]))]
                    axes[ii, jj].plot(*v.T, c='w')
                axes[ii, jj].set_title('min_corr = ' + f'{round(corr_array[ii],2)}')
                axes[ii, jj].set_ylabel('min_pnr = ' + f'{round(pnr_array[jj],2)}')

    fig_dir = 'data/interim/source_extraction/trial_wise/meta/figures/contours/'
    if session_wise:
        fig_dir = 'data/interim/source_extraction/session_wise/meta/figures/contours/'
    fig_name = fig_dir + db.create_file_name(3, new_row.name)+'_corr_min' + f'{round(corr_array[0],1)}'+ '_pnr_min'+f'{round(pnr_array[0],1)}' + '_.png'
    figure.savefig(fig_name)

    return figure


def plot_session_contours(selected_rows, version = None , corr_array = None, pnr_array = None):
    '''
    Plots different versions of contour images that change the initialization parameters for source extraction.
    The idea is to see the impact of different seed selection in the final source extraction result.
    :param selected_rows: rows corresponding to different trials
    :param version: array containing the version numbers of source extraction that will be plotted
    :param corr_array: array of the same length of version and pnr_array containing the min_corr values for those versions
    :param pnr_array: array of the same length of version and corr_array containing the min_pnr values for those versions
    :return: (saves multiple figures)
    '''

    for ii in range(corr_array.shape[0]):
        for jj in range(pnr_array.shape[0]):
            figure, axes = plt.subplots(len(selected_rows) / 5, 5, figsize=(50, 10*len(selected_rows) / 5))
            version_rows = selected_rows.query('(source_extraction_v == ' + f'{ii * len(corr_array.shape[0] + jj)}' + ')')
            for day in range(len(selected_rows)/5):
                for trial in range(5):
                    new_row = version_rows.iloc[day*5+trial]
                    output = eval(new_row.loc['source_extraction_output'])
                    cnm_file_path = output['main']
                    cnm = load_CNMF(db.get_file(cnm_file_path))
                    corr_path = output['meta']['corr']['main']
                    cn_filter = np.load(db.get_file(corr_path))
                    #axes[i].imshow(np.clip(cn_filter, min_corr, max_corr), cmap='viridis')
                    axes[day,trial].imshow(cn_filter)
                    coordinates = cm.utils.visualization.get_contours(cnm.estimates.A, np.shape(cn_filter), 0.2, 'max')
                    for c in coordinates:
                        v = c['coordinates']
                        c['bbox'] = [np.floor(np.nanmin(v[:, 1])), np.ceil(np.nanmax(v[:, 1])),
                                     np.floor(np.nanmin(v[:, 0])), np.ceil(np.nanmax(v[:, 0]))]
                        axes[day,trial].plot(*v.T, c='w')
                    axes[day,trial].set_title('Trial = ' + f'{i+1}',fontsize=30)
                    axes[day,trial].set_xlabel('#cells = ' + f'{cnm.estimates.A.shape[1]}',fontsize=30)

            figure.suptitle('min_corr = ' + f'{round(corr_array[ii],2)}' + 'min_pnr = ' + f'{round(pnr_array[jj],2)}', fontsize=50)

            fig_dir = 'data/interim/source_extraction/session_wise/meta/figures/contours/'
            fig_name = fig_dir + db.create_file_name(3, new_row.name)+'_version_' + f'{version[ii*len(pnr_array)+jj]}'+'.png'
            figure.savefig(fig_name)

    return



def plot_multiple_contours_session_wise(selected_rows, version = None , corr_array = None, pnr_array = None):
    '''
    Plots different versions of contour images that change the initialization parameters for source extraction.
    The idea is to see the impact of different seed selection in the final source extraction result.
    :param selected_rows: all analysis state selected
    :param version: array containing the version numbers of source extraction that will be plotted
    :param corr_array: array of the same length of version and pnr_array containing the min_corr values for those versions
    :param pnr_array: array of the same length of version and corr_array containing the min_pnr values for those versions
    :return: figure
    '''

    states_df = db.open_analysis_states_database()

    figure, axes = plt.subplots(len(corr_array), len(pnr_array), figsize=(15, 15))

    color = ['w','b','r','m','c']
    for row in range(len(selected_rows)):
        mouse_row = selected_rows.iloc[row]
        index = mouse_row.name
        output = eval(mouse_row.loc['source_extraction_output'])
        corr_path = output['meta']['corr']['main']
        cn_filter = np.load(db.get_file(corr_path))
        for ii in range(corr_array.shape[0]):
            for jj in range(pnr_array.shape[0]):
                axes[ii, jj].imshow(cn_filter)
                new_row = db.select(states_df, 'component_evaluation', mouse=index[0], session=index[1],
                                    trial=index[2], is_rest=index[3], cropping_v=index[5], motion_correction_v=index[6],
                                    source_extraction_v=version[ii * len(pnr_array) + jj])
                new_row = new_row.iloc[0]
                output = eval(new_row.loc['source_extraction_output'])
                cnm_file_path = output['main']
                cnm = load_CNMF(db.get_file(cnm_file_path))
                coordinates = cm.utils.visualization.get_contours(cnm.estimates.A, np.shape(cn_filter), 0.2, 'max')
                for c in coordinates:
                    v = c['coordinates']
                    c['bbox'] = [np.floor(np.nanmin(v[:, 1])), np.ceil(np.nanmax(v[:, 1])),
                                 np.floor(np.nanmin(v[:, 0])), np.ceil(np.nanmax(v[:, 0]))]
                    axes[ii, jj].plot(*v.T, c = color[row])
                axes[ii, jj].set_title('min_corr = ' + f'{round(corr_array[ii],2)}')
                axes[ii, jj].set_ylabel('min_pnr = ' + f'{round(pnr_array[jj],2)}')


    fig_dir = 'data/interim/source_extraction/session_wise/meta/figures/contours/'
    fig_name = fig_dir + db.create_file_name(3, new_row.name)+'_corr_min' + f'{round(corr_array[0],1)}'+ '_pnr_min'+f'{round(pnr_array[0],1)}' + '_all.png'
    figure.savefig(fig_name)

    return figure


def plot_multiple_contours_session_wise_evaluated(selected_rows):

    ## IN DEVELOPMENT!!!!!!!
    '''
    Plots different versions of contour images that change the initialization parameters for source extraction.
    The idea is to see the impact of different seed selection in the final source extraction result.
    :param selected_rows: all analysis state selected
    :return: figure
    '''

    figure, axes = plt.subplots(3, 5, figsize=(50, 30))

    for row in range(len(selected_rows)):
        mouse_row = selected_rows.iloc[row]
        index = mouse_row.name
        output = eval(mouse_row.loc['source_extraction_output'])
        corr_path = output['meta']['corr']['main']
        cn_filter = np.load(db.get_file(corr_path))
        axes[0,row].imshow(cn_filter)
        axes[1,row].imshow(cn_filter)
        axes[2,row].imshow(cn_filter)
        output = eval(mouse_row.loc['source_extraction_output'])
        cnm_file_path = output['main']
        cnm = load_CNMF(db.get_file(cnm_file_path))
        coordinates = cm.utils.visualization.get_contours(cnm.estimates.A, np.shape(cn_filter), 0.2, 'max')
        for c in coordinates:
            v = c['coordinates']
            c['bbox'] = [np.floor(np.nanmin(v[:, 1])), np.ceil(np.nanmax(v[:, 1])),
                         np.floor(np.nanmin(v[:, 0])), np.ceil(np.nanmax(v[:, 0]))]
            axes[0,row].plot(*v.T, c = 'w', linewidth=3)
        axes[0,row].set_title('Trial = ' + f'{row}')
        axes[0,row].set_ylabel('')

        output = eval(mouse_row.loc['component_evaluation_output'])
        cnm_file_path = output['main']
        cnm = load_CNMF(db.get_file(cnm_file_path))
        idx = cnm.estimates.idx_components
        coordinates = cm.utils.visualization.get_contours(cnm.estimates.A[:, idx], np.shape(cn_filter), 0.2, 'max')
        for c in coordinates:
            v = c['coordinates']
            c['bbox'] = [np.floor(np.nanmin(v[:, 1])), np.ceil(np.nanmax(v[:, 1])),
                         np.floor(np.nanmin(v[:, 0])), np.ceil(np.nanmax(v[:, 0]))]
            axes[1,row].plot(*v.T, c='b', linewidth=3)

        idx_b = cnm.estimates.idx_components_bad
        coordinates_b = cm.utils.visualization.get_contours(cnm.estimates.A[:,idx_b], np.shape(cn_filter), 0.2, 'max')

        for c in coordinates_b:
            v = c['coordinates']
            c['bbox'] = [np.floor(np.nanmin(v[:, 1])), np.ceil(np.nanmax(v[:, 1])),
                         np.floor(np.nanmin(v[:, 0])), np.ceil(np.nanmax(v[:, 0]))]
            axes[2,row].plot(*v.T, c='r', linewidth=3)


    source_extraction_parameters = eval(mouse_row['source_extraction_parameters'])
    corr_lim = source_extraction_parameters['min_corr']
    pnr_lim = source_extraction_parameters['min_pnr']
    component_evaluation_parameters = eval(mouse_row['component_evaluation_parameters'])
    pcc = component_evaluation_parameters['rval_thr']
    SNR = component_evaluation_parameters['min_SNR']
    figure.suptitle('Corr = ' + f'{corr_lim}' + 'PNR = ' + f'{pnr_lim}' + 'PCC = ' + f'{pcc}' + 'SNR = ' + f'{SNR}',
                    fontsize=50)
    fig_dir = 'data/interim/component_evaluation/session_wise/meta/figures/contours/'
    fig_name = fig_dir + db.create_file_name(3, index)+'_Corr = ' + f'{corr_lim}' + '_PNR = ' + f'{pnr_lim}' + '_PCC = ' + f'{pcc}' + '_SNR = ' + f'{SNR}' +'_.png'
    figure.savefig(fig_name)

    return figure


def plot_traces_multiple(rows, version = None , corr_array = None, pnr_array = None, session_wise = False):
    '''
    Plots different versions of contour images that change the inicialization parameters for source extraccion.
    The idea is to see the impact of different seed selection in the final source extraction result.
    :param row: one analysis state row
    :param version: array containing the version numbers of source extraction that will be ploted
    :param corr_array: array of the same length of version and pnr_array containing the min_corr values for those versions
    :param pnr_array: array of the same length of version and corr_array containing the min_pnr values for those versions
    :param: session_wise bool that indicates where the figure is save
    :return: None
    '''

    for ii in range(corr_array.shape[0]):
        for jj in range(pnr_array.shape[0]):
            fig, ax = plt.subplots(1)
            new_row = rows.query('(source_extraction_v == ' + f'{ii *corr_array.shape[0] + jj + 1}' + ')')
            new_row = new_row.iloc[0]

            output = eval(new_row.loc['source_extraction_output'])
            cnm_file_path = output['main']
            cnm = load_CNMF(db.get_file(cnm_file_path))
            C = cnm.estimates.C
            idx_components = cnm.estimates.idx_components
            C[0] += C[0].min()
            for i in range(1, len(C)):
                C[i] += C[i].min() + C[:i].max()
                ax.plot(C[i])
            ax.set_xlabel('t [frames]')
            ax.set_yticks([])
            ax.set_ylabel('activity')
            fig.set_size_inches([10., .3 * len(C)])


            fig_dir = 'data/interim/source_extraction/trial_wise/meta/figures/traces/'
            if session_wise:
                fig_dir = 'data/interim/source_extraction/session_wise/meta/figures/traces/'
            fig_name = fig_dir + db.create_file_name(3,new_row.name) + '_corr_min' + f'{round(corr_array[ii], 1)}' + '_pnr_min' + f'{round(pnr_array[jj], 1)}' + '_.png'
            fig.savefig(fig_name)

    return


def plot_contours_evaluated(row = None, session_wise = False):
    '''
    Plot contours for all cells, selected cells and non selected cells, and saves it in
    figure_path = '/data/interim/component_evaluation/trial_wise/meta/figures/contours/'
    :param row: one analysis state row
    '''
    index = row.name

    corr_min = round(eval(row['source_extraction_parameters'])['min_corr'],1)
    pnr_min = round(eval(row['source_extraction_parameters'])['min_pnr'],1)
    r_min = eval(row['component_evaluation_parameters'])['rval_thr']
    snf_min = eval(row['component_evaluation_parameters'])['min_SNR']

    output_source_extraction = eval(row.loc['source_extraction_output'])
    corr_path = output_source_extraction['meta']['corr']['main']
    cn_filter = np.load(db.get_file(corr_path))

    output_component_evaluation =  eval(row.loc['component_evaluation_output'])
    cnm_file_path = output_component_evaluation['main']
    cnm = load_CNMF(db.get_file(cnm_file_path))
    figure, axes = plt.subplots(1, 3)
    axes[0].imshow(cn_filter)
    axes[1].imshow(cn_filter)
    axes[2].imshow(cn_filter)

    coordinates = cm.utils.visualization.get_contours(cnm.estimates.A, np.shape(cn_filter), 0.2, 'max')
    for c in coordinates:
        v = c['coordinates']
        c['bbox'] = [np.floor(np.nanmin(v[:, 1])), np.ceil(np.nanmax(v[:, 1])),
                     np.floor(np.nanmin(v[:, 0])), np.ceil(np.nanmax(v[:, 0]))]
        axes[0].plot(*v.T, c='w')
    axes[0].set_title('All components')
    axes[0].set_ylabel('Corr=' + f'{corr_min}' + ', PNR = ' + f'{pnr_min}' + ', PCC = ' + f'{r_min}' + ', SNR =' + f'{snf_min}')

    idx = cnm.estimates.idx_components
    coordinates = cm.utils.visualization.get_contours(cnm.estimates.A[:,idx], np.shape(cn_filter), 0.2, 'max')

    for c in coordinates:
        v = c['coordinates']
        c['bbox'] = [np.floor(np.nanmin(v[:, 1])), np.ceil(np.nanmax(v[:, 1])),
                     np.floor(np.nanmin(v[:, 0])), np.ceil(np.nanmax(v[:, 0]))]
        axes[1].plot(*v.T, c='b')
    axes[1].set_title('Accepted components')

    idx_b = cnm.estimates.idx_components_bad
    coordinates_b = cm.utils.visualization.get_contours(cnm.estimates.A[:,idx_b], np.shape(cn_filter), 0.2, 'max')

    for c in coordinates_b:
        v = c['coordinates']
        c['bbox'] = [np.floor(np.nanmin(v[:, 1])), np.ceil(np.nanmax(v[:, 1])),
                     np.floor(np.nanmin(v[:, 0])), np.ceil(np.nanmax(v[:, 0]))]
        axes[2].plot(*v.T, c='r')
    axes[2].set_title('Rejected components')

    figure_path = '/home/sebastian/Documents/Melisa/calcium_imaging_analysis/data/interim/component_evaluation/trial_wise/meta/figures/contours/'
    if session_wise:
        figure_path = '/home/sebastian/Documents/Melisa/calcium_imaging_analysis/data/interim/component_evaluation/session_wise/meta/figures/contours/'
    figure_name = figure_path + db.create_file_name(5,index) + '.png'
    figure.savefig(figure_name)
    return figure

def plot_traces_multiple_evaluated(row = None, session_wise = False):
    '''
    Plots different versions of contour images that change the inicialization parameters for source extraccion.
    The idea is to see the impact of different seed selection in the final source extraction result.
    :param row: one analysis state row
    :return: figure
    '''

    corr_min = round(eval(row['source_extraction_parameters'])['min_corr'],1)
    pnr_min = round(eval(row['source_extraction_parameters'])['min_pnr'],1)
    r_min = eval(row['component_evaluation_parameters'])['rval_thr']
    snf_min = eval(row['component_evaluation_parameters'])['min_SNR']

    output_source_extraction = eval(row.loc['source_extraction_output'])
    corr_path = output_source_extraction['meta']['corr']['main']
    cn_filter = np.load(db.get_file(corr_path))
    cnm_file_path = output_source_extraction['main']
    cnm = load_CNMF(db.get_file(cnm_file_path))
    C = cnm.estimates.C

    output_component_evaluation =  eval(row.loc['component_evaluation_output'])
    cnm_file_path = output_component_evaluation['main']
    cnm_eval = load_CNMF(db.get_file(cnm_file_path))
    idx = cnm_eval.estimates.idx_components
    idx_b = cnm_eval.estimates.idx_components_bad

    fig, ax = plt.subplots(1)
    C[0] += C[0].min()
    for i in range(1, len(C)):
        C[i] += C[i].min() + C[:i].max()
        if i in idx_b:
            color = 'red'
        else:
            color = 'blue'
        ax.plot(C[i],color = color)
    ax.set_xlabel('t [frames]')
    ax.set_yticks([])
    ax.set_ylabel('activity')
    ax.set_title('Corr=' + f'{corr_min}' + ', PNR = ' + f'{pnr_min}' + ', PCC = ' + f'{r_min}' + ', SNR =' + f'{snf_min}')

    fig.set_size_inches([10., .3 * len(C)])

    fig_dir = 'data/interim/component_evaluation/trial_wise/meta/figures/traces/'
    if session_wise:
        fig_dir = 'data/interim/component_evaluation/session_wise/meta/figures/traces/'
    fig_name = fig_dir + db.create_file_name(5,row.name) + '.png'
    fig.savefig(fig_name)

    return

def play_movie(estimates, imgs, q_max=99.75, q_min=2, gain_res=1,
                   magnification=1, include_bck=True,
                   frame_range=slice(None, None, None),
                   bpx=0, thr=0., save_movie=False,
                   movie_name='results_movie.avi'):

    dims = imgs.shape[1:]
    if 'movie' not in str(type(imgs)):
        imgs = cm.movie(imgs)
    Y_rec = estimates.A.dot(estimates.C[:, frame_range])
    Y_rec = Y_rec.reshape(dims + (-1,), order='F')
    Y_rec = Y_rec.transpose([2, 0, 1])

    if estimates.W is not None:
        ssub_B = int(round(np.sqrt(np.prod(dims) / estimates.W.shape[0])))
        B = imgs[frame_range].reshape((-1, np.prod(dims)), order='F').T - \
            estimates.A.dot(estimates.C[:, frame_range])
        if ssub_B == 1:
            B = estimates.b0[:, None] + estimates.W.dot(B - estimates.b0[:, None])
        else:
            B = estimates.b0[:, None] + (np.repeat(np.repeat(estimates.W.dot(
                downscale(B.reshape(dims + (B.shape[-1],), order='F'),
                          (ssub_B, ssub_B, 1)).reshape((-1, B.shape[-1]), order='F') -
                downscale(estimates.b0.reshape(dims, order='F'),
                          (ssub_B, ssub_B)).reshape((-1, 1), order='F'))
                    .reshape(((dims[0] - 1) // ssub_B + 1, (dims[1] - 1) // ssub_B + 1, -1), order='F'),
                    ssub_B, 0), ssub_B, 1)[:dims[0], :dims[1]].reshape((-1, B.shape[-1]), order='F'))
        B = B.reshape(dims + (-1,), order='F').transpose([2, 0, 1])
    elif estimates.b is not None and estimates.f is not None:
        B = estimates.b.dot(estimates.f[:, frame_range])
        if 'matrix' in str(type(B)):
            B = B.toarray()
        B = B.reshape(dims + (-1,), order='F').transpose([2, 0, 1])
    else:
        B = np.zeros_like(Y_rec)
    if bpx > 0:
        B = B[:, bpx:-bpx, bpx:-bpx]
        Y_rec = Y_rec[:, bpx:-bpx, bpx:-bpx]
        imgs = imgs[:, bpx:-bpx, bpx:-bpx]

    Y_res = imgs[frame_range] - Y_rec - B

    mov = cm.concatenate((imgs[frame_range] - (not include_bck) * B, Y_rec,
                            Y_rec + include_bck * B, Y_res * gain_res), axis=2)

    if thr > 0:
        if save_movie:
            import cv2
            #fourcc = cv2.VideoWriter_fourcc('8', 'B', 'P', 'S')
            #fourcc = cv2.VideoWriter_fourcc(*'XVID')
            fourcc = cv2.VideoWriter_fourcc(*'MP4V')
            out = cv2.VideoWriter(movie_name, fourcc, 30.0,
                                  tuple([int(magnification*s) for s in mov.shape[1:][::-1]]))
        contours = []
        for a in estimates.A.T.toarray():
            a = a.reshape(dims, order='F')
            if bpx > 0:
                a = a[bpx:-bpx, bpx:-bpx]
            if magnification != 1:
                a = cv2.resize(a, None, fx=magnification, fy=magnification,
                               interpolation=cv2.INTER_LINEAR)
            ret, thresh = cv2.threshold(a, thr * np.max(a), 1., 0)
            contour, hierarchy = cv2.findContours(
                thresh.astype('uint8'), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contours.append(contour)
            contours.append(list([c + np.array([[a.shape[1], 0]]) for c in contour]))
            contours.append(list([c + np.array([[2 * a.shape[1], 0]]) for c in contour]))

        maxmov = np.nanpercentile(mov[0:10], q_max) if q_max < 100 else np.nanmax(mov)
        minmov = np.nanpercentile(mov[0:10], q_min) if q_min > 0 else np.nanmin(mov)
        for frame in mov:
            if magnification != 1:
                frame = cv2.resize(frame, None, fx=magnification, fy=magnification,
                                   interpolation=cv2.INTER_LINEAR)
            frame = np.clip((frame - minmov) * 255. / (maxmov - minmov), 0, 255)
            frame = np.repeat(frame[..., None], 3, 2)
            for contour in contours:
                cv2.drawContours(frame, contour, -1, (0, 255, 255), 1)
            cv2.imshow('frame', frame.astype('uint8'))
            if save_movie:
                out.write(frame.astype('uint8'))
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break
        if save_movie:
            out.release()
        cv2.destroyAllWindows()
        cv2.destroyAllWindows()
    else:
        mov.play(q_min=q_min, q_max=q_max, magnification=magnification,
                     save_movie=save_movie, movie_name=movie_name)

    return
