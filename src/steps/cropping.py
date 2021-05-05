# -*- coding: utf-8 -*-
"""
@author: Sebastian,Casper,Melisa
"""



import os
import datetime
import ast
import caiman as cm
import logging
import pylab as pl

import src.data_base_manipulation as db
import src.server as connect

def run_cropper(row, parameters):
    '''
    This function takes in a decoded analysis state and crops it according to 
    specified cropping points.
    
    Args:
        index: tuple
            The index of the analysis state to be cropped. 
        row: pd.DataFrame object
            The row corresponding to the analysis state to be cropped. 
            
    Returns
        row: pd.DataFrame object
            The row corresponding to the cropped analysis state. 
    '''

    row_local = row.copy()
    index=row_local.name
    # Get the input tif file path 
    input_tif_file_path = eval(row_local.loc['decoding_output'])['main']
    if index[4]==2:
        input_tif_file_path = eval(row_local.loc['decoding_output'])['equalizing_output']['main']
    if not os.path.isfile(input_tif_file_path):
        db.get_expected_file_path('decoding','main/',index, '.tif')
    
    # Determine output .tif file path
    step_index = 1
    row_local.loc['cropping_parameters'] = str(parameters)
    row_local = db.set_version_analysis('cropping',row_local)
    index = row_local.name
    file_name =  db.create_file_name(step_index, index)
    output_tif_file_path = os.environ['DATA_DIR'] + f"data/interim/cropping/main/{file_name}.tif"


    # Create a dictionary with the output
    output = {
            'main' : output_tif_file_path,
            'meta' : {
                    'analysis': {
                            'analyst': os.environ['ANALYST'],
                            'date':  datetime.datetime.today().strftime("%m-%d-%Y"),
                            'time': datetime.datetime.today().strftime("%H:%M:%S"),
                            }
                    }
            }

    # Spatial copping
    logging.info(f'{index} Loading movie')
    m = cm.load(input_tif_file_path)
    logging.info(f'{index} Loaded movie')
    [x_,_x,y_,_y] = parameters['cropping_points_spatial']
    if parameters['crop_spatial']:
        logging.info(f'{index} Performing spatial cropping')
        m = m[:,x_:_x,y_:_y]
        logging.info(f'{index} Spatial cropping finished')
    else:
        logging.info(f'{index} No spatial cropping')

    # Temporal cropping
    #if parameters['crop_temporal']:
        # m, timeline = do_temporal_cropping(m, parameters['cropping_points_temporal'])
        # The option below is to get a timeline which indicates on which
        # frames clips are cut out and how long those clips were.
        # I eventually decided this is not neccesary. The temporal cropping points are enough
        # to reconstruct this and are more easily saved (namely in the
        # master file list under 'cropping_parameters')

#        timeline_pkl_file_path = f'data/interim/cropping/meta/timeline/{file_name}.pkl'
#        output['meta']['timeline'] = timeline_pkl_file_path
#        with open(timeline_pkl_file_path,'wb') as f:
#            pickle.dump(timeline, f)

    # Save the movie
    m.save(output_tif_file_path)
    # Write necessary variables to the trial index and row_local
    row_local.loc['cropping_parameters'] = str(parameters)
    row_local.loc['cropping_output'] = str(output)

    return row_local

def upload_to_server_cropped_movie(row):
    '''
    This function copies the cropped file from the local machine to the server

    Input -> row array (change to dictionary) containing file address in ['cropping_output']

    Outout -> None
    '''
    index = row.name
    cropping_output_dict = ast.literal_eval(row['cropping_output'])
    output_tif_file_path = cropping_output_dict['main']
    
    print('Local env')
    print(eval(os.environ['LOCAL']))
    
    if eval(os.environ['LOCAL']):
        logging.info(f'{index} Uploading file to server')
        ssh =  connect.get_SSH_connection()
        sftp = ssh.open_sftp()
        print(os.environ['PROJECT_DIR_LOCAL'] + output_tif_file_path, os.environ['PROJECT_DIR_SERVER'] + output_tif_file_path)
        sftp.put(os.environ['PROJECT_DIR_LOCAL'] + output_tif_file_path, os.environ['PROJECT_DIR_SERVER'] + output_tif_file_path)
        
        ## make a chech about whether the file was copy to the server!!!!
        
#        if parameters['crop_temporal']:
#            sftp.put(os.environ['PROJECT_DIR_LOCAL'] + timeline_pkl_file_path, os.environ['PROJECT_DIR_SERVER'] + timeline_pkl_file_path)
       
        sftp.close()
        ssh.close()
        logging.info(f'{index} Uploading finished')
    # Remove the original movie
    #Add an if to only removed is the copy was succesful
    #os.remove(output_tif_file_path)
    
    return 


def cropping_interval():
    '''
    This function ask the user for cropping paramenters
    :param None:
    :return: dictionary with a new assignment to cropping_paramenters
    '''
    x1 = int(input("Limit X1 : "))
    x2 = int(input("Limit X2 : "))
    y1 = int(input("Limit Y1 : "))
    y2 = int(input("Limit Y2 : "))
    parameters_cropping = {'crop_spatial': True, 'cropping_points_spatial': [y1, y2, x1, x2], 'segmentation': False,
                           'crop_temporal': False, 'cropping_points_temporal': []}
    #print(parameters_cropping)
    return parameters_cropping

<<<<<<< HEAD
=======

>>>>>>> f40749622807a6c7b503bad95384622204adccd9
def cropping_segmentation(parameters_cropping):
    '''
    This function takes the cropping interval and segment the image in 4 different regions.
    The pipeline should lated run in all the different regions.
    Returns:
    '''
    cropping_parameters_list = []
    [y1, y2, x1, x2] = parameters_cropping['cropping_points_spatial']
    if parameters_cropping['segmentation'] == True:
        y1_new1 = y1
        y2_new1 = round((y2 + y1 ) / 2) - 15
        y1_new2 = round((y2 + y1) /2 )+ 15
        y2_new2 = y2
        x1_new1 = x1
        x2_new1 = round((x2 + x1 ) / 2) - 15
        x1_new2 = round((x2 + x1) /2) + 15
        x2_new2 = x2
        cropping_parameters_list.append({'crop_spatial': True, 'cropping_points_spatial': [y1_new1, y2_new1, x1_new1, x2_new1], 'segmentation': False,
                               'crop_temporal': False, 'cropping_points_temporal': []})
        cropping_parameters_list.append({'crop_spatial': True, 'cropping_points_spatial': [y1_new1, y2_new1, x1_new2, x2_new2], 'segmentation': False,
                               'crop_temporal': False, 'cropping_points_temporal': []})
        cropping_parameters_list.append({'crop_spatial': True, 'cropping_points_spatial': [y1_new2, y2_new2, x1_new1, x2_new1], 'segmentation': False,
                               'crop_temporal': False, 'cropping_points_temporal': []})
        cropping_parameters_list.append({'crop_spatial': True, 'cropping_points_spatial': [y1_new2, y2_new2, x1_new2, x2_new2], 'segmentation': False,
                               'crop_temporal': False, 'cropping_points_temporal': []})
    else:
        cropping_parameters_list.append(parameters_cropping)

    return cropping_parameters_list

<<<<<<< HEAD
=======

>>>>>>> f40749622807a6c7b503bad95384622204adccd9
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