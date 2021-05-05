# -*- coding: utf-8 -*-
"""
@author: Sebastian,Casper,Melisa

"""


import os
import logging
import subprocess 
import datetime

import src.data_base_manipulation as db
#import src.configuration

def run_decoder(row):
    '''
    This is the function for the decoding step. In the decoding step files are converted from .raw files to .tif files.

    This function requires a particular environment to work. Ask Francesco, Ronny, Morgane or Melisa for further
    information.

    :param row: pd.DataFrame object
            The row corresponding to the analysis state to be decoded.
    :return: row: pd.DataFrame object
            The row corresponding to the decoded analysis state.
    '''

    index = row.name
    row_local = row.copy()
    row_local = db.set_version_analysis('decoding', row_local)
    raw_output = eval(row_local['raw_output'])
    input_raw_file_path = raw_output['main']

    # Get the path WITHOUT -001 or -002 in the path and .raw as extension.
    # This does seem to work. All files are converted. 
<<<<<<< HEAD
=======
    #input_raw_file_path = ''
    #for path in input_raw_file_paths:
    #    if path[-8:-5] != '-00':
    #        input_raw_file_path = path
>>>>>>> f40749622807a6c7b503bad95384622204adccd9

    # Determine output .tif file path
    step_index = 0 # decoding is the first step
    file_name = db.create_file_name(step_index,index)
    output_tif_file_path = os.environ['DATA_DIR'] + f"data/interim/decoding/main/{file_name}.tif"

    # Decoder paths
    py_inscopix = os.environ['INSCOPIX_READER']
    decoder = os.environ['DECODER']
    
    # Create a dictionary with the parameters
    output = {
            'main' : output_tif_file_path,
            'meta' : { 
                    'analysis': {
                        'analyst': os.environ['ANALYST'],
                        'date': datetime.datetime.today().strftime("%m-%d-%Y"),
                        'time': datetime.datetime.today().strftime("%H:%M:%S")
                        }
                    }
            }
            
    # Decoding
    logging.info(f'{index} Performing decoding on raw file {input_raw_file_path}')

    # Convert the output tif file path to the full path such that the downsampler.py script can use them. 
    output_tif_file_path_full = os.path.join(os.environ['PROJECT_DIR'], output_tif_file_path)
    
    # Make a command usable by the decoder script (downsampler.py, see the script for more info)
    if len(raw_output['meta']['xml']) != 0:
        input_xml_file_path = raw_output['meta']['xml']
        cmd = ' '.join([py_inscopix, decoder, '"' + input_raw_file_path + '"', output_tif_file_path_full, '"' + input_xml_file_path + '"']) 
    else:
        cmd = ' '.join([py_inscopix, decoder, '"' + input_raw_file_path + '"', output_tif_file_path_full]) 
    
    # Run the command
    subprocess.check_output(cmd, shell = True)     
    logging.info(f'{index} Decoding finished')
    
    # Write necessary variables to the trial index and row
    row_local.loc['decoding_output'] = str(output)

    return row_local

<<<<<<< HEAD
=======

def fake_decoder(row):
    '''
    This is the function for the decoding step. In the decoding step files are converted from .raw files to .tif files.

    This function requires a particular environment to work. Ask Francesco, Ronny, Morgane or Melisa for further
    information.

    :param row: pd.DataFrame object
            The row corresponding to the analysis state to be decoded.
    :return: row: pd.DataFrame object
            The row corresponding to the decoded analysis state.
    '''

    index = row.name
    row_local = row.copy()
    row_local = db.set_version_analysis('decoding', row_local)
    raw_output = eval(row_local['raw_output'])
    input_raw_file_path = raw_output['main']

    # Get the path WITHOUT -001 or -002 in the path and .raw as extension.
    # This does seem to work. All files are converted.
    #input_raw_file_path = ''
    #for path in input_raw_file_paths:
    #    if path[-8:-5] != '-00':
    #        input_raw_file_path = path

    # Determine output .tif file path
    step_index = 0  # decoding is the first step
    file_name = db.create_file_name(step_index, index)
    output_tif_file_path = os.environ['DATA_DIR'] + f"data/interim/decoding/main/{file_name}.tif"

    # Create a dictionary with the parameters
    output = {
        'main': output_tif_file_path,
        'meta': {
            'analysis': {
                'analyst': os.environ['ANALYST'],
                'date': datetime.datetime.today().strftime("%m-%d-%Y"),
                'time': datetime.datetime.today().strftime("%H:%M:%S")
            }
        }
    }


    # Convert the output tif file path to the full path such that the downsampler.py script can use them.
    output_tif_file_path_full = os.path.join(os.environ['PROJECT_DIR_SERVER'], output_tif_file_path)

    # Write necessary variables to the trial index and row
    row_local.loc['decoding_output'] = str(output)

    return row_local
>>>>>>> f40749622807a6c7b503bad95384622204adccd9
