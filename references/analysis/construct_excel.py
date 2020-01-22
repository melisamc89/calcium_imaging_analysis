#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 10:52:40 2019

@author: Melisa

For creating a new excel shit based on the experimental excel organization, 
but now including the path to the raw data. 

"""

import numpy as np
import pandas as pd

# path to computer session and server

path_configuration = '/home/morgane/'
path_configuration= '/home/sebastian/'

path_server = 'trifle/homes/evelien/Calcium imaging/'
path_local = 'Documents/Melisa/calcium_imaging_analysis/'
path_references = 'references/analysis/'

## upload experimenters data base
experimenter_data_base = pd.read_excel(path_configuration + path_local + path_references + 'calcium_analysis_filelist.xlsx')

#take only the relevant information for the new excel shit
data_structure = ['mouse','session','trial', 'is_rest',  'date', 'timestamp' ]
experimenter_data_base =experimenter_data_base[data_structure]

#redefine the variable trial to be always an integer
for i in range(len(experimenter_data_base)):
    if type(experimenter_data_base.trial[i]) == str:
        experimenter_data_base.trial[i] = int(experimenter_data_base.trial[i][:-2])
    
##separate information in a dictionary (this is not relevant, at some point it was, but now it is not)
experimental_data = {'experimenter': None, 
        'date': experimenter_data_base.date.fillna(111111).astype(int).astype(str), 
        'time': experimenter_data_base.timestamp.fillna(111111).astype(int).astype(str)}
data = { 'mouse' : np.array(experimenter_data_base.mouse), 'session' : np.array(experimenter_data_base.session),
        'trial' : np.array(experimenter_data_base.trial), 'is_rest': np.array(experimenter_data_base.is_rest)}

experiment = pd.DataFrame(data=experimental_data)
mouse = pd.DataFrame(data=data)
result = pd.concat([mouse, experiment], axis=1)

## separate the data in batchs that have different path to raw data organization
batch1 = result.query('mouse <= 32366')
mouse_path = '32363-32366/'
final_path = []
path_list = []
for i in range(len(batch1)):
    mouse = batch1.mouse[batch1.index[i]]
    session = batch1.session[batch1.index[i]]
    trial = batch1.trial[batch1.index[i]]
    is_rest = batch1.is_rest[batch1.index[i]]
    year = batch1.date[batch1.index[i]][0:4]
    month = batch1.date[batch1.index[i]][4:6]
    day = batch1.date[batch1.index[i]][6:8]
    if trial == 1:
        end_date = int(day)+4
        session_path = 'Session '+ f'{session}'+ ' ' + day + '.' + month + '-' + f'{end_date}' + '.' + f'{month}' +'/'
    trial_path = day + '.' + month + '.' + year + '/'
    trial_path = trial_path + f'{mouse}' + '/'
    file_name = 'recording_' + batch1.date[batch1.index[i]] +'_' +batch1.time[batch1.index[i]]
    final_path.append( path_server + mouse_path + session_path + trial_path + file_name)
    path_list.append(path_configuration)


batch2 = result.query('mouse >= 39757 and mouse <= 40187')
mouse_path = '39757-40187/'
for i in range(len(batch2)):
    mouse = batch2.mouse[batch2.index[i]]
    session = batch2.session[batch2.index[i]]
    trial = batch2.trial[batch2.index[i]]
    is_rest = batch2.is_rest[batch2.index[i]]
    year = batch2.date[batch2.index[i]][0:4]
    month = batch2.date[batch2.index[i]][4:6]
    day = batch2.date[batch2.index[i]][6:8]
    if trial == 1:
        end_date = int(day)+4
        session_path = 'Session '+ f'{session}'+ ' ' + day + '.' + month + '.'+year+'-' + f'{end_date}' + '.' + f'{month}' + '.'+year+'/'
    trial_path = day + '.' + month + '.' + year + '/'
    trial_path = f'{mouse}' + '/' + trial_path 
    file_name = 'recording_' + batch2.date[batch2.index[i]] +'_' +batch2.time[batch2.index[i]]
    final_path.append( path_server + mouse_path + session_path + trial_path + file_name)
    path_list.append(path_configuration)


batch3 = result.query('mouse >= 56165 and mouse <= 56166')
mouse_path = '56165-56166/'
for i in range(len(batch3)):
    mouse = batch3.mouse[batch3.index[i]]
    session = batch3.session[batch3.index[i]]
    trial = batch3.trial[batch3.index[i]]
    is_rest = batch3.is_rest[batch3.index[i]]
    year = batch3.date[batch3.index[i]][0:4]
    month = batch3.date[batch3.index[i]][4:6]
    day = batch3.date[batch3.index[i]][6:8]
    date = batch3.date[batch3.index[i]]
    if trial == 1:
        date2 = int(date)+4
        session_path = date + '-' + f'{date2}' + '/'
    trial_path = year + '.' + month + '.' + day + '/'
    file_name = 'recording_' + batch2.date[batch2.index[i]] +'_' +batch2.time[batch2.index[i]]
    final_path.append(path_server + mouse_path + f'{mouse}'+ '/'+  session_path + trial_path + file_name)
    path_list.append(path_configuration)

## create a new data base
path_dict = {'origin_path' : path_list , 'raw_data' : final_path}
path_pandas = pd.DataFrame(path_dict)
batchs = pd.concat([batch1,batch2,batch3],axis = 0, ignore_index = True)

data_base = pd.concat([batchs, path_pandas], axis=1)
new_data_structure = ['mouse','session','trial', 'is_rest', 'experimenter', 'date', 'time', 'origin_path', 'raw_data'  ]
db = data_base[new_data_structure]

##save the new data path
db.reset_index().sort_values(by = new_data_structure).to_excel(path_configuration + path_local + path_references + 'calcium_imaging_data_base.xlsx', index = False)



