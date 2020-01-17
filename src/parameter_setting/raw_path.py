#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import pandas as pd
# This should be in another file. Let's leave it here for now
sys.path.append('/home/sebastian/Documents/Melisa/calcium_imaging_analysis/src/')
sys.path.remove('/home/sebastian/Documents/calcium_imaging_analysis')

import src.configuration
import src.data_base_manipulation as db
import src.paths as paths

# Paths
analysis_states_database_path = os.environ['PROJECT_DIR'] + 'references/analysis/analysis_states_database.xlsx'
analysis_states_database_path_server = os.environ['PROJECT_DIR']  + 'references/analysis/calcium_imaging_data_base_server.xlsx'
analysis_states_database_path_server_new =  'references/analysis/calcium_imaging_data_base_server_new.xlsx'

backup_path = os.environ['PROJECT_DIR'] +  'references/analysis/backup/'
#parameters_path = 'references/analysis/parameters_database.xlsx'

## Open thw data base with all data
local_db_path = db.open_analysis_states_database()
new_data_structure = ['mouse','session','trial', 'is_rest', 'experimenter', 'date', 'time', 'raw_data', 'xml_data'  ]
server_db_path = pd.read_excel(analysis_states_database_path_server,dtype = {'date' : 'str', 'time' : 'str'}).set_index(new_data_structure)

selected_rows = db.select(local_db_path,'decoding',56166, 2)
selected_rows2 = server_db_path.query('mouse == 56166')
selected_rows2 = selected_rows2.query('session == 2')

for i in range(len(selected_rows2)):
    # Create a dictionary with the output
    output = {'main': selected_rows2.iloc[i].name[7] ,
             'meta':{'xml' :[selected_rows2.iloc[i].name[8]]}}
    auxiliar_variable = selected_rows.iloc[i].copy()
    auxiliar_variable['raw_output'] = output
    selected_rows.iloc[i] = auxiliar_variable

selected_rows.reset_index().sort_values(by=paths.multi_index_structure)[paths.columns].to_excel(analysis_states_database_path_server_new , index=False)




    #%% create the list to manualy copy

selected_rows = db.select(local_db_path,'decoding',56165, 1)

file_list = []
for i in range(len(selected_rows)):
    input = eval(selected_rows.iloc[i]['raw_output'])['main']
    for j in range(len(input)):
        file_list.append(input[j])

list = str(file_list)
for i in range(len(list)):
    if list[i] != ',':
        list[i] = '  '