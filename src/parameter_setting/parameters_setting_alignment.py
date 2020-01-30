#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Melisa
"""

import os
import sys
import psutil
import pickle
import logging
import datetime
import numpy as np
import pylab as pl
import pandas as pd

# This should be in another file. Let's leave it here for now
sys.path.append('/home/sebastian/Documents/Melisa/calcium_imaging_analysis/src/')
sys.path.remove('/home/sebastian/Documents/calcium_imaging_analysis')


import matplotlib.pyplot as plt
import src.configuration
import caiman as cm
import src.data_base_manipulation as db
import src.analysis.figures as figures
from caiman.source_extraction.cnmf.cnmf import load_CNMF

# Paths
analysis_states_database_path = os.environ['PROJECT_DIR'] + 'references/analysis/calcium_imaging_data_base_server_new.xlsx'
backup_path = 'references/analysis/backup/'

states_df = db.open_analysis_states_database(path = analysis_states_database_path)

mouse_number = 56165
session = 1
init_trial = 1
end_trial = 22
is_rest = None

decoding_version = 1
cropping_version = 2
motion_correction_version = 1
alignment_version = 2
selected_rows = db.select(states_df, 'source_extraction', mouse=mouse_number, session=session,
                          decoding_v=decoding_version,
                          cropping_v=cropping_version,
                          motion_correction_v=motion_correction_version,
                          alignment_v=alignment_version,
                          source_extraction_v=0)

row = selected_rows.iloc[0]
figures.plot_temporal_evolution(row,session_wise = True)