import os

#%% ENVIRONMENT VARIABLES
os.environ['PROJECT_DIR_LOCAL'] = '/home/sebastian/Documents/Melisa/calcium_imaging_analysis/'
os.environ['PROJECT_DIR_SERVER'] = '/scratch/mmaidana/calcium_imaging_analysis/'
os.environ['CAIMAN_DIR_LOCAL'] = '/home/sebastian/CaImAn/'
os.environ['CAIMAN_DIR_SERVER'] ='/scratch/mamaidana/CaImAn/'
os.environ['CAIMAN_ENV_SERVER'] = '/scratch/mmaidana/anaconda3/envs/caiman/bin/python'

os.environ['LOCAL_USER'] = 'sebastian'
os.environ['SERVER_USER'] = 'mmaidana'
os.environ['SERVER_HOSTNAME'] = 'cn76'
os.environ['ANALYST'] = 'Meli'

#%% PROCESSING
os.environ['LOCAL'] = str((os.getlogin() == os.environ['LOCAL_USER']))
os.environ['SERVER'] = str(not(eval(os.environ['LOCAL'])))
os.environ['PROJECT_DIR'] = os.environ['PROJECT_DIR_LOCAL'] if eval(os.environ['LOCAL']) else os.environ['PROJECT_DIR_SERVER']
os.environ['CAIMAN_DIR'] = os.environ['CAIMAN_DIR_LOCAL'] if eval(os.environ['LOCAL']) else os.environ['CAIMAN_DIR_SERVER']