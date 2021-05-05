import os

#%% ENVIRONMENT VARIABLES
os.environ['PROJECT_DIR_LOCAL'] = '/home/sebastian/Documents/Melisa/calcium_imaging_analysis/'
os.environ['PROJECT_DIR_SERVER'] = '/home/mmaidana/src/calcium_imaging_analysis/'

os.environ['DATA_DIR_LOCAL'] = '/mnt/Data01/data/calcium_imaging_analysis/'
os.environ['DATA_DIR_SERVER'] ='/ceph/imaging1/melisa/calcium_imaging_analysis/'

os.environ['CAIMAN_ENV_SERVER'] = '/memdym/melisa/caiman/bin/python'

ana_3 = "~/anaconda3"
ana_2 = "~/anaconda2"
inscopix_env = 'inscopix_reader'
os.environ['INSCOPIX_READER_LOCAL'] = os.path.join(ana_2, 'envs', inscopix_env, 'bin/python')
os.environ['INSCOPIX_READER_SERVER'] = "/memdyn/melisa/envs/inscopix_reader/bin/python"

os.environ['DECODER_LOCAL'] =  "~/Documents/inscopix_reader_linux/python/downsampler.py"
os.environ['DECODER_SERVER'] = "/memdyn/melisa/src/inscopix_reader_linux/python/downsampler.py"

os.environ['LOCAL_USER'] = 'sebastian'
os.environ['SERVER_USER'] = 'mmaidana'
os.environ['SERVER_HOSTNAME'] = 'cn43'
os.environ['ANALYST'] = 'Meli'

#%% PROCESSING
os.environ['LOCAL'] = str((os.getlogin() == os.environ['LOCAL_USER']))
os.environ['SERVER'] = str(not(eval(os.environ['LOCAL'])))

os.environ['PROJECT_DIR'] = os.environ['PROJECT_DIR_LOCAL'] if eval(os.environ['LOCAL']) else os.environ['PROJECT_DIR_SERVER']
os.environ['DATA_DIR'] = os.environ['DATA_DIR_LOCAL'] if eval(os.environ['LOCAL']) else os.environ['DATA_DIR_SERVER']
os.environ['INSCOPIX_READER']= os.environ['INSCOPIX_READER_LOCAL'] if eval(os.environ['LOCAL']) else os.environ['INSCOPIX_READER_SERVER']
os.environ['DECODER']= os.environ['DECODER_LOCAL'] if eval(os.environ['LOCAL']) else os.environ['DECODER_SERVER']
