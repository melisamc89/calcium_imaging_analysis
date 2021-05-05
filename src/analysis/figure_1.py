
import numpy as np
import caiman as cm
from caiman.source_extraction.cnmf.cnmf import load_CNMF
import numpy as np

#%%## Figure 2
path1 = '/scratch/melisa/calcium_imaging_analysis/data/interim/alignment/main/'
path2 = '/scratch/melisa/calcium_imaging_analysis/data/interim/source_extraction/session_wise/meta/corr/'


mouse = 56165
session = 1

cropping_v = 1
input_mmap_file_path = path1 + 'mouse_56165_session_1_trial_1_v1.1.1.0_d1_203_d2_231_d3_1_order_C_frames_133956_.mmap'
corr_npy_file_path = path2 +  'mouse_56165_session_1_trial_1_v1.1.1.1_gSig_5.npy'
Yr, dims, T = cm.load_memmap(input_mmap_file_path)
images = Yr.T.reshape((T,) + dims, order='F')
cn_filter, pnr = cm.summary_images.correlation_pnr(images[::1], gSig=5, swap_dim=False)
with open(corr_npy_file_path, 'wb') as f:
    np.save(f,cn_filter)

cropping_v = 2
input_mmap_file_path = path1 + 'mouse_56165_session_1_trial_1_v1.2.1.0_d1_208_d2_233_d3_1_order_C_frames_133956_.mmap'
corr_npy_file_path = path2 + 'mouse_56165_session_1_trial_1_v1.2.1.1_gSig_5.npy'
Yr, dims, T = cm.load_memmap(input_mmap_file_path)
images = Yr.T.reshape((T,) + dims, order='F')
cn_filter, pnr = cm.summary_images.correlation_pnr(images[::1], gSig=5, swap_dim=False)
with open(corr_npy_file_path, 'wb') as f:
    np.save(f,cn_filter)

cropping_v = 3
input_mmap_file_path = path1 + 'mouse_56165_session_1_trial_1_v1.3.1.0_d1_204_d2_231_d3_1_order_C_frames_133956_.mmap'
corr_npy_file_path = path2 +  'mouse_56165_session_1_trial_1_v1.3.1.1_gSig_5.npy'
Yr, dims, T = cm.load_memmap(input_mmap_file_path)
images = Yr.T.reshape((T,) + dims, order='F')
cn_filter, pnr = cm.summary_images.correlation_pnr(images[::1], gSig=5, swap_dim=False)
with open(corr_npy_file_path, 'wb') as f:
    np.save(f,cn_filter)

cropping_v = 4
input_mmap_file_path = path1 + 'mouse_56165_session_1_trial_1_v1.4.1.0_d1_203_d2_231_d3_1_order_C_frames_133956_.mmap'
corr_npy_file_path = path2 +  'mouse_56165_session_1_trial_1_v1.4.1.1_gSig_5.npy'
Yr, dims, T = cm.load_memmap(input_mmap_file_path)
images = Yr.T.reshape((T,) + dims, order='F')
cn_filter, pnr = cm.summary_images.correlation_pnr(images[::1], gSig=5, swap_dim=False)
with open(corr_npy_file_path, 'wb') as f:
    np.save(f,cn_filter)
