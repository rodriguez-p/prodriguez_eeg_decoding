import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import sem
from stats_functions import wilcoxon_fdr_tfdecod
from utils_functions import load_scores_tf
from mne.time_frequency import AverageTFR
from mne import Epochs, create_info

#%%
FIGURES_DIR = 'C:/Users/Profa. Ana Chica/Desktop/prodriguez_eeg_decoding/figures'
PKL_PATH = 'C:/Users/Profa. Ana Chica/Desktop/prodriguez_eeg_decoding/data'
DECOD_TFR = 'C:/Users/Profa. Ana Chica/Desktop/prodriguez_eeg_decoding/data/decoding_time-frequency'

#%% specify freqs
sfreq=256
freqs = np.logspace(*np.log10([1, 40]), num=25)
times= np.load("{}/{}".format(PKL_PATH, 'times.npy'), allow_pickle=True) 

#%% target presence 
presence_tf = load_scores_tf(DECOD_TFR, 'decoding_time-frequency_presence_lsvm', average = False) 

presence_tf_avg = np.mean(presence_tf, axis = 0)

# Wilcoxon signed-rank test with FDR correction
x = presence_tf
reject, p_corrected, presence_sign_points = wilcoxon_fdr_tfdecod(x)

# plot results with contour for significant time-frequency bins
title = "Decoding target presence on time-frequency data"
av_tfr = AverageTFR(create_info(['freq'], sfreq), presence_tf_avg[np.newaxis, :], freqs = freqs[0:], nave=0, times=times)
av_tfr.plot('all', title=title, cmap='Blues', vmin = 0.5, vmax = np.abs(presence_tf_avg).max(), mask = presence_sign_points, mask_style = 'contour', yscale='linear')
plt.gcf().set_size_inches(15, 10)
plt.savefig(f"{FIGURES_DIR}/{title}")
plt.close()
 
#%% subject awareness
awareness_tf = load_scores_tf(DECOD_TFR, 'decoding_time-frequency_awareness_lsvm', average = False) 

awareness_tf_avg = np.mean(awareness_tf, axis = 0)

# Wilcoxon signed-rank test with FDR correction
x = awareness_tf
reject, p_corrected, awareness_sign_points = wilcoxon_fdr_tfdecod(x)

# plot results with contour for significant time-frequency bins
title = "Decoding subject awareness on time-frequency data"
av_tfr = AverageTFR(create_info(['freq'], sfreq), awareness_tf_avg[np.newaxis, :], freqs = freqs[0:], nave=0, times=times)
av_tfr.plot('all', title=title, cmap='Reds', vmin = 0.5, vmax = np.abs(awareness_tf_avg).max(), mask = awareness_sign_points, mask_style = 'contour', yscale='linear')
plt.gcf().set_size_inches(15, 10)
plt.savefig(f"{FIGURES_DIR}/{title}")
plt.close()

#%% gabor tilt
tilt_tf = load_scores_tf(DECOD_TFR, 'decoding_time-frequency_tilt_lsvm', average = False) 

tilt_tf_avg = np.mean(tilt_tf, axis = 0)

# Wilcoxon signed-rank test with FDR correction
x = tilt_tf
reject, p_corrected, tilt_sign_points = wilcoxon_fdr_tfdecod(x)

# plot results with contour for significant time-frequency bins
title = "Decoding Gabor tilt on time-frequency data"
av_tfr = AverageTFR(create_info(['freq'], sfreq), tilt_tf_avg[np.newaxis, :], freqs = freqs[0:], nave=0, times=times)
av_tfr.plot('all', title=title, cmap='Greens', vmin = 0.5, vmax = np.abs(tilt_tf_avg).max(), mask = tilt_sign_points, mask_style = 'contour', yscale='linear')
plt.gcf().set_size_inches(15, 10)
plt.savefig(f"{FIGURES_DIR}/{title}")
plt.close()