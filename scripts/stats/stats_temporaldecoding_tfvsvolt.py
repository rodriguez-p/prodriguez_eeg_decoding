import numpy as np
from scipy.stats import sem
from plotting import plot_stats_tfvsvoltage
from stats import wilcoxon_fdr
from utils import load_scores

#%%
FIGURES_DIR = 'C:/Users/Profa. Ana Chica/Desktop/prodriguez_eeg_decoding/figures'
PKL_PATH = 'C:/Users/Profa. Ana Chica/Desktop/prodriguez_eeg_decoding/data'
DECOD_TFR = 'C:/Users/Profa. Ana Chica/Desktop/prodriguez_eeg_decoding/data/decoding_time-frequency'
DECOD_TIME = 'C:/Users/Profa. Ana Chica/Desktop/prodriguez_eeg_decoding/data/decoding_time'

#%%
times= np.load("{}/{}".format(PKL_PATH, 'times.npy'), allow_pickle=True) 

#%% target presence 

presence_volt = load_scores(DECOD_TIME, 'temporal_decoding_presence_lsvm_experimental')
presence_tf = load_scores(DECOD_TFR, 'temporal_decoding_time-frequency_presence_lsvm')

presence_tf_se = np.mean(sem(presence_tf), axis = 0)
presence_tf_avg = np.mean(presence_tf, axis = 0)

presence_volt_se = np.mean(sem(presence_volt), axis = 0)
presence_volt_avg = np.mean(presence_volt, axis = 0)

# Wilcoxon signed-rank test with FDR correction

x = presence_tf
y = presence_volt

reject, p_corrected, presence_sign_points = wilcoxon_fdr(x, y, times)

# plot data with dots for significant comparisons

title='Target presence - TF power vs voltage'
save_figure = "{}/{}".format(FIGURES_DIR, title)

plot_stats_tfvsvoltage(presence_tf_avg, presence_volt_avg, presence_tf_se, presence_volt_se, 
                        'navy', 'royalblue', 'lavender', 'lightsteelblue', 
                        times, presence_sign_points, 0.46, title, save_figure)

#%% subject awareness 

awareness_volt = load_scores(DECOD_TIME, 'temporal_decoding_awareness_lsvm_experimental')
awareness_tf = load_scores(DECOD_TFR, 'temporal_decoding_time-frequency_awareness_lsvm')

awareness_tf_se = np.mean(sem(awareness_tf), axis = 0)
awareness_tf_avg = np.mean(awareness_tf, axis = 0)

awareness_volt_se = np.mean(sem(awareness_volt), axis = 0)
awareness_volt_avg = np.mean(awareness_volt, axis = 0)

# Wilcoxon signed-rank test with FDR correction

x = awareness_tf
y = awareness_volt

reject, p_corrected, awareness_sign_points = wilcoxon_fdr(x, y, times)

# plot data with dots for significant comparisons

title='Subject awareness - TF power vs voltage'
save_figure = "{}/{}".format(FIGURES_DIR, title)

plot_stats_tfvsvoltage(awareness_tf_avg, awareness_volt_avg, awareness_tf_se, awareness_volt_se, 
                        'coral', 'red', 'bisque', 'lightcoral', 
                        times, awareness_sign_points, 0.46, title, save_figure)

#%% gabor tilt 

tilt_volt = load_scores(DECOD_TIME, 'temporal_decoding_tilt_lsvm_experimental')
tilt_tf = load_scores(DECOD_TFR, 'temporal_decoding_time-frequency_tilt_lsvm')

tilt_tf_se = np.mean(sem(tilt_tf), axis = 0)
tilt_tf_avg = np.mean(tilt_tf, axis = 0)

tilt_volt_se = np.mean(sem(tilt_volt), axis = 0)
tilt_volt_avg = np.mean(tilt_volt, axis = 0)

# Wilcoxon signed-rank test with FDR correction

x = tilt_tf
y = tilt_volt

reject, p_corrected, tilt_sign_points = wilcoxon_fdr(x, y, times)

# plot data with dots for significant comparisons

title='Gabor tilt - TF power vs voltage'
save_figure = "{}/{}".format(FIGURES_DIR, title)

plot_stats_tfvsvoltage(tilt_tf_avg, tilt_volt_avg, tilt_tf_se, tilt_volt_se, 
                        'green', 'olivedrab', 'darkseagreen', 'lightgreen', 
                        times, tilt_sign_points, 0.46, title, save_figure)
