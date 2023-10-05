import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import sem
from stats import wilcoxon_fdr
from utils import load_scores

#%%
FIGURES_DIR = 'C:/Users/Profa. Ana Chica/Desktop/prodriguez_eeg_decoding/figures'
PKL_PATH = 'C:/Users/Profa. Ana Chica/Desktop/prodriguez_eeg_decoding/data'
DECOD_TFR = 'C:/Users/Profa. Ana Chica/Desktop/prodriguez_eeg_decoding/data/decoding_time-frequency'

#%%
times= np.load("{}/{}".format(PKL_PATH, 'times.npy'), allow_pickle=True) 

#%% target presence 

presence_tf = load_scores(DECOD_TFR, 'temporal_decoding_time-frequency_presence_lsvm')

presence_tf_se = np.mean(sem(presence_tf), axis = 0)
presence_tf_avg = np.mean(presence_tf, axis = 0)

# Wilcoxon signed-rank test with FDR correction
x = presence_tf
y = np.full((30, 1024), 0.5)

reject, p_corrected, presence_sign_points = wilcoxon_fdr(x, y, times)

#%% subject awareness

awareness_tf = load_scores(DECOD_TFR, 'temporal_decoding_time-frequency_awareness_lsvm')

awareness_tf_se = np.mean(sem(awareness_tf), axis = 0)
awareness_tf_avg = np.mean(awareness_tf, axis = 0)

# Wilcoxon signed-rank test with FDR correction
x = awareness_tf
y = np.full((30, 1024), 0.5)

# Wilcoxon signed-rank test with FDR correction
reject, p_corrected, awareness_sign_points = wilcoxon_fdr(x, y, times)

#%% gabor tilt

tilt_tf = load_scores(DECOD_TFR, 'temporal_decoding_time-frequency_tilt_lsvm')

tilt_tf_se = np.mean(sem(tilt_tf), axis = 0)
tilt_tf_avg = np.mean(tilt_tf, axis = 0)

# Wilcoxon signed-rank test with FDR correction
x = tilt_tf
y = np.full((30, 1024), 0.5)

reject, p_corrected, tilt_sign_points = wilcoxon_fdr(x, y, times)

#%% plot data with dots for significant comparisons

title='Plot stats TF power'
save_figure = "{}/{}".format(FIGURES_DIR, title)

data_to_plot = {
    'Target presence': (presence_tf_avg, presence_tf_se, 'navy', 'lavender'),
    'Subject awareness': (awareness_tf_avg, awareness_tf_se, 'coral', 'bisque'),
    'Gabor tilt': (tilt_tf_avg, tilt_tf_se, 'green', 'darkseagreen')
}

fig, ax = plt.subplots()

for label, (avg_data, se_data, color, color_fill) in data_to_plot.items():
    ax.plot(times, avg_data, color=color, label=label)
    ax.fill_between(times, (avg_data + se_data), (avg_data - se_data), color=color_fill, alpha=0.5)

ax.axhline(0.5, color='k', linestyle='--', label='Chance level', linewidth=0.5)
ax.set_xlabel('Times')
ax.set_ylabel('AUC')  # Area Under the Curve
ax.legend()
ax.axvline(0.0, color='k', linestyle='-', linewidth=0.5)

for i in presence_sign_points:
    plot_points = [0.46]*1024
    ax.scatter(times[i], plot_points[i], color='navy', alpha=0.2, s = 20)
    
for j in awareness_sign_points:
    plot_points = [0.455]*1024
    ax.scatter(times[j], plot_points[j], color='coral', alpha=0.2, s = 20)
    
for k in tilt_sign_points:
    plot_points = [0.47]*1024
    ax.scatter(times[k], plot_points[k], color='coral', alpha=0.2, s = 0.1)

ax.spines[['right', 'top']].set_visible(False)
ax.set_title(title)
fig.set_size_inches(15, 10)
plt.savefig(save_figure)
# plt.close()
