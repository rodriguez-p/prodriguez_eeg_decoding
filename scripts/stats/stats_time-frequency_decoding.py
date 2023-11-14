import matplotlib.pyplot as plt
import numpy as np
from stats_functions import wilcoxon_fdr_tfdecod
from utils_functions import load_scores_tf
from plotting_functions import plot_stats_timefrequency_decoding

#%%
FIGURES_DIR = 'F:/EEG_Decoding_Offline/prodriguez_eeg_decoding/figures'
DATA_PATH = 'F:/EEG_Decoding_Offline/prodriguez_eeg_decoding/data'
DECOD_TFR = f'{DATA_PATH}/decoding_time-frequency'

#%% 
times= np.load("{}/{}".format(DATA_PATH, 'times.npy'), allow_pickle=True) 

#%% target presence 
presence_tf = load_scores_tf(DECOD_TFR, 'decoding_time-frequency_presence_lsvm', average = False) 

presence_tf_avg = np.mean(presence_tf, axis = 0)

# Wilcoxon signed-rank test with FDR correction

stats_file = 'stats_time-frequency_decoding_presence'

reject, p_corrected, presence_sign_points = wilcoxon_fdr_tfdecod(presence_tf, times, stats_file)

# plot results with contour for significant time-frequency bins
title = "Decoding target presence on time-frequency data"
color = 'Blues'
save_figure = f"{FIGURES_DIR}/{title}"

plot_stats_timefrequency_decoding(presence_tf_avg, presence_sign_points, color, times, title,
                                      save_figure)                                   
 
#%% subject awareness
awareness_tf = load_scores_tf(DECOD_TFR, 'decoding_time-frequency_awareness_lsvm', average = False) 

awareness_tf_avg = np.mean(awareness_tf, axis = 0)

# Wilcoxon signed-rank test with FDR correction

stats_file = 'stats_time-frequency_decoding_awareness'

reject, p_corrected, awareness_sign_points = wilcoxon_fdr_tfdecod(awareness_tf, times, stats_file)

# plot results with contour for significant time-frequency bins
title = "Decoding subject awareness on time-frequency data"
color = 'Reds'
save_figure = f"{FIGURES_DIR}/{title}"

plot_stats_timefrequency_decoding(awareness_tf_avg, awareness_sign_points, color, times, title,
                                      save_figure)         

#%% gabor tilt
tilt_tf = load_scores_tf(DECOD_TFR, 'decoding_time-frequency_tilt_lsvm', average = False) 

tilt_tf_avg = np.mean(tilt_tf, axis = 0)

# Wilcoxon signed-rank test with FDR correction

stats_file = 'stats_time-frequency_decoding_tilt'
reject, p_corrected, tilt_sign_points = wilcoxon_fdr_tfdecod(tilt_tf, times, stats_file)

# plot results with contour for significant time-frequency bins
title = "Decoding Gabor tilt on time-frequency data"
color = 'Greens'
save_figure = f"{FIGURES_DIR}/{title}"

plot_stats_timefrequency_decoding(tilt_tf_avg, tilt_sign_points, color, times, title,
                                      save_figure)  

#%% Target presence seen vs unseen

presence_seen = load_scores_tf(DECOD_TFR, 'decoding_time-frequency_presence-seen', average = False) 
presence_seen_avg = np.mean(presence_seen, axis = 0)

# # Wilcoxon signed-rank test with FDR correction
x = presence_seen
reject, p_corrected, sign_points = wilcoxon_fdr_tfdecod(x)

# plot results with contour for significant time-frequency bins
title = "Decoding Gabor tilt on time-frequency data"
av_tfr = AverageTFR(create_info(['freq'], sfreq), presence_seen_avg[np.newaxis, :], freqs = freqs[0:], nave=0, times=times)
av_tfr.plot('all', title=title, cmap='Blues', vmin = 0.5, vmax = np.abs(presence_seen_avg).max(), 
            mask = sign_points, mask_style = 'contour', yscale='linear'
            )
plt.gcf().set_size_inches(15, 10)
# plt.savefig(f"{FIGURES_DIR}/{title}")
# plt.close()
plt.show()

presence_unseen = load_scores_tf(DECOD_TFR, 'decoding_time-frequency_presence-unseen', average = False) 
presence_unseen_avg = np.mean(presence_unseen, axis = 0)

# # Wilcoxon signed-rank test with FDR correction
x = presence_unseen
reject, p_corrected, sign_points = wilcoxon_fdr_tfdecod(x)

# plot results with contour for significant time-frequency bins
title = "Decoding Gabor tilt on time-frequency data"
av_tfr = AverageTFR(create_info(['freq'], sfreq), presence_unseen_avg[np.newaxis, :], freqs = freqs[0:], nave=0, times=times)
av_tfr.plot('all', title=title, cmap='Blues', vmin = 0.5, vmax = np.abs(presence_unseen_avg).max(), 
            mask = tilt_sign_points, mask_style = 'contour', yscale='linear'
            )
plt.gcf().set_size_inches(15, 10)
# plt.savefig(f"{FIGURES_DIR}/{title}")
# plt.close()
plt.show()