"""
Author: Pablo Rodríguez-San Esteban (prodriguez@ugr.es)

Defines and runs functions for the temporal decoding on the epoched data for the localizer blocks.
"""

# import packages
import numpy as np

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import mne
from mne.decoding import (SlidingEstimator, cross_val_multiscore, Vectorizer)

# set up directories
DATA_EPOCH = '../epochdata' # directory where our epoched data is stored
SAVE_DATA = '../epochdata/temporal_decoding' # folder to save our data after running the decoding analyses
FILE_PREFIX = 'PRODRIGUEZ_' # prefix of the files, set up during BrainVision recording

# define the temporal decoding functions
def temporal_decoding(id, condition, model):
    # read epoched data
    epochs = mne.read_epochs("{}/{}{:06d}-epo_localizer.fif".format(DATA_EPOCH, FILE_PREFIX, id))

    # select the condition to load (target presence or tilt orientation), and combine event codes
    if condition == 'presence':
        epochs_present = mne.epochs.combine_event_ids(epochs['Present/Seen/Left', 'Present/Seen/Right'], ['Present/Seen/Left', 'Present/Seen/Right'], {'Present': 100})
        epochs_absent = mne.epochs.combine_event_ids(epochs['Absent/Seen', 'Absent/Unseen'], ['Absent/Seen', 'Absent/Unseen'], {'Absent': 101})
        epochs = mne.concatenate_epochs([epochs_present, epochs_absent], add_offset=True)
        epochs = epochs['Present', 'Absent']

    elif condition == 'tilt':
        epochs_right = mne.epochs.combine_event_ids(epochs['Present/Seen/Right'], ['Present/Seen/Right'], {'Tilt Right': 100})
        epochs_left = mne.epochs.combine_event_ids(epochs['Present/Seen/Left'], ['Present/Seen/Left'], {'Tilt Left': 101})
        epochs = mne.concatenate_epochs([epochs_right, epochs_left], add_offset=True)
        epochs = epochs['Tilt Right', 'Tilt Left']

    # select data (X) and labels (y) from epochs
    X = epochs_PA.get_data()
    y = epochs_PA.events[:,-1]

    # define the classifier pipeline
    if model == 'lsvm':
        clf = make_pipeline(Vectorizer(), StandardScaler(), SVC(kernel='linear', probability=True, class_weight="balanced", max_iter=-1))

    elif model == 'lda':
        clf = make_pipeline(Vectorizer(), StandardScaler(), LinearDiscriminantAnalysis())

    # apply the slidingestimator to fit and test classifier across all time points
    time_decod = SlidingEstimator(clf, n_jobs=-1, scoring='roc_auc', verbose=True)

    # define cross-validator and cv splits
    scores_td = cross_val_multiscore(time_decod, X, y, cv=10, n_jobs=-1)

    # compute mean scores across cross-validation splits
    mean_scores_td = np.mean(scores_td, axis=0)
    return mean_scores_td

# run classifier across subjects for the different conditions
avg_td = None
conditions = ['presence', 'tilt']
models = ['lsvm', 'lda']

for m in models:
    for c in conditions:
        for id in range(10, 43): # the range of all our subjects, in this case (10, 43)
            if id == 11 or id == 24 or id == 26: # indicate subjects excluded from the analysis
                pass
            else:
                mean_scores_id = temporal_decoding(id, c, m)
                np.save("{}/S{}_temporal_decoding_{}_{}_localizer.npy".format(SAVE_DATA, id, c, m)) # save subject data

                if np.any(avg_td):
                    avg_td = np.vstack((avg_td, mean_scores_id))
                else:
                    avg_td = mean_scores_id

# save group data
np.save("{}/temporal_decoding_{}_{}_localizer.npy".format(SAVE_DATA, c, m), avg_td, allow_pickle=True, fix_imports=True)
