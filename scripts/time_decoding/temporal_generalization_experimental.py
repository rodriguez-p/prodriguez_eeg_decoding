"""
Author: Pablo Rodríguez-San Esteban (prodriguez@ugr.es)

Defines and runs functions for the temporal generalization analyses on the epoched data for the experimental blocks.
"""

# import packages
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import mne
from mne.decoding import (SlidingEstimator, cross_val_multiscore, Vectorizer, GeneralizingEstimator)


# set up directories
DATA_EPOCH = '../epochdata' # directory where our epoched data is stored
SAVE_DATA = '../epochdata/temporal_decoding' # folder to save our data after running the decoding analyses
FILE_PREFIX = 'PRODRIGUEZ_' # prefix of the files, set up during BrainVision recording

# define functions
def temporal_generalization(id, condition):

    # read epoched data
    epochs = mne.read_epochs("{}/{}{:06d}-epo.fif".format(DATA_LOCALIZER, FILE_PREFIX, id))

    # select the condition to load (target presence, awareness or tilt orientation), and combine event codes
    if condition == 'presence':
        epochs_present = mne.epochs.combine_event_ids(epochs['Present/Seen/Left', 'Present/Seen/Right'], ['Present/Seen/Left', 'Present/Seen/Right'], {'Present': 100})
        epochs_absent = mne.epochs.combine_event_ids(epochs['Absent/Unseen'], ['Absent/Unseen'], {'Absent': 101})
        epochs = mne.concatenate_epochs([epochs_present, epochs_absent], add_offset=True)
        epochs = epochs['Present', 'Absent']

    elif condition == 'awareness':
        epochs_seen = mne.epochs.combine_event_ids(epochs['Present/Seen/Left', 'Present/Seen/Right'],['Present/Seen/Left', 'Present/Seen/Right'], {'Present/Seen': 100})
        epochs_unseen = mne.epochs.combine_event_ids(epochs['Present/Unseen/Left', 'Present/Unseen/Right'],['Present/Unseen/Left', 'Present/Unseen/Right'], {'Present/Unseen': 101})
        epochs = mne.concatenate_epochs([epochs_seen, epochs_unseen], add_offset=True)
        epochs = epochs['Present/Seen', 'Present/Unseen']

    elif condition == 'tilt':
        epochs_right = mne.epochs.combine_event_ids(epochs['Present/Seen/Right'], ['Present/Seen/Right'], {'Tilt Right': 100})
        epochs_left = mne.epochs.combine_event_ids(epochs['Present/Seen/Left'], ['Present/Seen/Left'], {'Tilt Left': 101})
        epochs = mne.concatenate_epochs([epochs_right, epochs_left], add_offset=True)
        epochs = epochs['Tilt Right', 'Tilt Left']

    # select data (X) and labels (y) from epochs
    X = epochs.get_data()
    y = epochs.events[:,-1]

    # define the classifier pipeline
    clf = make_pipeline(Vectorizer(), StandardScaler(), SVC(kernel='linear', probability=True, class_weight="balanced", max_iter=-1))

    # apply MNE GeneralizingEstimator
    time_gen = GeneralizingEstimator(clf, n_jobs=-1, scoring='roc_auc', verbose=True)

    # define cross-validator and cv splits
    scores_tg = cross_val_multiscore(time_gen, X, y, cv=10)

    # compute mean scores across cross-validation splits
    mean_scores_tg = np.mean(scores_tg, axis=0)
    return mean_scores_tg

# run classifier across subjects for the different conditions
avg_tg = []
conditions = ['presence', 'awareness', 'tilt']

for c in conditions:
    for id in range(13, 43): # the range of all our subjects, in this case (10, 43)
        if id == 11 or id == 24 or id == 26: # indicate subjects excluded from the analysis
            pass
        else:
            mean_scores_tg = temporal_generalization(id, c)
            np.save("{}/S{}_temporal-generalization_{}_experimental.npy".format(SAVE_DATA, id, c), mean_scores_tg, allow_pickle=True, fix_imports=True) # save subject data

            if np.any(avg_tg):
                avg_tg = np.vstack((avg_tg, mean_scores_tg))
            else:
                avg_tg = mean_scores_tg

# save group data
np.save("{}/group_temporal-generalization_{}_experimental.npy".format(SAVE_DATA, c), avg_tg, allow_pickle=True, fix_imports=True)
