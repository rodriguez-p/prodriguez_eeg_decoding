"""
Author: Pablo Rodríguez-San Esteban (prodriguez@ugr.es)

Defines and runs functions for the generalization across blocks analysis (training on localizer blocks and testing on experimental).
"""

# import packages
import mne
from mne.decoding import (GeneralizingEstimator, cross_val_multiscore, Vectorizer)

import numpy as np

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score

# set up directories
DATA_EPOCH = '../data/epochdata' # directory where our epoched data is stored
SAVE_DATA = '../data/decoding_time' # folder to save our data after running the decoding analyses
FILE_PREFIX = 'PRODRIGUEZ_' # prefix of the files, set up during BrainVision recording'

# define functions
def blocks_generalization(id, condition, model):

    # read epochs from the localizer and experimental blocks
    epochs_experimental = mne.read_epochs("{}/{}{:06d}-epo.fif".format(DATA_EPOCH, FILE_PREFIX, id))
    epochs_localizer = mne.read_epochs("{}/{}{:06d}_localizer-epo.fif".format(DATA_LOCALIZER, FILE_PREFIX, id))

    # select the condition to load (target presence or tilt orientation), and combine event codes
    if condition == 'presence':
        epochs_present = mne.epochs.combine_event_ids(epochs['Present/Seen/Left', 'Present/Seen/Right'], ['Present/Seen/Left', 'Present/Seen/Right'], {'Present': 100})
        epochs_absent = mne.epochs.combine_event_ids(epochs['Absent/Seen', 'Absent/Unseen'], ['Absent/Seen', 'Absent/Unseen'], {'Absent': 101})
        epochs_present_localizer = mne.epochs.combine_event_ids(epochs_localizer['Present/Seen/Left', 'Present/Seen/Right'], ['Present/Seen/Left', 'Present/Seen/Right'], {'Present_Localizer': 200})
        epochs_absent_localizer = mne.epochs.combine_event_ids(epochs_localizer['Absent/Unseen'], ['Absent/Unseen'], {'Absent_Localizer': 201})

        epochs = mne.concatenate_epochs([epochs_present, epochs_absent], add_offset=True)
        epochs_experimental = epochs['Present', 'Absent']
        epochs_localizer = mne.concatenate_epochs([epochs_present_localizer, epochs_absent_localizer], add_offset=True)
        epochs_localizer = epochs_localizer['Present', 'Absent']

    elif condition == 'tilt':
        epochs_right = mne.epochs.combine_event_ids(epochs['Present/Seen/Right'], ['Present/Seen/Right'], {'Tilt Right': 100})
        epochs_left = mne.epochs.combine_event_ids(epochs['Present/Seen/Left'], ['Present/Seen/Left'], {'Tilt Left': 101})
        epochs_right_localizer = mne.epochs.combine_event_ids(epochs_localizer['Present/Seen/Right'], ['Present/Seen/Right'], {'Tilt Right_Localizer': 200})
        epochs_left_localizer = mne.epochs.combine_event_ids(epochs_localizer['Present/Seen/Left'], ['Present/Seen/Left'], {'Tilt Left_Localizer': 201})

        epochs = mne.concatenate_epochs([epochs_right, epochs_left], add_offset=True)
        epochs_experimental = epochs['Tilt Right', 'Tilt Left']
        epochs_localizer = mne.concatenate_epochs([epochs_right_localizer, epochs_left_localizer], add_offset=True)
        epochs_localizer = epochs_localizer['Tilt Right', 'Tilt Left']

    # select training data (from localizer blocks) and testing data (experimental blocks)
    X_train = epochs_localizer.get_data()
    y_train = epochs_localizer.events[:,-1]
    X_test = epochs_experimental.get_data()
    y_test = epochs_experimental.events[:,-1]


    # define the classifier pipeline
    clf = make_pipeline(Vectorizer(), StandardScaler(), SVC(kernel='linear', probability=True, class_weight="balanced", max_iter=-1))

    # apply MNE GeneralizingEstimator
    block_gen = GeneralizingEstimator(clf, scoring='roc_auc', n_jobs=-1, verbose=True)

    # fit the classifier to the localizer data
    block_gen.fit(X_train, y_train)

    # test the classifier on experimental data
    scores = block_gen.score(X_test, y_test)

# run function across subjects for the different conditions
scores_gen = None
conditions =  ['presence', 'tilt']
models = ['lsvm', 'lda']

for m in models:
    for c in conditions:
        for id in range(10, 43):
            if id == 11 or id == 24 or id == 26:
                pass
            else:
                mean_scores_gen = blocks_generalization(id, c)
                if np.any(scores_gen):
                    scores_gen = np.vstack((scores_gen, mean_scores_gen))
                else:
                    scores_gen = mean_scores_gen

#%% save data to files
np.save("{}/{}".format(SAVE_DATA, 'gen_loc_exp_tfr.npy'), scores_gen, allow_pickle=True, fix_imports=True)
