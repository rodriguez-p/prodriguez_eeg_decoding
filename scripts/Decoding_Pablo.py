#%%
import pathlib
import matplotlib
import matplotlib.pyplot as plt
import mne
from mne.epochs import equalize_epoch_counts
import numpy as np
from sklearn.pipeline import make_pipeline
from mne.decoding import Scaler, Vectorizer, cross_val_multiscore, LinearModel, get_coef, Vectorizer, CSP, GeneralizingEstimator
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from mne.decoding import SlidingEstimator
from sklearn.metrics import classification_report, confusion_matrix

#%%
DATA_ROOT = '/home/ubuntu/Escritorio/prodriguez_eeg_decoding/data'
SUBJECT = 21
FILE_PREFIX = 'PRODRIGUEZ_'
MONTAGE_FILE = 'easycap-M10.txt'

#%% Load epoched data and select epochs of interest
epochs = mne.read_epochs("{}/S{}/{}{:06d}-epo.fif".format(DATA_ROOT , SUBJECT, FILE_PREFIX, SUBJECT))
epochs_decod = epochs['Present/Seen', 'Present/Unseen']
epochs_seen = epochs['Present/Seen']
epochs_unseen = epochs['Present/Unseen']

# First, create X and y.
X = epochs_decod.get_data()
y = epochs_decod.events[:, 2]

#%% Calculate evoked differences
evoked_diff = mne.combine_evoked(
    [epochs_seen.average(),
     epochs_unseen.average()],
    weights=[1, -1]  # Subtraction
)

evoked_diff.plot(gfp=True)
mne.viz.plot_compare_evokeds(
    [epochs_seen.average(),
     epochs_unseen.average(),
     evoked_diff]
)

#%% Equalize number of epochs per condition
equalize_epoch_counts([epochs_seen, epochs_unseen])

#%% Temporal decoding
clf = make_pipeline(
    StandardScaler(),
    LogisticRegression(solver='liblinear')
)

time_decod = SlidingEstimator(clf, n_jobs=2, scoring='roc_auc', verbose=True)
# here we use cv=3 just for speed
scores = cross_val_multiscore(time_decod, X, y, cv=3, n_jobs=2)

# Mean scores across cross-validation splits
scores = np.mean(scores, axis=0)

# Plot
fig, ax = plt.subplots()
ax.plot(epochs.times, scores, label='score')
ax.axhline(.5, color='k', linestyle='--', label='chance')
ax.set_xlabel('Times')
ax.set_ylabel('AUC')  # Area Under the Curve
ax.legend()
ax.axvline(.0, color='k', linestyle='-')
ax.set_title('Sensor space decoding')

#%% Temporal generalization
# define the Temporal generalization object
time_gen = GeneralizingEstimator(clf, n_jobs=2, scoring='roc_auc',
                                 verbose=True)

# again, cv=3 just for speed
scores = cross_val_multiscore(time_gen, X, y, cv=3, n_jobs=2)

# Mean scores across cross-validation splits
scores = np.mean(scores, axis=0)

# Plot the diagonal (it's exactly the same as the time-by-time decoding above)
fig, ax = plt.subplots()
ax.plot(epochs.times, np.diag(scores), label='score')
ax.axhline(.5, color='k', linestyle='--', label='chance')
ax.set_xlabel('Times')
ax.set_ylabel('AUC')
ax.legend()
ax.axvline(.0, color='k', linestyle='-')
ax.set_title('Decoding MEG sensors over time')

#%% Generalization matrix
fig, ax = plt.subplots(1, 1)
im = ax.imshow(scores, interpolation='lanczos', origin='lower', cmap='RdBu_r',
               extent=epochs.times[[0, -1, 0, -1]], vmin=0., vmax=1.)
ax.set_xlabel('Testing Time (s)')
ax.set_ylabel('Training Time (s)')
ax.set_title('Temporal generalization')
ax.axvline(0, color='k')
ax.axhline(0, color='k')
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('AUC')





