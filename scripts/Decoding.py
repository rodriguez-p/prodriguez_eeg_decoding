#%%
import pathlib
import matplotlib
import matplotlib.pyplot as plt
import mne
import numpy as np

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from mne.decoding import Scaler, Vectorizer, cross_val_multiscore, SlidingEstimator

#%%
DATA_ROOT = '/home/ubuntu/Escritorio/prodriguez_eeg_decoding/data'
SUBJECT = 17
FILE_PREFIX = 'PRODRIGUEZ_'
MONTAGE_FILE = 'easycap-M10.txt'

#%% Load epoched data and select epochs of interest
epochs = mne.read_epochs("{}/S{}/{}{:06d}-epo.fif".format(DATA_ROOT , SUBJECT, FILE_PREFIX, SUBJECT))
epochs_present = epochs['Present/Seen', 'Present/Unseen']

#%% Calculate evoked differences
evoked_diff = mne.combine_evoked(
    [epochs_present['Present/Seen'].average(),
     epochs_present['Present/Unseen'].average()],
    weights=[1, -1]  # Subtraction
)

evoked_diff.plot(gfp=True)
mne.viz.plot_compare_evokeds(
    [epochs_present['Present/Seen'].average(),
     epochs_present['Present/Unseen'].average(),
     evoked_diff]
)

#%% Equalize the number of epochs
epochs_present.equalize_event_counts(epochs_present.event_id)
epochs_present

#%% Create input X and response y
# Create an vector with length = no. of trials.
y = np.empty(len(epochs_present.events), dtype=int)  

# Which trials are LEFT, which are RIGHT?
idx_left = epochs_present.events[:, 2] == epochs_present.event_id['Present/Seen']
idx_right = epochs_present.events[:, 2] == epochs_present.event_id['Present/Unseen']

# Encode: LEFT = 0, RIGHT = 1.
y[idx_left] = 0
y[idx_right] = 1

print(y)
print(f'\nSize of y: {y.size}')

epochs_present = epochs_present.pick_types(eeg=True)
data = epochs_present.get_data()
print(data.shape)

n_trials = data.shape[0]
X = data.reshape(n_trials, -1)
print(X.shape)

#%% Create the classifier
# The classifier pipeline: it is extremely important to scale the data
# before running the actual classifier (logistic regression in our case).
clf = make_pipeline(StandardScaler(),
                    LogisticRegression())

# Run cross-validation.
# CV without shuffling – "block cross-validation" – is what we want here
# (scikit-learn doesn't shuffle by default, which is good for us).
n_splits = 5
scoring = 'roc_auc'
cv = StratifiedKFold(n_splits=n_splits)
scores = cross_val_score(clf, X=X, y=y, cv=cv, scoring=scoring)

# Mean and standard deviation of ROC AUC across cross-validation runs.
roc_auc_mean = round(np.mean(scores), 3)
roc_auc_std = round(np.std(scores), 3)

print(f'CV scores: {scores}')
print(f'Mean ROC AUC = {roc_auc_mean:.3f} (SD = {roc_auc_std:.3f})')

#%% Visualize the cross-validation results
fig, ax = plt.subplots()
ax.boxplot(scores,
           showmeans=True, # Green triangle marks the mean.
           whis=(0, 100),  # Whiskers span the entire range of the data.
           labels=['Seen vs Unseen'])
ax.set_ylabel('Score')
ax.set_title('Cross-Validation Scores')
plt.savefig('Cross-validation scores')

#%%
X = epochs_present.get_data()
y = epochs_present.events[:, 2]

# Classifier pipeline.
clf = make_pipeline(
    # An MNE scaler that correctly handles different channel types –
    # isn't that great?!
    Scaler(epochs_present.info),
    # Remember this annoying and error-prone NumPy array reshaping we had to do
    # earlier? Not anymore, thanks to the MNE vectorizer!
    Vectorizer(),
    # And, finally, the actual classifier.
    LogisticRegression())

# Run cross-validation.
# Note that we're using MNE's cross_val_multiscore() here, not scikit-learn's
# cross_val_score() as above. We simply pass the number of desired CV splits,
# and MNE will automatically do the rest for us.
n_splits = 5
scoring = 'roc_auc'
scores = cross_val_multiscore(clf, X, y, cv=5, scoring='roc_auc')

# Mean and standard deviation of ROC AUC across cross-validation runs.
roc_auc_mean = round(np.mean(scores), 3)
roc_auc_std = round(np.std(scores), 3)

print(f'CV scores: {scores}')
print(f'Mean ROC AUC = {roc_auc_mean:.3f} (SD = {roc_auc_std:.3f})')

#%% Decoding over time
# Classifier pipeline. No need for vectorization as in the previous example.
clf = make_pipeline(StandardScaler(),
                    LogisticRegression())

# The "sliding estimator" will train the classifier at each time point.
scoring = 'roc_auc'
time_decoder = SlidingEstimator(clf, scoring=scoring, n_jobs=1, verbose=True)

# Run cross-validation.
n_splits = 5
scores = cross_val_multiscore(time_decoder, X, y, cv=5, n_jobs=1)

# Mean scores across cross-validation splits, for each time point.
mean_scores = np.mean(scores, axis=0)

# Mean score across all time points.
mean_across_all_times = round(np.mean(scores), 3)
print(f'\n=> Mean CV score across all time points: {mean_across_all_times:.3f}')

#%% Visualize the classification results
fig, ax = plt.subplots()

ax.axhline(0.5, color='k', linestyle='--', label='chance')  # AUC = 0.5
ax.axvline(0, color='k', linestyle='-')  # Mark time point zero.
ax.plot(epochs.times, mean_scores, label='score')

ax.set_xlabel('Time (s)')
ax.set_ylabel('Mean ROC AUC')
ax.legend()
ax.set_title('Seen vs Unseen')
fig.suptitle('Sensor Space Decoding')
plt.savefig('Sensor space decoding')
