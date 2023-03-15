#%% import packages
import mne
from mne.preprocessing import EOGRegression
import numpy as np
import matplotlib.pyplot as plt
from pyprep.find_noisy_channels import NoisyChannels
import pyprep

# %% set up directories
RAW_DATA = '../raw_data' # directory with the .eeg, .vmrk and .vhdr files
RAW_CLEAN = '../raw_clean'  # directory where our cleaned raw data will be saved
DATA_EPOCH = '../epochdata' # directory where our epoched data will be saved
FILE_PREFIX = 'PRODRIGUEZ_' # prefix of the files, set up during BrainVision recording
MONTAGE_FILE = 'easycap-M10.txt' # EEG cap montage file

# %% load raw data
def preprocessing(id, block_type):
    SUBJECT=id
    raw = mne.io.read_raw_brainvision("{}/S{}/{}{:06d}.vhdr".format(RAW_DATA, SUBJECT, FILE_PREFIX), preload=True, eog={'EOG1', 'EOG2'})
    montage = mne.channels.read_custom_montage('../data/easycap-M10.txt')
    mapping = {'FCz': 'Iz'} # recode channels that were incorrectly named in the BrainVision Recorder mapping
    mne.rename_channels(raw.info, mapping)
    raw.set_montage(montage)

    # reduce EOG artifacts through regression
    raw_eog = raw
    raw_eog.set_eeg_reference('average')
    raw_eog.filter(0.3, None, picks='all')
    weights = EOGRegression().fit(raw_eog)
    raw = weights.apply(raw, copy=True)

    # notch filter to correct power line noise
    freqs = (50, 100, 150)
    raw_notch = raw.notch_filter(freqs=freqs, n_jobs=-1, method='spectrum_fit', p_value=0.05, verbose=None)

    # automatic detection of bad channels
    nd=NoisyChannels(raw_notch)
    ransac_bads = nd.find_bad_by_ransac(channel_wise=False)
    raw_notch.info['bads'] = nd.bad_by_ransac

    # interpolate bad channels
    raw_interp = raw_notch.interpolate_bads(reset_bads=True)

    # delete some unused objects to free some memory
    del raw_notch
    del nd

    # low-pass filter
    raw_filt = raw_interp.filter(l_freq=None, h_freq=None, n_jobs=-1)
    del raw_interp

    # # re-reference to average
    raw_avg_ref = raw_filt.set_eeg_reference(ref_channels='average')
    del raw_filt

    # save our clean raw data to file
    raw_avg_ref.save("{}/S{}/{}{:06d}_raw_clean.fif".format(RAW_CLEAN, SUBJECT, FILE_PREFIX, SUBJECT), overwrite=True)
    raw = raw_avg_ref

    # select events for epochs segmentation
    events = mne.events_from_annotations(raw, event_id='auto', regexp='Stimulus/')

    def recode_gabor_events(input_events, block_type):
    #     """
    #     This functions returns a new events list with the time information (time_stamps) of our events (triggers) for the screen of the presentation of the Gabor stimulus (code = 10) that were sent during
    #     the experimental session, but the code of the new events are replaced by some of the following depending on the trial type:
    #     - Present/Seen: 55
    #     - Present/Unseen: 56
    #     - Absent/Seen: 57
    #     - Present/Unseen: 59

    #     Parameters
    #     ----------
    #     input_events : numpy array of size (Nx3)
    #         Array with the experiment events
    #     block_type: block type for which we are searching the events: experimental (71) or localizador (70)
    #
    #     Returns
    #     -------
    #     Numpy array of size (Nx3)
    #         Numpy array with the recoded events
    #     """
        good_trial_types = [55, 56, 57, 59]
        gabor_tilt = [74, 75]
        event_dict = {'Present/Seen/Left': 5574,
                      'Absent/Unseen': 56,
                      'Absent/Seen': 57,
                      'Present/Unseen/Left': 5974,
                      'Present/Seen/Right': 5575,
                    'Present/Unseen/Right': 5975}
        output_events = []
        # Recorremos todos los eventos de presentación del Gabor y vemos el siguiente trigger de anotación del trial
        # Run through all of the Gabor presentation events and check the following trial type trigger
        trial_types_array = np.squeeze(input_events[:, 2])
        gabor_event_indexs = np.squeeze(np.argwhere(trial_types_array == 10))
        for i in range(0, len(gabor_event_indexs)):
            # Índice en el vector input_events del trigger actual de presentación del Gabor
            # Index in the input_events vector of the Gabor presentation trigger
            e = gabor_event_indexs[i]
            # Índice en el vector input_events del siguiente trigger de presentación del Gabor
            # Index in the input_events vector of the next Gabor presentation trigger
            next_e = gabor_event_indexs[i+1] if (i + 1) < len(gabor_event_indexs) else None
            time_stamp = input_events[e, 0]
            # Buscamos entre los triggers que se han mandado en este trial el código de alguno de los de good_trial_types
            # Search among the triggers sent in this trial some of the codes included in good_trial_types
            trials_events = trial_types_array[e+1:next_e]
            bt = np.isin(trials_events, [block_type])
            if np.any(bt):
                mask1 = np.isin(trials_events, good_trial_types)
                mask2 = np.isin(trials_events, gabor_tilt)
                if np.any(mask1):
                    trial_type = trials_events[mask1][0]
                    if np.any(mask2):
                        tilt_type = trials_events[mask2][0]
                        event_type = 100*trial_type + tilt_type
                    else: # For absent trials there are no Gabor stimuli -> no tilt information
                        event_type = trial_type
                    output_events.append((time_stamp, 0, event_type))

        # returns the recoded events array
        return np.array(output_events), event_dict


    gabor_events, events_dict = recode_gabor_events(events[0], block_type)

    # build the epochs with the recoded events
    epochs = mne.Epochs(raw, events=gabor_events, event_id=events_dict, tmin=-2.0, tmax=2.0, baseline=None, preload=True, on_missing='warn')

    # apply baseline correction
    epochs.apply_baseline(baseline=(-2, 0))

    # original sampling frequency of 1000Hz, resample to 256Hz to save computation time and reduce file size
    epochs.resample(sfreq=256)

    # save our clean and segmented epochs
    if block_type == 70:
        epochs.save("{}/{}{:06d}-epo_localizer.fif".format(DATA_EPOCH, FILE_PREFIX, SUBJECT), overwrite=True)
    elif:
        epochs.save("{}/{}{:06d}-epo.fif".format(DATA_EPOCH, FILE_PREFIX, SUBJECT), overwrite=True)

#%% run the function across subjects
block_type = [70, 71]
for block in block_type:
    for id in range(10, 43): # the range of all our subjects, in this case (10, 43)
        if id == 11 or id == 26: # indicate subjects excluded from the analysis
            pass
        else:
            preprocessing(id, block)
