# %%
from autoreject import get_rejection_threshold
from autoreject import AutoReject
import mne
import numpy as np
import matplotlib.pyplot as plt
from pyprep.find_noisy_channels import NoisyChannels
import pyprep

# %%
DATA_ROOT = '/home/ubuntu/Escritorio/prodriguez_eeg_decoding/data'
SUBJECT = 10
FILE_PREFIX = 'PRODRIGUEZ_'
MONTAGE_FILE = 'easycap-M10.txt'

# =============================================================================
# # Load raw data
# =============================================================================

# %%
raw = mne.io.read_raw_brainvision("{}/S{}/{}{:06d}.vhdr".format(
    DATA_ROOT, SUBJECT, FILE_PREFIX, SUBJECT), preload=True, eog={'EOG1', 'EOG2'})
# .crop(tmax=60)
montage = mne.channels.read_custom_montage(
    fname='{}/{}'.format(DATA_ROOT, MONTAGE_FILE))
mapping = {'FCz': 'Iz'}
mne.rename_channels(raw.info, mapping)
raw.set_montage(montage)
sfreq = raw.info['sfreq']

# %%
# montage.plot()

# %%
# file info
# print(raw.info)

# %%
# channel names list
# print(raw.ch_names)

# %%
# plot raw signal to check artifacts/bad channels
# raw.plot()

# %%
# plot channels spectrum to check power line noise
#raw.plot_psd(picks='eeg', fmax=raw.info['sfreq']/2.0)

# %%
# plot channels spectrum to check power line noise
#raw.plot_psd(picks='eeg', fmax=raw.info['sfreq']/2.0, tmax=15*60., average=True)

# %%
# if there is power line distorsion, apply filter
freqs = (50, 100, 150, 200)
raw_notch_fit = raw.notch_filter(
    freqs=freqs, method='spectrum_fit', p_value=0.05, n_jobs=4, verbose=None)
#raw_notch_fit.plot_psd(picks='eeg', fmax=raw.info['sfreq']/2.0, tmax=15*60., average=True)

# %%
nd = NoisyChannels(raw)
nd.find_bad_by_ransac(channel_wise=False, max_chunk_size=1)

# %%
# use the average of all channels as reference
raw_avg_ref = raw_notch_fit.set_eeg_reference(ref_channels='average')
raw_avg_ref.save("{}/S{}/{}{:06d}_raw_clean.fif".format(DATA_ROOT,
                 SUBJECT, FILE_PREFIX, SUBJECT), overwrite=True)  # guardamos la señal
# raw_avg_ref.plot()

# %%
del raw
del raw_notch_fit
del raw_avg_ref

# =============================================================================
# # 2. Create epoched data based on events
# =============================================================================

# %%
# load cleaned and averaged raw data
raw_clean = mne.io.read_raw_fif("{}/S{}/{}{:06d}_raw_clean.fif".format(DATA_ROOT, SUBJECT, FILE_PREFIX,
                                SUBJECT), allow_maxshield=False, preload=False, on_split_missing='raise', verbose=None)

# %%
# read events from annotations
events = mne.events_from_annotations(
    raw_clean, event_id='auto', regexp='Stimulus/')
print(events)

# %%


def recode_gabor_events(input_events):
    """
    Esta función devuelve una nueva lista de eventos con la información temporal (time_stamps) de los eventos (triggers) de tipo presentación de Gabor (código=10) que se enviaron durante el experimento,
    pero donde el código de los nuevos eventos se reemplaza alguno de los siguientes tipos en función del tipo de ensayo:
    - Present/Seen: 55
    - Present/Unseen: 56
    - Absent/Seen: 57
    - Present/Unseen: 59

    Parameters
    ----------
    input_events : numpy array of size (Nx3)
        Array con los eventos del experimento

    Returns
    -------
    Numpy array of size (Nx3)
        Array Numpy con los eventos de tipo presentación de Gabor con los códigos recodificados
    """
    good_trial_types = [55, 56, 57, 59]
    experimental_block = [71]
    event_dict = {'Present/Seen': 55,
                  'Absent/Unseen': 56,
                  'Absent/Seen': 57,
                  'Present/Unseen': 59}
    output_events = []
    # Recorremos todos los eventos de presentación del Gabor y vemos el siguiente trigger de anotación del trial
    trial_types_array = np.squeeze(input_events[:, 2])
    gabor_event_indexs = np.squeeze(np.argwhere(trial_types_array == 10))
    for i in range(0, len(gabor_event_indexs)):
        # Índice en el vector input_events del trigger actual de presentación del Gabor
        e = gabor_event_indexs[i]
        # Índice en el vector input_events del siguiente trigger de presentación del Gabor
        next_e = gabor_event_indexs[i+1] if i + \
            1 < len(gabor_event_indexs) else None
        time_stamp = input_events[e, 0]
        # Buscamos entre los triggers que se han mandado en este trial el código de alguno de los de good_trial_types
        trials_events = trial_types_array[e+1:next_e]
        block_type = np.isin(trials_events, experimental_block)
        if np.any(block_type):
            mask = np.isin(trials_events, good_trial_types)
            if np.any(mask):
                output_events.append((time_stamp, 0, trials_events[mask][0]))
#        for i in range(e+1, len(input_events)):
#            etype = input_events[i, 2]
#            if etype==10 or etype==58: # No se ha grabado la respuesta con el tipo de trial o el trial es erróneo
#                break
#            elif etype in trial_types:
#                # Creamos un nuevo evento con el tipo de trial
#                output_events.append((time_stamp, 0, etype))

    return np.array(output_events), event_dict


# %%
gabor_events, events_dict = recode_gabor_events(events[0])
gabor_events.shape

# %%
# Segmentamos la señal en épochs (ensayos) en base a los eventos
epochs = mne.Epochs(raw_clean, gabor_events, event_id=events_dict,
                    tmin=-2.0, tmax=2.0, baseline=(-2., -0.01), preload=True)
epochs

# %%
# trials['Present/Seen'].plot()

# %%
# repairing EOG artifacts with regression
_, betas = mne.preprocessing.regress_artifact(epochs.subtract_evoked())
# We then use those coefficients to remove the EOG signal from the original data
epochs_eog_clean, _ = mne.preprocessing.regress_artifact(epochs, betas=betas)

raw_clean.load_data()
raw_clean.filter(None, 40)
eog_trials = mne.preprocessing.create_eog_epochs(raw_clean)

# # get ready to plot
# order = np.concatenate([
#      mne.pick_types(raw_clean.info, meg=False, eog=True, ecg=False),
#      mne.pick_types(raw_clean.info, eeg=True)])

# raw_kwargs = dict(events=eog_trials.events, order=order, start=20, duration=5,n_channels=40)

# # # plot original data
# # raw.plot(**raw_kwargs)

# regress (using betas computed above) & plot
# raw_eog_clean, _ = mne.preprocessing.regress_artifact(raw_clean, betas=betas)
# raw_eog_clean.plot(**raw_kwargs)

# del epochs
# del betas
# del raw_clean

# %%
reject = get_rejection_threshold(epochs_eog_clean)
print(reject)

# %%

n_interpolates = np.array([1, 2, 4])
consensus = np.linspace(0.5, 1.0, 6)

ar = AutoReject(n_interpolates, consensus, thresh_method='random_search',
                random_state=42)

ar.fit(epochs_eog_clean)
# %%
for ch_name in epochs_eog_clean.info['ch_names'][:5]:
    print('%s: %s' % (ch_name, ar.threshes_[ch_name]))

plt.hist(np.array(list(ar.threshes_.values())), 30, color='g', alpha=0.4)

# %%
reject_log = ar.get_reject_log(epochs_eog_clean)
reject_log.plot()

# %%
reject_log.plot_epochs(epochs_eog_clean)

# %%
epochs_clean = ar.transform(epochs_eog_clean)

# %%
evoked = epochs_eog_clean.average()
evoked.info['bads'] = ['Fp1', 'Fz', 'F3', 'F7', 'FT9']
evoked.plot(exclude=[])

evoked_clean = epochs_clean.average()
evoked_clean.info['bads'] = []
evoked_clean.plot(exclude=[])


# %%
# =============================================================================
# ch_names = [epochs.ch_names[ii] for ii in ransac.picks][7::10]
# fig, ax = plt.subplots(1, 1, figsize=(12, 6))
# ax.imshow(ransac.bad_log, cmap='Reds',
#           interpolation='nearest')
# ax.grid(False)
# ax.set_xlabel('Sensors')
# ax.set_ylabel('Trials')
# plt.setp(ax, xticks=range(7, len(ransac.picks), 10),
#          xticklabels=ch_names)
# plt.setp(ax.get_yticklabels(), rotation=0)
# plt.setp(ax.get_xticklabels(), rotation=90)
# ax.tick_params(axis=u'both', which=u'both', length=0)
# fig.tight_layout(rect=[None, None, None, 1.1])
# plt.show()
# =============================================================================

# %%
trials_eog_clean.save("{}/S{}/{}{:06d}-epo.fif".format(DATA_ROOT, SUBJECT, FILE_PREFIX,
                      SUBJECT), overwrite=True)  # guardamos la señal segmentada en un archivo
