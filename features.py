import numpy as np
import biosppy.signals.eeg as eeg
import biosppy.signals.emg as emg

# Returns average power of different frequency bands
# Input is a N x 2 x 512 matrix
# Output is a N x 10 matrix
def power_features(x):
    pf_all = np.zeros((x.shape[0], x.shape[1], 5))
    for i in range(x.shape[0]):
        pf = eeg.get_power_features(x[i].transpose(), sampling_rate=128, size=4)
        pf_all[i] = np.vstack(pf[1:]).transpose()
    return pf_all.reshape(-1, 10)


# Returns Phase-Locking Features (PLF)
# Input is a N x 2 x 512 matrix
# Output is a N-array
def PLF(x):
    plf_all = np.zeros((x.shape[0]))
    for i in range(x.shape[0]):
        ts, plf_par, plf = eeg.get_plf_features(x[i].transpose(), sampling_rate=128, size=4)
        plf_all[i] = plf.item()
    return plf_all


# Finds onset indices of EMG pulses
# Input is a N x 512 matrix
# Output is a N array
# Not working yet. emg.emg has problems with our signal
def onset_features(x):
    onset_all = []
    for i in range(x.shape[0]):
        ts, filtered, onset_indices = emg.emg(x[i], sampling_rate=128, show=True)
        onset_all.append(onset_indices)
    return onset_indices.transpose()

