import numpy as np
import biosppy.signals.eeg as eeg
import biosppy.signals.emg as emg

# Returns average power of different frequency bands
# Input is a N x 2 x 512 matrix
# Output is a N x 10 matrix
def power_features(x):
    x = x.reshape(-1, 21600, 2, 512)
    pf_all = np.zeros((x.shape[0], 21600, 10))
    for i in range(x.shape[0]):
        pf = eeg.get_power_features(x[i].transpose((0,2,1)).reshape(-1, 2), sampling_rate=128, size=4, overlap=0)
        pf_all[i] = np.hstack(pf[1:])
    return pf_all.reshape(-1, 10)


# Returns Phase-Locking Features (PLF)
# Input is a N x 2 x 512 matrix
# Output is a N x 2 matrix
def PLF(x):
    x = x.reshape(-1, 21600, 2, 512)
    plf_all = np.zeros((x.shape[0], 21600, 2))
    for i in range(x.shape[0]):
        ts, plf_par, plf = eeg.get_plf_features(x[i].transpose((0,2,1)).reshape(-1,2),
                sampling_rate=128, size=4, overlap=0)
        plf_all[i] = plf
    return plf_all.reshape(-1, 2)


# Finds number of onsets for each 4s window
# Input is a N x 512 matrix
# Output is a N array
def onset_features(x):
    x = x.reshape(-1, 21600 * 512)
    onset_freq = np.zeros((x.shape[0], 21600))
    for i in range(x.shape[0]):
        onsets = emg.find_onsets(x[i], sampling_rate=128)[0]
        onset_freq[i] = np.histogram(onsets, bins=21600)[0]
    return onset_freq.flatten()

