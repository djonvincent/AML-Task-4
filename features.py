import numpy as np
import biosppy.signals.eeg as eeg

# Returns average power of different frequency bands
# Input is a N x 2 x 512 matrix
# Output is a N x 2 x 5 matrix
def power_features(x):
    pf_all = np.zeros((x.shape[0], x.shape[1], 5))
    for i in range(x.shape[0]):
        pf = eeg.get_power_features(x[i].transpose(), sampling_rate=128, size=4)
        pf_all[i] = np.vstack(pf[1:]).transpose()
    return pf_all
    
