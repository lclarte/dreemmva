# analysis.py
# methods to transform (Wavelets, STFT, spindle detection) EEG signals

import yasa
import numpy as np
from scipy import signal

def spindle_number(x, sampling_frequency):
    """
    Single channel spindle detection. Returns the number of spindles in the given data
    """
    spindles = yasa.spindles_detect(x, sampling_frequency)
    if spindles is None:
        return 0
    return spindles.shape[0]

def compute_stft(x, sampling_frequency, wduration):
    """
    wduration : duration (in second) of each windows where the FT is computed
    returns: 
        f, t, Zxx 
    """
    nperseg = int(wduration * sampling_frequency)
    f, t, Zxx = signal.stft(x, sampling_frequency, nperseg=nperseg)
    return f, t, Zxx

def cut_frequency(Zxx, f, min_freq = -float('inf'), max_freq = float('inf')):
    """
    Retourne une nouvelle matrice contenant uniquement les frequences entre les min_freq et max_freq 
    returns:
        f, Zxx
    """
    indices = [i for i in range(len(f)) if min_freq <= f[i] <= max_freq]
    return f[indices], Zxx[indices, :].squeeze()

def welch_method(x, sf):
    return signal.welch(x, sf)
    
def batch_welch_method(x, sf):
    # shape can be anything BUT timesteps must come last and batch must have 4 dims
    a, b, c, w = x.shape
    x2 = x.reshape((a*b*c, w))
    
# swap sensors because it seems that some are more correlated w/ each other than the others
# the channels to swap are hardcoded

def swap_correlated_channels(x_train, nhwc_format=True):
    """
    If nhwc_format = False, it is in format nchw = (946, 40, 7, 500)
    """
    x_copy = np.copy(x_train)
    if nhwc_format == False:
        x_copy[:, :, [1, 4], :] = x_copy[:, :, [4, 1], :]
    else:
        x_copy[:,, [1, 4] :, :] = x_copy[:, [4, 1], :, :]
    return x_copy