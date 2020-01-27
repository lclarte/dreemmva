# data.py
# script to load, handle and display data

import h5py
import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

"""
Data loading functions
"""

def load_x(file_name):
    """
    Load X dataset from h5 file
    """
    with h5py.File(file_name, 'r') as file:
        print('Started loading file', file_name)
        data = file['features'][()]
        print('Finished loading the file.')
    return data

def load_y(file_name):
    """
    Load Y data from csv
    """
    print('Started loading file', file_name)
    data = pd.read_csv(file_name)
    print('Finished loading the file.')
    return data

def vectorize_y(Y):
    N, _ = Y.shape
    vecY = np.zeros(shape=(N, 2))
    for i in range(N):
        vecY[i, Y[i][1]] = 1.
    return vecY

"""
data formatting functions
"""

def reorder_nhwc(x):
    n, c, h, w = x.shape
    x.shape = (n, h, w, c)

def flatten_data(x, y):
    """
    takes the 40 independent samples and puts them in 40 different data points
    So we have 40 times more data points
    Shape of the input x array : N, C, H, W = (N, 40, 7, 500) here
    Shape of the output x2 array : N*C, 1, H, W (only one channel)
    """
    n, c, h, w = x.shape
    x2 = x.reshape((n*c, 1, h, w), order='C')
    y2= np.tile(y, (1, c)).reshape((n*c, 2), order='C')
    return x2, y2

"""
data transformation functions
"""

def fft_eeg(x):
    """
    TODO
    Fait une transformee de Fourier sur les 7 channels donnes en entree
    """
    pass