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
    """
    n, c, h, w = x.shape
    x2 = x.reshape((n*c, 1, h, w), order='C')
    y2= np.tile(y, (1, c)).reshape((n*c, 2), order='C')
    return x2, y2

"""
Fonctions pour la visualisation des EEG
"""

def plot_eeg_sample(x):
    fig, axs = plt.subplots(nrows=2, ncols=4)
    for i in range(7):
        ax = axs[i // 4, i % 4]
        ax.set_title('Channel ' + str(i+1))
        ax.plot(x[i])
    plt.show()