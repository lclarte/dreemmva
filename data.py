# data.py
# script to load, handle and display data

import h5py
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
Fonctions pour la visualisation des EEG
"""

def plot_eeg_sample(x):
    fig, axs = plt.subplots(nrows=2, ncols=4)
    for i in range(7):
        ax = axs[i // 4, i % 4]
        ax.set_title('Channel ' + str(i+1))
        ax.plot(x[i])
    plt.show()