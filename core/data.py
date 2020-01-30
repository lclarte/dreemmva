# data.py
# script to load, handle and display data

import h5py
import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score

"""
Data loading functions
"""

def load_x(file_name):
    """
    Load X dataset from h5 file.
    Warning : returns the data in the format (N, C, H, W) = (_, 40, 7, 500)
    """
    with h5py.File(file_name, 'r') as file:
        print('Started loading file', file_name)
        data = file['features'][()]
        print('Finished loading the file.')
        data = np.array(data)
    return data

def load_y(file_name):
    """
    Load Y data from csv
    """
    print('Started loading file', file_name)
    data = pd.read_csv(file_name)
    print('Finished loading the file.')
    data = np.array(data)
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
    x2 = np.copy(x)
    x2.shape = (n, h, w, c)
    return x2

def flatten_x(x):
    """
    Applatit les elements de X i.e prend les 40 echantillons independants et en fait 40 inputs
    differents. 
    """
    n, c, h, w = x.shape
    return x.reshape((n*c, 1, h, w), order='C')
    
def flatten_y(y : np.ndarray, repeat : int):
    """
    Recopie chaque entree de y un nombre repeat de fois
    """
    n, _ = y.shape
    return np.tile(y, (1, repeat)).reshape((n*repeat, 2), order='C')

def categorize_y(y):
    """
    
    """

def flatten_data(x, y):
    """
    takes the 40 independent samples and puts them in 40 different data points
    So we have 40 times more data points
    Shape of the input x array : N, C, H, W = (N, 40, 7, 500) here
    Shape of the output x2 array : N*C, 1, H, W (only one channel)
    """
    assert (len(x.shape) == 4)
    return flatten_x(x), flatten_y(y, repeat=x.shape[1])

def load_all(name_x, name_y):
    """
    Charger les donees x et y correspondentes. Les retourne au format (N, 40, 7, 500)
    """
    X = load_x(name_x)
    Y = vectorize_y(load_y(name_y))
    return X, Y

def compare_predict(y_pred, y_true):
    return accuracy_score(y_true, y_pred)

"""
data transformation functions
"""

def fft_eeg(xs):
    """
    Fait une transformee de Fourier sur les 7 channels des donnees en entree
    Taille : (_, 7, 500, 1) car le format d'entree est NHWC
    """
    shape = xs.shape
    # xs must be an
    if len(shape) == 3:
        xs.shape = (1,) + shape
    elif len(shape) < 3:
        raise ValueError()
    # fourier transform sur l'avant derniere coordonnees
    return np.fft.fft(xs, axis=2)
    
