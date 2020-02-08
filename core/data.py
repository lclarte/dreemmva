# data.py
# script to load, handle and display data

import csv
import h5py
import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal as signal
from sklearn.metrics import accuracy_score

"""
Data loading and saving function functions
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
    Load Y data from csv. 
    Returns ONLY the value w/o the index, in the one hot format
    example : in the original file, [(0, 0), (1, 0), (2, 1)] becomes 
    [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    """
    print('Started loading file', file_name)
    data = pd.read_csv(file_name)
    print('Finished loading the file.')
    data = np.array(data)
    return np.eye(2)[data[:, 1]]

def save_y(y, file_name):
    with open(file_name, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(['id', 'label'])
        for i in range(len(y)):
            writer.writerow([str(i), str(y[i])])


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
    n = len(y)
    return np.tile(y, (1, repeat)).reshape((n*repeat, 2), order='C')

def vectorize_y(Y):
    N, _ = Y.shape
    vecY = np.zeros(shape=(N, 2))
    for i in range(N):
        vecY[i, Y[i][1]] = 1.
    return vecY

def categorize_y(y):
    """
    Prend en argument les predictions y sous forme vectorisee et les transforme en categoriel
    {0, 1}
    """
    return np.argmax(y, axis=1)

def flatten_data(x, y):
    """
    takes the 40 independent samples and puts them in 40 different data points
    So we have 40 times more data points
    Shape of the input x array : N, C, H, W = (N, 40, 7, 500) here
    Shape of the output x2 array : N*C, 1, H, W (only one channel)
    """
    assert (len(x.shape) == 4)
    return flatten_x(x), flatten_y(y, repeat=x.shape[1])

def compare_predict(y_pred, y_true):
    return accuracy_score(y_true, y_pred)


def weight_data(y):
    """
    Pondere les inputs selon la representation des différentes classes
    Forme de y : 
    """
    # compte le nombre d'element de y de chaque classe (0 et 1)
    condition = lambda i : np.array_equal(y[i], [0., 1.])
    n = len(y)
    y_1 = np.sum([1. for i in range(n) if condition(i)])
    y_0 = float(n) - y_1
    return np.array([float(n) / y_1 if condition(i) else float(n) / y_0 for i in range(n)])

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

def subsample(x, base_freq=250, target_freq=125): 
    """
    x.shape = (channels, steps = time * base_freq) par exemple (7, 500)
    Idealement, le ratio de frequences base_freq / target_freq doit etre entier
    Arguments par defauts : cf. donnees de Dreem
    returns : 
        - array de shape (channels, time * target_freq)
    """
    ratio = int(base_freq / target_freq)
    return x[:, 0:-1:ratio]

def bandpass_filter(x, low=0.5, high=25, freq=125):
    """
    Filtre de butterworth du premier ordre entre les deux frequences donees en entree 
    Arguments par defaut : on fait un passe bande entre 0.5 et 25 Hz
    Shape de x : (channels, steps)
    """
    b, a = signal.butter(1, [low, high], btype='bandpass', fs=freq)
    y = np.array([signal.lfilter(b, a, x[i]) for i in range(len(x))])
    return y
