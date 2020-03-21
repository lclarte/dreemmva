# data.py
# script to load, handle and display data

import csv
import h5py
import itertools

from imblearn.under_sampling import RandomUnderSampler
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal as signal
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_class_weight

"""
Data loading and saving function functions. 
"""

def load_x(file_name):
    """
    Load X dataset from h5 file.
    Warning : returns the data in the format (N, C, H, W) = (_, 40, 7, 500)
    => the data must be formatted before use 
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

def save_csv(y, file_name):
    with open(file_name, 'w') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerow(['id', 'label'])
        for i in range(len(y)):
            writer.writerow([str(i), str(int(y[i]))])

"""
Data formatting functions

Utilisation : le format qu'on utilise par defaut est N, H, W, C
Par exemple, (946, 7, 500, 40) channels = nb d'echantillons ici 
ou encore (646 * 40, 7, 500, 1) quand on a "flatten" les echantillons

code : 
x = load_x(file_name)
y = load_y(file_name)

# pour garder C = 40
x = reorder_nhwc(x)

# pour avoir C = 1 et N <- N * 40
x, y = flatten_data(x, y)
OU 
x, y = flatten_x(x), flatten_y(y, 40)
x = reorder_nhwc(x)
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

def categorize_y(y):
    """
    Takes as input y in the one-hot vectors format and returns y in the format 
    0 for men, 1 for women
    """
    return np.argmax(y, axis=1)

def flatten_y(y : np.ndarray, repeat : int):
    """
    Recopie chaque entree de y un nombre repeat de fois
    """
    if len(y.shape) == 1:
        y0 = np.eye(2)[y]
    n = len(y0)
    retour =  np.tile(y0, (1, repeat)).reshape((n*repeat, 2), order='C')
    if len(y.shape) == 1:
        return np.argmax(retour, axis=1)
    return retour

def undersampling(X, y):
    """
    Retourne le plus grande nombre de sample tels qu'il y a autant d'hommes que de femmes
    """
    # Y doit etre categorise
    assert (len(y.shape) == 1 or (len(y.shape == 2) and y.shape[1] == 1))

    # indices where 
    men, women = np.argwhere(y == 0).squeeze(), np.argwhere(y == 1).squeeze()
    nb = min(len(men), len(women))
    men_sub_indices = np.random.choice(men, size=nb, replace=False)
    women_sub_indices = np.random.choice(women, size=nb, replace=False)
    sub_indices = np.concatenate((men_sub_indices, women_sub_indices))
    return X[sub_indices], y[sub_indices]
    
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

def class_weights(y):
    """
    Retourne un poids pour chaque classe
    """
    return compute_class_weight('balanced', np.unique(y), y)

def samples_weights(y):
    """
    Retourne un poids pour chaque input selon la representation des différentes classes
    Forme de y : 1D array
    """
    # compte le nombre d'element de y de chaque classe (0 et 1)
    weights = class_weights(y)
    sweights = np.array([weights[t] for t in y])
    return sweights

def average_predictions(predictions, nb_trials = 40):
    """
    Average the predictions for the nb_trials independent samples to predict
    the sex of each subject.
    """
    # number of samples
    n = int(len(predictions) / nb_trials)
    avg_preds = np.zeros(n)
    for i in range(n):
        sample_preds = predictions[i*nb_trials:(i+1)*nb_trials]
        avg_preds[i] = int(np.mean(sample_preds) > 0.5)
    return avg_preds
