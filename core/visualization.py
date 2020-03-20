# visualization.py
# file to visualize data (EEG)

import matplotlib.pyplot as plt
import numpy as np

"""
Fonctions pour la visualisation des EEG
"""

def plot_eeg_sample(x):
    plot_eeg_samples([x])

def plot_eeg_samples(xs):
    """
    Format d'entree : [N, 7] car 7 channels
    """
    fig, axs = plt.subplots(nrows=2, ncols=4)
    for i in range(7):
        ax = axs[i // 4, i % 4]
        ax.set_title('Channel ' + str(i+1))
        ax.set(xlabel='time (seconds)')
        absc = np.linspace(0, 2, 500)
        for l in range(len(xs)):
            ax.plot(xs[l][i])
    plt.show()

def save_eeg_sample(x, title, savefile):
    fig ,axs = plt.subplots(nrows=2, ncols=4)
    for i in range(7):
        ax = axs[i // 4, i % 4]
        ax.set_title(str(i+1))
        ax.plot(x[i])
    plt.suptitle(title)
    plt.savefig(savefile)
    plt.close()

