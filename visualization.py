# visualization.py
# file to visualize data (EEG)

import matplotlib.pyplot as plt

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

def plot_eeg_samples(xs):
    fig, axs = plt.subplots(nrows=2, ncols=4)
    for i in range(7):
        ax = axs[i // 4, i % 4]
        ax.set_title('Channel ' + str(i+1))
        for l in range(len(xs)):
            ax.plot(xs[l][i])
    plt.show()