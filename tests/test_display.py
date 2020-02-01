import sys
sys.path.append('../core')

import data
import visualization
import numpy as np
import unittest

class DisplayTest(unittest.TestCase):
    def test_display(self):
        X = np.array(data.load_x('../data/X_train.h5'))
        X2 = data.flatten_x(X)
 
        samples = [X[0, i] for i in range(10)]

        for i in range(10):
            visualization.plot_eeg_sample(samples[i])

    def test_fft_display(self):
        X = np.array(data.load_x('../data/X_train.h5'))
        X = data.reorder_nhwc(data.flatten_x(X))
        X = data.fft_eeg(X)

        samples = [X[i, :, :, 0] for i in range(10)]

        for i in range(10):
            visualization.plot_eeg_sample(samples[i])

    def test_butter_display(self):
        X = np.array(data.load_x('../data/X_train.h5'))
        X = data.reorder_nhwc(data.flatten_x(X))

        samples = [X[i, :, :, 0] for i in range(10)]

        for i in range(10):
            y = data.subsample(samples[i])
            y = data.bandpass_filter(y)
            x_i = samples[i][:, 0:500:2]
            visualization.plot_eeg_samples([x_i, y])

if __name__ == '__main__':
    unittest.main()