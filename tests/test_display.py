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
 
        samples = [X[0, 0]]

        visualization.plot_eeg_samples(samples)

if __name__ == '__main__':
    unittest.main()