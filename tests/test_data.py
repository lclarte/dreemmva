import sys
sys.path.append('../core')

import data
import numpy as np
import unittest
import matplotlib.pyplot as plt

y_train_file = '../data/y_train.csv'

class DataTest(unittest.TestCase):
    def test_flatten_data(self):
        n, c, h, w = 6, 2, 2, 2
        x = np.random.randn(n, c, h, w)
        y = np.array([[0., 1.], [0., 1.], [0., 1.], [1., 0.], [1., 0.], [1., 0.]])
        x2, y2 = data.flatten_data(x, y)
        y_valid = True
        for i in range(12):
            for j in range(2):
                if y2[i, j] != y[i // 2, j]:
                    y_valid = False
        x_valid = True
        for i in range(6):
            for j in range(2):
                for k in range(2):
                    for l in range(2):
                        if x2[2*i + j, 0, k, l] != x[i, j, k, l]:
                            x_valid = False
        self.assertTrue(x_valid and y_valid)

    def test_class_weight(self):
        y = data.load_y(y_train_file)
        y = np.argmax(y, axis=1)
        nb_class_0, nb_class_1 = np.count_nonzero(y == 0), np.count_nonzero(y == 1)
        print('Number of elements in each class : ', nb_class_0, ' & ', nb_class_1)
        print('Ratio of class cardinals : ', nb_class_0 / float(len(y)), ' & ', nb_class_1 / float(len(y)))
        weights = data.class_weights(y)
        print('Class weights : ', weights[0], '&', weights[1])
        

    def test_subsample(self):
        x = np.array(list(range(10))).reshape((1, 10))
        y = data.subsample(x, base_freq=2., target_freq=1.)
        self.assertTrue(np.array_equal(y[0], list(range(0, 10, 2))))
    
    def test_butter(self):
        # Test on synthetic data : sdeux secondes, 100 Hz
        abscisse = np.linspace(-1, 1, 200)
        x = np.cos(np.pi*abscisse)

        epsilon = 0.1
        x_noise = x + epsilon*np.cos(10*np.pi*abscisse)

        y = data.bandpass_filter([x_noise], low=0.1, high=1.0, freq=100)

        plt.plot(abscisse, x_noise)
        plt.plot(abscisse, x)
        plt.plot(abscisse, y[0])
        plt.show()

if __name__ == '__main__':
    unittest.main()