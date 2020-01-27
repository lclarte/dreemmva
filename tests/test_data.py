import sys
sys.path.append('../core')

import data
import numpy as np
import unittest

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

if __name__ == '__main__':
    unittest.main()