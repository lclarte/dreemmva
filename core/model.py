# model.py
# classe abstraite qui inclut les NN et autres modeles (modeles lineaires)

from core.network import *
import core.data as data
import numpy as np

import os
import psutil

class TrainableModel:
    def __init__(self, config):
        self.init(config)

    def init(self, config):
        pass
        
    def train(self, x_train, y_train):
        pass

    def predict(self, x_test):
        pass

class NNModel(TrainableModel):
    def __init__(self, config):
        self.init(config)

    def init(self, config):
        factories = {'SimpleNetworkFactory' : SimpleNetworkFactory, 'BaseNetworkFactory' : BaseNetworkFactory}
        # define useful constants
        self.epoch            = config['epoch'] 
        self.optimizer        = config['optimizer']
        self.batch_size       = 32
        self.validation_split = 0.25

        # initialize NN
        factory = factories[config['nn_factory']]()
        self.network = factory.get_network()
        self.network.compile(loss='binary_crossentropy', optimizer=self.optimizer, metrics=['accuracy'])

    def train(self, x_train, y_train):
        """
        x_train.shape = (N, 40, 7, 500)
        y_train.shape = (N, 2)
        """
        x_train, y_train = data.flatten_data(x_train, y_train)
        x_train = data.reorder_nhwc(x_train)
        # x_train = data.fft_eeg(x_train)

        weights = data.get_sample_weights(y_train)

        self.network.fit(x_train, y_train, epochs=self.epoch, batch_size=self.batch_size, validation_split=self.validation_split, sample_weight=weights)
        return True

    def set_network(self, network):
        self.network = network

    def get_network(self):
        return self.network

    def predict(self, x_test):
        """
        Retourne la prediction pour chacune des classes. Format : y \in {0, 1}
        x_test.shape = (N_test, 40, 7, 500)
        If y_test is not None, we compare our output to compute precision
        """
        n, c, h, w = x_test.shape
        # classify data one by one
        y = np.zeros(shape=(n, 2))
        for i in range(n):
            x = x_test[i].reshape((1, c, h, w))
            x = data.fft_eeg(data.reorder_nhwc(data.flatten_x(x)))
            # set of 40 predictions, one by independant sample
            predictions = self.network.predict(x)
            indx = np.argmax(np.mean(predictions, axis=0))
            y[i, indx] = 1.
        return y