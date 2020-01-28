# model.py
# classe abstraite qui inclut les NN et autres modeles (modeles lineaires)

from core.network import BaseNetworkFactory
import core.data as data

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
        # define useful constants
        self.epoch            = 3
        self.batch_size       = 32
        self.validation_split = 0.25

        # initialize NN
        factory = BaseNetworkFactory()
        self.network = factory.get_network()
        self.network.compile(loss='binary_crossentropy', optimizer='adamax', metrics=['accuracy'])

    def train(self, x_train, y_train):
        """
        x_train.shape = (N, 40, 7, 500)
        y_train.shape = (N, 2)
        """
        x_train, y_train = data.flatten_data(x_train, y_train)
        x_train = data.reorder_nhwc(x_train)

        self.network.fit(x_train, y_train, epochs=self.epoch, batch_size=self.batch_size, validation_split=self.validation_split)
        return True

    def predict(self, x_test):
        """
        Retourne la prediction pour chacune des classes. Format : y \in {0, 1}
        """
        return self.network.predict(x_test)