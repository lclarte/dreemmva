# network.py
# Fichier pour générer les réseaux à l'aide de keras

import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten

class NetworkFactory:
    def __init__(self):
        self.h, self.w, self.c = 7, 500, 1
        self.input_shape = (self.h, self.w, self.c)
        self.activation = 'relu'

    def get_network(self):
        raise NotImplementedError()

class SimpleNetworkFactory(NetworkFactory):
    """
    Reseau tres simple pour tester que le code tourne
    """
    def __init__(self):
        super().__init__()
        self.dense_shape = self.h*self.w*self.c
        self.output_shape = 2
    
    def get_network(self):
        model = keras.models.Sequential()
        model.add(Flatten(input_shape=self.input_shape))
        model.add(Dense(self.dense_shape, activation=self.activation))
        model.add(Dense(self.output_shape, activation='softmax'))
        return model

class BaseNetworkFactory(NetworkFactory):
    """
    Reseau propose dans l'article
    """
    def __init__(self):
        # need to put channel last otherwise there are bugs
        super().__init__()
        self.fsize      = 2
        self.nfilters1  = 300
        self.nfilters2  = 50     
        self.dense_shape = self.h*self.w*self.c

    def get_network(self):
        model = keras.models.Sequential()
        model.add(Conv2D(self.nfilters1, self.fsize, activation=self.activation, input_shape=self.input_shape, data_format='channels_last'))
        model.add(MaxPooling2D())
        model.add(Dropout(0.5))
        model.add(Conv2D(self.nfilters2, self.fsize, activation=self.activation))
        model.add(MaxPooling2D())
        model.add(Dropout(0.5))
        model.add(Flatten())
        model.add(Dense(self.dense_shape, activation=self.activation))
        model.add(Dense(2, activation='softmax'))
        return model