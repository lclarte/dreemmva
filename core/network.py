# network.py
# Fichier pour générer les réseaux à l'aide de keras

import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, Dropout, Flatten, Reshape

class NetworkFactory:
    """
    For all networks, the input shape is NHWC with C = 1 (one channel)
    """
    def __init__(self):
        self.h, self.w, self.c = 7, 500, 1
        self.input_shape = (self.h, self.w, self.c)
        self.activation = 'relu'

    def get_network(self):
        raise NotImplementedError()

    def format_data(self, x, y):
        raise NotImplementedError

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
    Reseau propose dans l'article, un peu modifie pour prendre en compte les differences de taille
    entre les inputs
    """
    def __init__(self):
        # need to put channel last otherwise there are bugs
        super().__init__()
        self.fsize      = 2
        self.nfilters1  = 100
        self.nfilters2  = 10    
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

class Conv1DNetworkFactory(NetworkFactory):
    def __init__(self):
        super().__init__()
        self.fsize = [2, 2, 2, 2, 2, 2, 2]
        self.nfilters = [300, 300, 200, 200, 100, 100, 100]
        self.dense_shape = self.h * self.w * self.c

    def get_network(self):
        model = keras.models.Sequential()
        # reshape to remove channel dim, permute h and w because we have channel last
        model.add(Reshape((self.w, self.h), input_shape=self.input_shape))
        for i in range(0, len(self.fsize)):
            model.add(Conv1D(self.nfilters[i], self.fsize[i], activation=self.activation, data_format='channels_last'))
            model.add(MaxPooling1D(2))
            model.add(Dropout(0.5))
        model.add(Flatten())
        model.add(Dense(self.dense_shape, activation=self.activation))
        model.add(Dense(2, activation='softmax'))
        return model