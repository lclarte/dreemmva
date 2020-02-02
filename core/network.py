# network.py
# Fichier pour générer les réseaux à l'aide de keras

import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, Dropout, Flatten, Reshape

class NetworkFactory:
    """
    For all networks, the input shape is NHWC with C = 1 (one channel)
    """
    def __init__(self, *, h=7, w=500, c=1):
        self.h, self.w, self.c = h, w, c
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
        super().__init__(h=7, w=250, c=1)
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
        super().__init__(h=7, w=250, c=1)
        self.nfilters1  = 100
        self.nfilters2  = 300
        self.dense_shape = self.h*self.w*self.c

    def get_network(self):
        model = keras.models.Sequential()
        model.add(Conv2D(self.nfilters1, kernel_size=(3, 3), activation=self.activation, input_shape=self.input_shape, data_format='channels_last'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Conv2D(self.nfilters1, kernel_size=(2, 3), activation=self.activation))
        model.add(MaxPooling2D(pool_size=(1, 2)))
        model.add(Dropout(0.5))
        model.add(Conv2D(self.nfilters2, kernel_size=(1, 7), activation=self.activation))
        model.add(MaxPooling2D(pool_size=(1, 2)))
        model.add(Dropout(0.25))
        model.add(Conv2D(self.nfilters1, kernel_size=(1, 3), activation=self.activation))
        model.add(Conv2D(self.nfilters1, kernel_size=(1, 3), activation=self.activation))
        model.add(Flatten())
        model.add(Dense(self.dense_shape, activation=self.activation))
        # output = vecteur de probas pour chaque classe
        model.add(Dense(1, activation='softmax'))
        return model