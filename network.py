# network.py
# Fichier pour générer les réseaux à l'aide de keras

import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten

class NetworkFactory:
    def __init__(self):
        pass

    def get_network(self):
        pass

class BaseNetworkFactory(NetworkFactory):
    """
    Reseau propose dans l'article
    """
    def __init__(self):
        self.input_shape = (1, 7, 500)
        self.activation = 'relu'
        self.fsize      = 2
        self.nfilters1  = 300
        self.nfilters2  = 50     
        self.dense_shape = 7*500

    def get_network(self):
        model = keras.models.Sequential()
        model.add(Conv2D(self.nfilters1, self.fsize, activation=self.activation, input_shape=self.input_shape, data_format='channels_first'))
        model.add(MaxPooling2D())
        model.add(Dropout(0.5))
        model.add(Conv2D(self.nfilters2, self.fsize, activation=self.activation))
        model.add(MaxPooling2D())
        model.add(Dropout(0.5))
        model.add(Flatten())
        model.add(Dense(self.dense_shape, activation=self.activation))
        model.add(Dense(2, activation='softmax'))
        return model


fac = BaseNetworkFactory()
model = fac.get_network()
print(model.summary())