# network.py
# Fichier pour générer les réseaux à l'aide de keras

from tensorflow import keras

class NetworkFactory:
    def __init__(self):
        pass

    def get_network(self):
        pass

class BaseNetworkFactory(NetworkFactory):
    def __init__(self):
        pass

    def get_network(self):
        model = keras.models.Sequential()
        return model

fac = BaseNetworkFactory()
model = fac.get_network()
print(model)