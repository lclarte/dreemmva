from core.data import *
from core.network import *
from core.visualization import *

from sklearn import model_selection

def test_train():
	# test that we can load data
	X, Y = load_all('data/X_train.h5', 'data/y_train.csv')

	X = fft_eeg(X)

	# load network and compile it 
	model = Conv1DNetworkFactory().get_network()
	print("Number of samples", X.shape[0])
	input("Press a key to continue ... ")
	
	model.compile(loss='binary_crossentropy', optimizer='adamax', metrics=['accuracy'])

	model.fit(X, Y, epochs=50, batch_size=32, validation_split=0.25)

if __name__=='__main__':
	test_train()