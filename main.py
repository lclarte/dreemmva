from core.data import *
from core.network import *
from core.visualization import *
from core.model import *

from sklearn import model_selection

def test_train():
	# test that we can load data
	X, Y = load_all('data/X_train.h5', 'data/y_train.csv')
	X_test = load_x('data/X_train.h5')
	print('Shape of the data :')
	print("X.shape = ", X.shape)
	print("Y.shape = ", Y.shape)

	# for testing : what if we work on Fourier Transform ?
	X = fft_eeg(X)

	model = NNModel(None)
	X, Y = X[:50], Y[:50]

	model.train(X, Y)
	y_pred = model.predict(X)
	print("Accuracy = ", compare_predict(y_pred, Y))

if __name__=='__main__':
	test_train()