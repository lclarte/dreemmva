from core.data import *
from core.network import *
from core.visualization import *

from sklearn import model_selection

def test_display():
	X = np.array(load_x('data/X_train.h5'))
	Y = vectorize_y(np.array(load_y('data/y_train.csv')))
	X2, Y2 = flatten_data(X, Y)

	samples = [X[1, 1], X2[41, 0]]

	plot_eeg_samples(samples)

def test_train():
	# test that we can load data
	X = np.array(load_x('data/X_train.h5'))
	Y = vectorize_y(np.array(load_y('data/y_train.csv')))
	X, Y = flatten_data(X, Y)
	reorder_nhwc(X)

	# load network and compile it 
	model = Conv1DNetworkFactory().get_network()
	print("Number of samples", X.shape[0])
	input("Press a key to continue ... ")
	
	model.compile(loss='binary_crossentropy', optimizer='adamax', metrics=['accuracy'])

	size = 10000
	X, Y = X[:size], Y[:size]
	model.fit(X, Y, epochs=50, batch_size=32, validation_split=0.25)

if __name__=='__main__':
	test_display()