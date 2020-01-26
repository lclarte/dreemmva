from data import *
from network import *

from sklearn import model_selection

def main():
	# test that we can load data
	X = np.array(load_x('data/X_train.h5'))
	Y = np.array(load_y('data/y_train.csv'))
	X, Y = flatten_data(X, Y)
	reorder_nhwc(X)

	# load network and compile it 
	model = BaseNetworkFactory().get_network()
	print(model.summary())
	input("Press a key to continue ... ")
	
	model.compile(loss='binary_crossentropy', optimizer='adamax', metrics=['accuracy'])

	size = 1000
	X, Y = X[:size], Y[:size]
	model.fit(X, Y, epochs=50, batch_size=32, validation_split=0.25)

if __name__=='__main__':
	main()