from data import *
from network import *

def main():
	# test that we can load data
	x_train = np.array(load_x('data/X_train.h5'))
	y_train = np.array(load_y('data/y_train.csv'))
	x_train, y_train = flatten_data(x_train, y_train)
	reorder_nhwc(x_train)
	print(x_train.shape, y_train.shape)
	input()

	# load network and compile it 
	model = BaseNetworkFactory().get_network()
	print(model.summary())
	model.compile(loss='binary_crossentropy', optimizer='adamax', metrics=['accuracy'])

	model.fit(x_train, y_train, epochs=10, batch_size=32)

if __name__=='__main__':
	main()