from core.data import *
from core.network import *
from core.visualization import *
from core.model import *

from sklearn import model_selection

def test_train():
	# test that we can load data
	X, Y = load_x('data/X_train.h5'), vectorize_y(load_y('data/y_train.csv'))
	X_test = load_x('data/X_test.h5')

	model = NNModel(config={
		'nn' : 'BaseNetworkFactory',
		'epoch' : 10
	})

	model.train(X, Y)

	y_pred = categorize_y(model.predict(X_test))

if __name__=='__main__':
	test_train()