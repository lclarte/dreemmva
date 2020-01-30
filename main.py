from core.data import *
from core.network import *
from core.visualization import *
from core.model import *

from sklearn import model_selection

def test_train():
	# test that we can load data
	X, Y = load_x('data/X_train.h5'), vectorize_y(load_y('data/y_train.csv'))
	X_test = load_x('data/X_test.h5')

	model = NNModel(None)

	model.train(X, Y)
	model.get_network().save('data/base_model.h5')

	y_pred = categorize_y(model.predict(X_test))
	save_y(y_pred, 'data/y_test.csv')

if __name__=='__main__':
	test_train()