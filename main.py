from braindecode.models.shallow_fbcsp import ShallowFBCSPNet
from braindecode.torch_ext.optimizers import AdamW
from core.data import *
import numpy as np
from sklearn import model_selection
import torch
import torch.nn.functional as F

train_x = 'data/X_train.h5'
train_y = 'data/y_train.csv'

def test_braindecode():
	x, y = load_x(train_x), load_y(train_y)
	y = flatten_y(y, 40)
	x = reorder_nhwc(flatten_x(x))
	x = np.squeeze(x, axis=3)
	y = np.squeeze(y, axis=1)

	x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.33)

	# on recupere juste le NN
	model = ShallowFBCSPNet(in_chans=7, n_classes=2,
                        input_time_length=500,
                        final_conv_length='auto')

	optimizer = AdamW(model.parameters(), lr=1*0.01, weight_decay=0.5*0.001)
	model.compile(loss=F.nll_loss, optimizer=optimizer, iterator_seed=1,)
	
	model.fit(x_train, y_train, epochs=50, batch_size=64, scheduler='cosine',validation_data=(x_test, y_test),)
	print(model.epochs_df)
	return model

if __name__=='__main__':
	test_braindecode()