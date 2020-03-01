import braindecode
import torch
import numpy as np
from sklearn.model_selection import train_test_split

# load nn model from braindecode
from braindecode.models.shallow_fbcsp import ShallowFBCSPNet
from braindecode.models.deep4 import Deep4Net
from torch import nn
from braindecode.torch_ext.util import set_random_seeds
from braindecode.torch_ext.optimizers import AdamW
import torch.nn.functional as F

import core.data as data
import core.visualization as visu

def visualize_all_data():
        x_loaded, y_loaded = data.load_x('data/x_train.h5'), data.load_y('data/y_train.csv')
        # We have 946 subjects, 40 samples per subject and 7 channels per sample
        print(x_loaded.shape)
        S, N, h, w = x_loaded.shape
        print('This function will save ALL the EEG recordings in the folder images. Sure ? ')
        input()
        for i in range(S):
            for j in range(N):
                title = 'EEG recordings for subject ' + str(i+1) + ', sample number ' + str(j + 1)
                savefile = 'images/recording_' + str(i) + '_' + str(j) + '.png' 
                visu.save_eeg_sample(x_loaded[i, j], title, savefile)

def test_braindecode():
        # load data 
        x_loaded, y_loaded = data.load_x('data/x_train.h5'), data.load_y('data/y_train.csv')
        x, y = data.flatten_x(x_loaded), data.flatten_y(y_loaded, repeat=40)
        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.66)
        x_train, x_test = x_train.squeeze(), x_test.squeeze()

        # Only one value : 0 or 1  
        y_train, y_test = np.argmax(y_train, axis=1), np.argmax(y_test, axis=1)

        # get class weights
        weights = data.class_weights(y_train)

        cuda = False
        set_random_seeds(seed=20170629, cuda=cuda)
        n_classes = 2
        in_chans = x_train.shape[1]
        input_time_length=x_train.shape[2]
        final_conv_length='auto'
        # final_conv_length = auto ensures we only get a single output in the time dimension
        model = Deep4Net(in_chans=in_chans, n_classes=n_classes,
                                                        input_time_length=input_time_length,
                                                        final_conv_length=final_conv_length)

        #optimizer = AdamW(model.parameters(), lr=1*0.01, weight_decay=0.5*0.001) # these are good values for the deep model
        optimizer = AdamW(model.parameters(), lr=0.0625 * 0.01, weight_decay=0)
        criterion = lambda prediction, targets : F.nll_loss(prediction, targets, weight=torch.from_numpy(weights).float())
        model.compile(loss=criterion, optimizer=optimizer, iterator_seed=1,)

        model.fit(x_train, y_train, epochs=10, batch_size=64, scheduler='cosine'
         ,validation_data=(x_test, y_test)
         ,input_time_length = 450) # supercropsize for cropped training
        # Rk : here, 1 timestep = 1 / 250 seconds

        x_challenge = data.load_x('data/x_test.h5')
        x_challenge = data.flatten_x(x_challenge).squeeze()

        y_challenge = model.predict_classes(x_challenge)
        y_challenge2 = data.average_predictions(y_challenge)
        
        data.save_csv(y_challenge2, 'data/result.csv')

if __name__=='__main__':
        visualize_all_data()
        # test_braindecode()
