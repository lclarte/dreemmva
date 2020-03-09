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

x_train_file = 'data/x_train.h5'
x_test_file = 'data/x_test.h5'
y_train_file = 'data/y_train.csv'

# NE PAS UTILISER CE FICHIER, UTILISER PREFERABLEMENT LES NOTEBOOKS JUPYTER 