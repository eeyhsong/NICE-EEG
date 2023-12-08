import os
import argparse
import math
import glob
import random
import itertools
import datetime
import time
import sys
import warnings
import scipy.io
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch import Tensor

import torch.autograd as autograd 
from torch.autograd import Variable
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.backends import cudnn

import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.utils import save_image, make_grid
from torchvision.transforms import Compose, Resize, ToTensor

from torchsummary import summary

from PIL import Image
import matplotlib.pyplot as plt


from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce


# hugging face
from transformers import AutoImageProcessor, ViTForImageClassification, ViTFeatureExtractor, ViTModel
from datasets import load_dataset

gpus = [9]
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, gpus))

import mne
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from matplotlib import mlab as mlab
import matplotlib.animation as animation
import numpy as np



train_data = []
train_label = []

# sub 1 7 10 is good!
nSub = 1
eeg_data_path = '/home/Data/Things-EEG2/Preprocessed_data_250Hz/'

train_data = np.load(eeg_data_path + '/sub-' + format(nSub, '02') + '/preprocessed_eeg_training.npy', allow_pickle=True)
train_data = train_data['preprocessed_eeg_data']
# train_data = train_data[:, 0:4, :, :]
train_data = np.mean(train_data, axis=1)
train_data = np.expand_dims(train_data, axis=1)

test_data = np.load(eeg_data_path + '/sub-' + format(nSub, '02') + '/preprocessed_eeg_test.npy', allow_pickle=True)
test_data = test_data['preprocessed_eeg_data']
test_data = np.mean(test_data, axis=1)
test_data = np.expand_dims(test_data, axis=1)

data = np.mean(train_data, 0)
dd = data[0]

# data = np.delete(data, [32, 42, 53, 57, 59, 63], 1) # delete PO5 and PO7
# dd = np.mean(data, axis=0)
# # dd = np.mean(dd, axis=1)


# biosemi_montage = mne.channels.make_standard_montage('biosemi64')
easycapm1_montage = mne.channels.make_standard_montage('easycap-M1')
ch_name = ['Fp1', 'Fp2', 'AF7', 'AF3', 'AFz', 'AF4', 'AF8', 'F7', 'F5', 'F3',
			'F1', 'F2', 'F4', 'F6', 'F8', 'FT9', 'FT7', 'FC5', 'FC3', 'FC1', 
			'FCz', 'FC2', 'FC4', 'FC6', 'FT8', 'FT10', 'T7', 'C5', 'C3', 'C1',
			'Cz', 'C2', 'C4', 'C6', 'T8', 'TP9', 'TP7', 'CP5', 'CP3', 'CP1', 
			'CPz', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10', 'P7', 'P5', 'P3', 'P1',
			'Pz', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO3', 'POz', 'PO4', 'PO8',
			'O1', 'Oz', 'O2']

info = mne.create_info(ch_names=ch_name, sfreq=1000., ch_types='eeg')
evoked = mne.EvokedArray(dd, info)
evoked.set_montage(easycapm1_montage)

# plt.figure(1)
# tmp_data = evoked.data[:, :250]
# topo_data = np.mean(tmp_data, axis=1)
# # topo_data = (topo_data - np.mean(topo_data)) / np.std(topo_data)
# topo_data = 2 * (topo_data - np.min(topo_data)) / (np.max(topo_data) - np.min(topo_data)) - 1
# # mne.viz.plot_topomap(topo_data, evoked.info, show=False)
# im, cn = mne.viz.plot_topomap(topo_data, evoked.info, show=False, cmap='coolwarm', sensors=False, res=600, vlim=(-1, 1))
# plt.colorbar(im)
# plt.savefig('./pic/Conf/topo.png', dpi=300)

# # * calculate power
# for i in range(63):
#     for j in range(250):
#         evoked.data[i, j] = evoked.data[i, j] * evoked.data[i, j]
# * ten 
fig, axs = plt.subplots(nrows=1, ncols=10, figsize=(20, 3))
for i in range(10):
    tmp_data = evoked.data[:, i*25:(i+1)*25]
    topo_data = np.mean(tmp_data, axis=1)
    topo_data = (topo_data - np.mean(topo_data)) / np.std(topo_data)
    # topo_data = 2 * (topo_data - np.min(topo_data)) / (np.max(topo_data) - np.min(topo_data)) - 1
    im, cn = mne.viz.plot_topomap(topo_data, evoked.info, show=False, cmap='coolwarm', sensors=False, res=600, vlim=(-1, 1), axes=axs[i])
    axs[i].set_title('%d-%d ms' % (i*100, (i+1)*100))
cax = fig.add_axes([0.92, 0.33, 0.005, 0.4])
fig.colorbar(im, cax=cax)
plt.savefig('./pic/topo_ten.svg', dpi=300)

print('the end')