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

from utils import GradCAM, show_cam_on_image

cudnn.benchmark = False
cudnn.deterministic = True


train_data = []
train_label = []
eeg_data_path = '/home/Data/Things-EEG2/Preprocessed_data_250Hz/'
nSub = 8
train_data = np.load(eeg_data_path + '/sub-' + format(nSub, '02') + '/preprocessed_eeg_training.npy', allow_pickle=True)
train_data = train_data['preprocessed_eeg_data']
train_data = np.mean(train_data, axis=1)
train_data = np.expand_dims(train_data, axis=1)

data = train_data
print(np.shape(data))


# ! A crucial step for adaptation on Transformer
# reshape_transform  b 61 40 -> b 40 1 61
def reshape_transform(tensor):
    result = rearrange(tensor, 'b (h w) e -> b e (h) (w)', h=1)
    return result


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = Enc_eeg()


model.load_state_dict(torch.load('./model/cat_saEnc_eeg_cls.pth', map_location=device))
target_layers = [model[0]]  # set the target layer 
cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)



# TODO: Class Activation Topography (proposed in the paper)
import mne
from matplotlib import mlab as mlab

easycapm1_montage = mne.channels.make_standard_montage('easycap-M1')
ch_name = ['Fp1', 'Fp2', 'AF7', 'AF3', 'AFz', 'AF4', 'AF8', 'F7', 'F5', 'F3',
			'F1', 'F2', 'F4', 'F6', 'F8', 'FT9', 'FT7', 'FC5', 'FC3', 'FC1', 
			'FCz', 'FC2', 'FC4', 'FC6', 'FT8', 'FT10', 'T7', 'C5', 'C3', 'C1',
			'Cz', 'C2', 'C4', 'C6', 'T8', 'TP9', 'TP7', 'CP5', 'CP3', 'CP1', 
			'CPz', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10', 'P7', 'P5', 'P3', 'P1',
			'Pz', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO3', 'POz', 'PO4', 'PO8',
			'O1', 'Oz', 'O2']
info = mne.create_info(ch_names=ch_name, sfreq=250., ch_types='eeg')


all_cam = []
# this loop is used to obtain the cam of each trial/sample
for i in range(np.shape(data)[0]):
    test = torch.as_tensor(data[i:i+1, :, :, :], dtype=torch.float32)
    test = torch.autograd.Variable(test, requires_grad=True)

    grayscale_cam = cam(input_tensor=test)
    grayscale_cam = grayscale_cam[0, :]
    all_cam.append(grayscale_cam)


# the mean of all data
test_all_data = np.squeeze(np.mean(data, axis=0))
test_all_data = (test_all_data - np.mean(test_all_data)) / np.std(test_all_data)
mean_all_test = np.mean(test_all_data, axis=1)

# the mean of all cam
test_all_cam = np.mean(all_cam, axis=0)
test_all_cam = (test_all_cam - np.mean(test_all_cam)) / np.std(test_all_cam)
mean_all_cam = np.mean(test_all_cam, axis=1)

# apply cam on the input data
# hyb_all = test_all_data * test_all_cam

hyb_all = np.einsum('b c h w, b h w -> b h w', data, all_cam)
hyb_all = np.mean(hyb_all, axis=0)

hyb_all = (hyb_all - np.mean(hyb_all)) / np.std(hyb_all)
mean_hyb_all = np.mean(hyb_all, axis=1)

evoked = mne.EvokedArray(test_all_data, info)
evoked.set_montage(easycapm1_montage)

fig, [ax1, ax2, ax3] = plt.subplots(nrows=3)

# print(mean_all_test)
plt.subplot(311)
im1, cn1 = mne.viz.plot_topomap(mean_all_test, evoked.info, show=False, axes=ax1, res=1200)

plt.subplot(312)
im2, cn2 = mne.viz.plot_topomap(mean_all_cam, evoked.info, show=False, axes=ax2, res=1200)

plt.subplot(313)
im3, cn3 = mne.viz.plot_topomap(mean_hyb_all, evoked.info, show=False, axes=ax3, res=1200)


