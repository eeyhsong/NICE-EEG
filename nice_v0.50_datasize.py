"""
Try to perform object recognition

with Things-EEG2 dataset

image retriveal (using the sitmuli image)

image classification (using other images of the same category to construct the contrast center)

use validation set - 1674 images - just use the loss

! record test accuracy using the checkpoint of lowest validation loss in all epochs

use 250 Hz data

Baseline
"""


import os
import argparse
# gpus = [2, 7]
# os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, gpus))
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

gpus = [7]
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, gpus))

average_num = 80
result_path = '/home/songyonghao/Documents/Code/NICE/results/Conf/datasize/repeat_test/repeat%d/'%average_num 
model_idx = 'rp80'
 
parser = argparse.ArgumentParser(description='Experiment Stimuli Recognition test with CLIP encoder')
# parser.add_argument('data', metavar='DIR', nargs='?', default='/home/songyonghao/Documents/Data/IE/stimuli',
#                     help='path to dataset (default: imagenet)')
parser.add_argument('--dnn', default='clip', type=str)
parser.add_argument('--epoch', default='200', type=int)
parser.add_argument('--num_sub', default=10, type=int,
                    help='number of subjects used in the experiments. ')
parser.add_argument('-batch_size', '--batch-size', default=1000, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--seed', default=2023, type=int,
                    help='seed for initializing training. ')


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


class PatchEmbedding(nn.Module):
    def __init__(self, emb_size=40):
        # self.patch_size = patch_size
        super().__init__()
        self.shallownet = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), (1, 1)),
            nn.AvgPool2d((1, 51), (1, 5)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Conv2d(40, 40, (63, 1), (1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Dropout(0.5),
        )

        self.projection = nn.Sequential(
            nn.Conv2d(40, emb_size, (1, 1), stride=(1, 1)),  # 5 is better than 1
            Rearrange('b e (h) (w) -> b (h w) e'),
        )

    def forward(self, x: Tensor) -> Tensor:
        b, _, _, _ = x.shape
        x = self.shallownet(x)
        x = self.projection(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)  # batch, num_heads, query_len, key_len
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )

class GELU(nn.Module):
    def forward(self, input: Tensor) -> Tensor:
        return input*0.5*(1.0+torch.erf(input/math.sqrt(2.0)))

class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size,
                 num_heads=10,
                 drop_p=0.5,
                 forward_expansion=4,
                 forward_drop_p=0.5):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, num_heads, drop_p),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))


class TransformerEncoder(nn.Sequential):
    def __init__(self, depth, emb_size):
        super().__init__(*[TransformerEncoderBlock(emb_size) for _ in range(depth)])


class FlattenHead(nn.Sequential):
    def __init__(self, emb_size, n_classes):
        super().__init__()
        # self.fc = nn.Sequential(
        #     nn.Linear(2440, 256),
        #     nn.ELU(),
        #     nn.Dropout(0.5),
        #     nn.Linear(256, 40),
        # )

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)
        # cls_out = self.fc(x) 
        return x


class ClassificationHead(nn.Sequential):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            # nn.Linear(2440, 256),
            # nn.ELU(),
            # nn.Dropout(0.5),
            nn.Linear(256, 40),
        )

    def forward(self, x):
        # x = x.contiguous().view(x.size(0), -1)
        cls_out = self.fc(x) 
        return x, cls_out   


class Enc_eeg(nn.Sequential):
    def __init__(self, emb_size=40, depth=3, n_classes=4, **kwargs):
        super().__init__(
            PatchEmbedding(emb_size),
            # TransformerEncoder(depth, emb_size),
            FlattenHead(emb_size, n_classes)
        )

        
# class Enc_img(nn.Sequential):
#     def __init__(self, emb_size=40, depth=6, n_classes=4, **kwargs):
#         super().__init__(
#             PatchEmbedding(emb_size),
#             # TransformerEncoder(depth, emb_size),
#             ClassificationHead(emb_size, n_classes)
#         )

        
class Proj_eeg(nn.Sequential):
    def __init__(self, embedding_dim=1440, proj_dim=768, drop_proj=0.5):
        super().__init__(
            nn.Linear(embedding_dim, proj_dim),
            ResidualAdd(nn.Sequential(
                nn.GELU(),
                nn.Linear(proj_dim, proj_dim),
                nn.Dropout(drop_proj),
            )),
            nn.LayerNorm(proj_dim),
            # ClassificationHead()
        )


class Proj_img(nn.Sequential):
    def __init__(self, embedding_dim=768, proj_dim=768, drop_proj=0.3):
        super().__init__(
            nn.Linear(embedding_dim, proj_dim),
            ResidualAdd(nn.Sequential(
                nn.GELU(),
                nn.Linear(proj_dim, proj_dim),
                nn.Dropout(drop_proj),
            )),
            nn.LayerNorm(proj_dim),
            # ClassificationHead()
        )
    def forward(self, x):

        return x 


# Image2EEG
class IE():
    def __init__(self, args, nsub):
        super(IE, self).__init__()
        self.args = args
        self.num_class = 200
        self.batch_size = args.batch_size
        self.batch_size_test = 200
        self.batch_size_img = 500 
        self.n_epochs = args.epoch

        self.lambda_cen = 0.003
        self.alpha = 0.5

        self.proj_dim = 256

        self.lr = 0.0002
        self.b1 = 0.5
        self.b2 = 0.999
        self.nSub = nsub

        self.start_epoch = 0
        self.eeg_data_path = '/home/songyonghao/Documents/Data/Things-EEG2/Preprocessed_data_250Hz/'
        self.img_data_path = '/home/songyonghao/Documents/Data/Things-EEG2/DNN_feature_maps/pca_feature_maps/' + args.dnn + '/pretrained-True/'
        self.test_center_path = '/home/songyonghao/Documents/Data/Things-EEG2/Image_set/'
        # self.label_path = '/home/songyonghao/Documents/Data/IE/Label/exp/'
        # self.order_path = './order/' # used to make corrsponding order of eeg and img

        self.pretrain = False

        self.log_write = open(result_path + "log_subject%d.txt" % self.nSub, "w")

        # self.img_shape = (self.channels, self.img_height, self.img_width)

        self.Tensor = torch.cuda.FloatTensor
        self.LongTensor = torch.cuda.LongTensor

        self.criterion_l1 = torch.nn.L1Loss().cuda()
        self.criterion_l2 = torch.nn.MSELoss().cuda()
        self.criterion_cls = torch.nn.CrossEntropyLoss().cuda()

        # self.model = CLIP().cuda()
        # self.model = nn.DataParallel(self.model, device_ids=[i for i in range(len(gpus))])
        # self.model = self.model.cuda()
        self.Enc_eeg = Enc_eeg().cuda()
        self.Proj_eeg = Proj_eeg().cuda()
        self.Proj_img = Proj_img().cuda()
        self.Enc_eeg = nn.DataParallel(self.Enc_eeg, device_ids=[i for i in range(len(gpus))])
        self.Proj_eeg = nn.DataParallel(self.Proj_eeg, device_ids=[i for i in range(len(gpus))])
        self.Proj_img = nn.DataParallel(self.Proj_img, device_ids=[i for i in range(len(gpus))])

        # # pre-training visual recognition model
        # vit_model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
        # self.Enc_img = vit_model.vit.cuda()
        # self.Enc_img = nn.DataParallel(self.Enc_img, device_ids=[i for i in range(len(gpus))])

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.centers = {}

        print('initial define done.')

        # summary(self.model, (1, 22, 1000))

        # self.centers = {}


    def get_eeg_data(self):
        train_data = []
        train_label = []
        test_data = []
        test_label = np.arange(200)

        # train_data = np.load(self.eeg_data_path + 'sub-' + format(self.nSub, '02') + '_training.npy', allow_pickle=True)
        # train_data = train_data['preprocessed_eeg_data']
        # train_data = np.expand_dims(train_data, axis=1)

        # test_data = np.load(self.eeg_data_path + 'sub-' + format(self.nSub, '02') + '_test.npy', allow_pickle=True)
        # test_data = test_data['preprocessed_eeg_data']
        # test_data = np.expand_dims(test_data, axis=1)

        train_data = np.load(self.eeg_data_path + '/sub-' + format(self.nSub, '02') + '/preprocessed_eeg_training.npy', allow_pickle=True)
        train_data = train_data['preprocessed_eeg_data']
        # train_data = train_data[:, 0:4, :, :]
        train_data = np.mean(train_data, axis=1)
        train_data = np.expand_dims(train_data, axis=1)

        test_data = np.load(self.eeg_data_path + '/sub-' + format(self.nSub, '02') + '/preprocessed_eeg_test.npy', allow_pickle=True)
        test_data = test_data['preprocessed_eeg_data']
        # test_data = np.mean(test_data, axis=1)

        # average_num = 80
        times = test_data.shape[1] // average_num

        tmp_test_data = np.zeros((test_data.shape[0], times, test_data.shape[2], test_data.shape[3]))
        for i in range(times):
            tmp_test_data[:, i, :, :] = np.mean(test_data[:, i * average_num:(i + 1) * average_num, :, :], axis=1)

        test_data = tmp_test_data
        test_data = rearrange(test_data, 'n c h w -> (c n) h w')
        test_data = np.expand_dims(test_data, axis=1)
        test_label = repeat(test_label, 'n -> (c n)', c=times)

        return train_data, train_label, test_data, test_label


    def get_image_data(self):
        train_img_feature = np.load(self.img_data_path + self.args.dnn + '_feature_maps_training.npy', allow_pickle=True)
        test_img_feature = np.load(self.img_data_path + self.args.dnn + '_feature_maps_test.npy', allow_pickle=True)

        train_img_feature = np.squeeze(train_img_feature)
        test_img_feature = np.squeeze(test_img_feature)

        return train_img_feature, test_img_feature
    
    # TODO try shuffle in 10 condition and 4 repetition?
    # def pair_shuffle(self, data, label):
    #     shuffle_idx = np.random.permutation(len(data))
    #     data = data[shuffle_idx]
    #     label = label[shuffle_idx]
    #     return data, label


    def select_image(self, image, label):
        # select 90 images for each class
        select_image = []
        for i in range(self.num_class):
            catgory_idx = np.where(label == i)
            select_idx = np.random.choice(catgory_idx[0], 180, replace=False)
            select_image.append(image[select_idx])
        select_image = torch.cat(select_image)
        return select_image


    def interaug(self, eeg, label): # inter-class augmentation
        # adding augmentation data for each class
        # the label is not ascending order  
        # label: (3600)
        aug_eeg = []
        aug_label = []


        for i in range(self.num_class):
            catgory_idx = np.where(label == label[90*(i+1)-1])
            one_class_eeg = eeg[catgory_idx]
            one_class_label = label[catgory_idx]
            aug_eeg.append(one_class_eeg)
            aug_label.append(one_class_label)

            tmp_aug_data = np.zeros(np.shape(one_class_eeg))
            for augi in range(len(tmp_aug_data)):
                for augj in range(8):
                    aug_radm_idx = np.random.randint(0, len(one_class_eeg), 8)
                    tmp_aug_data[augi, :, :, augj*125:(augj+1)*125] = one_class_eeg[aug_radm_idx[augj], :, :, augj*125:(augj+1)*125]
            aug_eeg.append(tmp_aug_data)
            aug_label.append(one_class_label)

        aug_eeg = np.concatenate(aug_eeg)
        aug_label = np.concatenate(aug_label)

        return aug_eeg, aug_label

    def update_centers(self, feature, label):
        deltac = {}
        count = {}
        count[0] = 0
        for i in range(len(label)):
            l = label[i]
            if l in deltac:
                deltac[l] += self.centers[l]-feature[i]
            else:
                deltac[l] = self.centers[l]-feature[i]
            if l in count:
                count[l] += 1
            else:
                count[l] = 1

        for ke in deltac.keys():
            deltac[ke] = deltac[ke]/(count[ke]+1)

        return deltac
        
    def update_lr(self, optimizer, lr):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


    def train(self):
        
        self.Enc_eeg.apply(weights_init_normal)
        self.Proj_eeg.apply(weights_init_normal)
        self.Proj_img.apply(weights_init_normal)

        train_eeg, _, test_eeg, test_label = self.get_eeg_data()
        train_img_feature, _ = self.get_image_data() 
        test_center = np.load(self.test_center_path + 'center_' + self.args.dnn + '.npy', allow_pickle=True)

        # shuffle the training data
        train_shuffle = np.random.permutation(len(train_eeg))
        train_eeg = train_eeg[train_shuffle]
        train_img_feature = train_img_feature[train_shuffle]

        # train_eeg = train_eeg[:4135*4]
        # train_img_feature = train_img_feature[:4135*4]

        val_eeg = torch.from_numpy(train_eeg[:740])
        val_image = torch.from_numpy(train_img_feature[:740])

        train_eeg = torch.from_numpy(train_eeg[740:])
        train_image = torch.from_numpy(train_img_feature[740:])



        dataset = torch.utils.data.TensorDataset(train_eeg, train_image)
        self.dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=True)
        val_dataset = torch.utils.data.TensorDataset(val_eeg, val_image)
        self.val_dataloader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=self.batch_size, shuffle=False)

        test_eeg = torch.from_numpy(test_eeg)
        # test_img_feature = torch.from_numpy(test_img_feature)
        test_center = torch.from_numpy(test_center)
        test_label = torch.from_numpy(test_label)
        test_dataset = torch.utils.data.TensorDataset(test_eeg, test_label)
        self.test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=self.batch_size_test, shuffle=False)

        # test_img_label = torch.from_numpy(test_img_label)
        # img_fea_dataset = torch.utils.data.TensorDataset(test_img_feature, test_img_label)
        # self.img_fea_dataloader = torch.utils.data.DataLoader(dataset=img_fea_dataset, batch_size=self.batch_size_img, shuffle=False)

        # for i in range(self.num_class):
        #     self.centers[i] = torch.randn(self.proj_dim)
        #     self.centers[i] = self.centers[i].cuda()

        # Optimizers
        self.optimizer = torch.optim.Adam(itertools.chain(self.Enc_eeg.parameters(), self.Proj_eeg.parameters(), self.Proj_img.parameters()), lr=self.lr, betas=(self.b1, self.b2))
        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(self.b1, self.b2))

        # test_data = Variable(test_data.type(self.Tensor))
        # test_label = Variable(test_label.type(self.LongTensor))

        num = 0
        best_loss_val = np.inf

        # Train the cnn model
        # total_step = len(self.dataloader)
        # curr_lr = self.lr

        for e in range(self.n_epochs):
            in_epoch = time.time()

            # select_image_feature = self.select_image(train_img_feature, train_img_label)
            # aug_train_eeg, aug_train_label = self.interaug(train_eeg, train_label)
            # aug_train_eeg = torch.from_numpy(aug_train_eeg)
            # aug_train_label = torch.from_numpy(aug_train_label)

            # dataset = torch.utils.data.TensorDataset(aug_train_eeg, select_image_feature, aug_train_label)
            # self.dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=True)

            self.Enc_eeg.train()
            self.Proj_eeg.train()
            self.Proj_img.train()

            # starttime_epoch = datetime.datetime.now()

            for i, (eeg, img) in enumerate(self.dataloader):

                eeg = Variable(eeg.cuda().type(self.Tensor))
                # img = Variable(img.cuda().type(self.Tensor))
                img_features = Variable(img.cuda().type(self.Tensor))
                # label = Variable(label.cuda().type(self.LongTensor))
                labels = torch.arange(eeg.shape[0])  # used for the loss
                labels = Variable(labels.cuda().type(self.LongTensor))

                # obtain the features
                eeg_features = self.Enc_eeg(eeg)
                # img_features = self.Enc_img(img).last_hidden_state[:,0,:]

                # project the features to a multimodal embedding space
                eeg_features = self.Proj_eeg(eeg_features)
                img_features = self.Proj_img(img_features)

                # normalize the features
                eeg_features = eeg_features / eeg_features.norm(dim=1, keepdim=True)
                img_features = img_features / img_features.norm(dim=1, keepdim=True)

                # L2 loss
                # loss_l2 = self.criterion_l2(eeg_features, img_features)

                # cosine similarity as the logits
                logit_scale = self.logit_scale.exp()
                logits_per_eeg = logit_scale * eeg_features @ img_features.t()
                logits_per_img = logits_per_eeg.t()

                loss_eeg = self.criterion_cls(logits_per_eeg, labels)
                loss_img = self.criterion_cls(logits_per_img, labels)

                loss_cos = (loss_eeg + loss_img) / 2

                # # classification loss
                # loss_cls_eeg = self.criterion_cls(eeg_cls, label)
                # loss_cls_img = self.criterion_cls(img_cls, label)
                # loss_cls = (loss_cls_eeg + loss_cls_img) / 2

                # Center loss
                # cen_feature_st = torch.cat((img_features, eeg_features), axis=0)  # source and target
                # cen_label_st = torch.cat((label, label))

                # cen_feature = eeg_features
                # cen_label = label
                # nplabela = cen_label.cpu().numpy()

                # loss_cen = 0
                # for k in range(len(cen_label)):
                #     la = nplabela[k]
                #     if k == 0:
                #         loss_cen = self.criterion_l2(self.centers[la], cen_feature[k])
                #     else:
                #         loss_cen += self.criterion_l2(self.centers[la], cen_feature[k])


                # total loss
                loss = loss_cos

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # # update centers
                # deltacA = self.update_centers(cen_feature, cen_label.cpu().numpy())
                # with torch.no_grad():
                #     for ke in deltacA.keys():
                #         self.centers[ke] = self.centers[ke] - self.alpha * deltacA[ke]

            # endtime_epoch = datetime.datetime.now()
            # print('train epoch %d duration: '%(i+1) + str(endtime_epoch - starttime_epoch))
            # starttime_epoch = datetime.datetime.now()


            if (e + 1) % 1 == 0:
                # with torch.zero_grad(): # that way can save the calculation cost
                self.Enc_eeg.eval()
                self.Proj_eeg.eval()
                self.Proj_img.eval()
                with torch.no_grad():
                    # * validation part
                    for i, (veeg, vimg) in enumerate(self.val_dataloader):

                        veeg = Variable(veeg.cuda().type(self.Tensor))
                        vimg_features = Variable(vimg.cuda().type(self.Tensor))
                        vlabels = torch.arange(veeg.shape[0])
                        vlabels = Variable(vlabels.cuda().type(self.LongTensor))

                        veeg_features = self.Enc_eeg(veeg)
                        veeg_features = self.Proj_eeg(veeg_features)
                        vimg_features = self.Proj_img(vimg_features)

                        veeg_features = veeg_features / veeg_features.norm(dim=1, keepdim=True)
                        vimg_features = vimg_features / vimg_features.norm(dim=1, keepdim=True)

                        logit_scale = self.logit_scale.exp()
                        vlogits_per_eeg = logit_scale * veeg_features @ vimg_features.t()
                        vlogits_per_img = vlogits_per_eeg.t()

                        vloss_eeg = self.criterion_cls(vlogits_per_eeg, vlabels)
                        vloss_img = self.criterion_cls(vlogits_per_img, vlabels)

                        vloss = (vloss_eeg + vloss_img) / 2

                        if vloss <= best_loss_val:
                            best_loss_val = vloss
                            best_epoch = e + 1
                            torch.save(self.Enc_eeg.module.state_dict(), './model/' + model_idx + 'Enc_eeg_cls.pth')
                            torch.save(self.Proj_eeg.module.state_dict(), './model/' + model_idx + 'Proj_eeg_cls.pth')
                            torch.save(self.Proj_img.module.state_dict(), './model/' + model_idx + 'Proj_img_cls.pth')

                            # # * test part
                    
                            # # # TODO Think Think Think!!!
                            # all_center = test_center

                            # # test process
                            # total = 0
                            # top1 = 0
                            # top3 = 0
                            # top5 = 0

                            # for i, (teeg, tlabel) in enumerate(self.test_dataloader):
                            #     teeg = Variable(teeg.type(self.Tensor))
                            #     tlabel = Variable(tlabel.type(self.LongTensor))
                            #     all_center = Variable(all_center.type(self.Tensor))       

                            #     tfea = self.Proj_eeg(self.Enc_eeg(teeg))
                            #     tcen = self.Proj_img(all_center)
                            #     tfea = tfea / tfea.norm(dim=1, keepdim=True)
                            #     tcen = tcen / tcen.norm(dim=1, keepdim=True)

                            #     logit_scale = self.logit_scale.exp()
                            #     tlogits_per_eeg  = logit_scale * tfea @ tcen.t()
                            #     tlogits_per_img = tlogits_per_eeg.t()

                            #     tloss_eeg = self.criterion_cls(tlogits_per_eeg, tlabel)
                            #     tloss_img = self.criterion_cls(tlogits_per_img, tlabel)

                            #     tloss = (tloss_eeg + tloss_img) / 2

                            #     similarity = (100.0 * tfea @ all_center.t()).softmax(dim=-1)  # no use 100?
                            #     _, indices = similarity.topk(5)

                            #     tt_label = tlabel.view(-1, 1)
                            #     # y_pred = torch.max(Cls, 1)[1]
                            #     total += tlabel.size(0)
                            #     top1 += (tt_label == indices[:, :1]).sum().item()
                            #     top3 += (tt_label == indices[:, :3]).sum().item()
                            #     top5 += (tt_label == indices).sum().item()



                            # top1_acc = float(top1) / float(test_label.size(0))
                            # top3_acc = float(top3) / float(test_label.size(0))
                            # top5_acc = float(top5) / float(test_label.size(0))

                # print('The epoch is:', e, '  The accuracy is:', acc)
                print('Epoch:', e,
                      '  Cos eeg: %.4f' % loss_eeg.detach().cpu().numpy(),
                      '  Cos img: %.4f' % loss_img.detach().cpu().numpy(),
                      '  loss val: %.4f' % vloss.detach().cpu().numpy(),
                    #   '  loss test: %.4f' % tloss.detach().cpu().numpy(),
                    #   '  L2: %.6f' % loss_l2.detach().cpu().numpy(),
                    #   '  Cls eeg: %.6f' % loss_cls_eeg.detach().cpu().numpy(),
                    #   '  Cls img: %.6f' % loss_cls_img.detach().cpu().numpy(),
                    #   '  Cen: %.6f' % (self.lambda_cen / 2 * loss_cen).detach().cpu().numpy(),
                    #   '  Top1 %.4f' % top1_acc,
                    #   '  Top3 %.4f' % top3_acc,
                    #   '  Top5 %.4f' % top5_acc
                      )
                # self.log_write.write('Epoch %d: Top1-%.6f, Top3-%.6f, Top5-%.6f \n' % (e, top1_acc, top3_acc, top5_acc))
                self.log_write.write('Epoch %d: Cos eeg: %.4f, Cos img: %.4f, loss val: %.4f\n'%(e, loss_eeg.detach().cpu().numpy(), loss_img.detach().cpu().numpy(), vloss.detach().cpu().numpy()))


        # * test part
        all_center = test_center
        total = 0
        top1 = 0
        top3 = 0
        top5 = 0
        # test process by loading the best model

        # self.Enc_eeg = Enc_eeg().cuda()
        # self.Proj_eeg = Proj_eeg().cuda()
        # self.Proj_img = Proj_img().cuda()
        # self.Enc_eeg = nn.DataParallel(self.Enc_eeg, device_ids=[i for i in range(len(gpus))])
        # self.Proj_eeg = nn.DataParallel(self.Proj_eeg, device_ids=[i for i in range(len(gpus))])
        # self.Proj_img = nn.DataParallel(self.Proj_img, device_ids=[i for i in range(len(gpus))])

        self.Enc_eeg.load_state_dict(torch.load('./model/' + model_idx + 'Enc_eeg_cls.pth'), strict=False)
        self.Proj_eeg.load_state_dict(torch.load('./model/' + model_idx + 'Proj_eeg_cls.pth'), strict=False)
        self.Proj_img.load_state_dict(torch.load('./model/' + model_idx + 'Proj_img_cls.pth'), strict=False)
        
        # self.Enc_eeg = self.Enc_eeg.to(self.device)
        # self.Proj_eeg = self.Proj_eeg.to(self.device)
        # self.Proj_img = self.Proj_img.to(self.device)

        self.Enc_eeg.eval()
        self.Proj_eeg.eval()
        self.Proj_img.eval()

        with torch.no_grad():
            for i, (teeg, tlabel) in enumerate(self.test_dataloader):
                teeg = Variable(teeg.type(self.Tensor))
                tlabel = Variable(tlabel.type(self.LongTensor))
                all_center = Variable(all_center.type(self.Tensor))            

                tfea = self.Proj_eeg(self.Enc_eeg(teeg))
                tfea = tfea / tfea.norm(dim=1, keepdim=True)
                similarity = (100.0 * tfea @ all_center.t()).softmax(dim=-1)  # no use 100?
                _, indices = similarity.topk(5)

                tt_label = tlabel.view(-1, 1)
                # y_pred = torch.max(Cls, 1)[1]
                total += tlabel.size(0)
                top1 += (tt_label == indices[:, :1]).sum().item()
                top3 += (tt_label == indices[:, :3]).sum().item()
                top5 += (tt_label == indices).sum().item()

            
            top1_acc = float(top1) / float(total)
            top3_acc = float(top3) / float(total)
            top5_acc = float(top5) / float(total)
        
        print('The test Top1-%.6f, Top3-%.6f, Top5-%.6f' % (top1_acc, top3_acc, top5_acc))
        self.log_write.write('The best epoch is: %d\n' % best_epoch)
        self.log_write.write('The test Top1-%.6f, Top3-%.6f, Top5-%.6f\n' % (top1_acc, top3_acc, top5_acc))
        
        return top1_acc, top3_acc, top5_acc
        # writer.close()


def main():
    args = parser.parse_args()

    num_sub = args.num_sub   
    aver = 0
    aver3 = 0
    aver5 = 0
    result_write = open(result_path + "sub_result.txt", "w")

    for i in range(num_sub):
        starttime = datetime.datetime.now()
        seed_n = np.random.randint(args.seed)

        print('seed is ' + str(seed_n))
        random.seed(seed_n)
        np.random.seed(seed_n)
        torch.manual_seed(seed_n)
        torch.cuda.manual_seed(seed_n)
        torch.cuda.manual_seed_all(seed_n)
        # i = 3
        print('Subject %d' % (i+1))
        ie = IE(args, i + 1)

        Acc, Acc3, Acc5 = ie.train()
        print('THE BEST ACCURACY IS ' + str(Acc))
        result_write.write('Subject ' + str(i + 1) + ' : ' + 'Seed: ' + str(seed_n) + "\n")
        result_write.write('Average: Top1-%.6f, Top3-%.6f, Top5-%.6f\n' % (Acc, Acc3, Acc5))

        endtime = datetime.datetime.now()
        print('subject %d duration: '%(i+1) + str(endtime - starttime))
        # plot_confusion_matrix(Y_true, Y_pred, i+1)


        aver += Acc
        aver3 += Acc3
        aver5 += Acc5

    aver = aver / num_sub
    aver3 = aver3 / num_sub
    aver5 = aver5 / num_sub

    result_write.write('overall Aver: Top1-%.6f, Top3-%.6f, Top5-%.6f\n' % (aver, aver3, aver5))

    result_write.close()


if __name__ == "__main__":
    print(time.asctime(time.localtime(time.time())))
    main()
    print(time.asctime(time.localtime(time.time())))
