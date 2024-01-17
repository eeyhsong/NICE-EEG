"""
Package all the ViT features

"""

import argparse
import numpy as np
import os

parser = argparse.ArgumentParser()
parser.add_argument('--dnn', default='vit', type=str)
parser.add_argument('--pretrained', default=True, type=bool)
parser.add_argument('--project_dir', default='/home/Data/Things-EEG2/', type=str)
args = parser.parse_args()

print('>>> Apply PCA on the feature maps <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))

# Set random seed for reproducible results
seed = 20200220

# Load the feature maps
feats = []
fmaps_dir = os.path.join(args.project_dir, 'DNN_feature_maps',
	'full_feature_maps', args.dnn, 'pretrained-'+str(args.pretrained),
	'training_images')
fmaps_list = os.listdir(fmaps_dir)
fmaps_list.sort()
for f, fmaps in enumerate(fmaps_list):
	fmaps_data = np.load(os.path.join(fmaps_dir, fmaps))
	feats.append(fmaps_data)

# Save the downsampled feature maps
save_dir = os.path.join(args.project_dir, 'DNN_feature_maps',
	'pca_feature_maps', args.dnn, 'pretrained-'+str(args.pretrained))
file_name = 'vit_feature_maps_training'
if os.path.isdir(save_dir) == False:
	os.makedirs(save_dir)
np.save(os.path.join(save_dir, file_name), feats)
del feats

# Load the feature maps
feats = []
fmaps_dir = os.path.join(args.project_dir, 'DNN_feature_maps',
	'full_feature_maps', args.dnn, 'pretrained-'+str(args.pretrained),
	'test_images')
fmaps_list = os.listdir(fmaps_dir)
fmaps_list.sort()
for f, fmaps in enumerate(fmaps_list):
	fmaps_data = np.load(os.path.join(fmaps_dir, fmaps))
	feats.append(fmaps_data)

# Save the downsampled feature maps
file_name = 'vit_feature_maps_test'
np.save(os.path.join(save_dir, file_name), feats)
del feats

