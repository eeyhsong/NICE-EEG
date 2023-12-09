"""
Obtain ViT features of training and test images in Things-EEG.

using huggingface pretrained ViT model

"""

import argparse
import torch.nn as nn
import numpy as np
import torch
from torch.autograd import Variable as V
from torchvision import transforms as trn
import os
from PIL import Image
from transformers import ViTForImageClassification

gpus = [7]
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, gpus))

parser = argparse.ArgumentParser()
parser.add_argument('--pretrained', default=True, type=bool)
parser.add_argument('--project_dir', default='/home/Data/Things-EEG2/', type=str)
args = parser.parse_args()

print('Extract feature maps ViT <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))

# Set random seed for reproducible results
seed = 20200220
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

vit_model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
model = vit_model.vit.cuda()
model = nn.DataParallel(model, device_ids=[i for i in range(len(gpus))])

centre_crop = trn.Compose([
	trn.Resize((224, 224)),
	trn.ToTensor(),
	# trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	trn.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Image directories
img_set_dir = os.path.join(args.project_dir, 'Image_set/image_set')
img_partitions = os.listdir(img_set_dir)
for p in img_partitions:
	part_dir = os.path.join(img_set_dir, p)
	image_list = []
	for root, dirs, files in os.walk(part_dir):
		for file in files:
			if file.endswith(".jpg") or file.endswith(".JPEG"):
				image_list.append(os.path.join(root,file))
	image_list.sort()
	# Create the saving directory if not existing
	save_dir = os.path.join(args.project_dir, 'DNN_feature_maps',
		'full_feature_maps', 'vit', 'pretrained-'+str(args.pretrained), p)
	if os.path.isdir(save_dir) == False:
		os.makedirs(save_dir)

	# Extract and save the feature maps
	# * better to use a dataloader
	for i, image in enumerate(image_list):
		img = Image.open(image).convert('RGB')
		input_img = V(centre_crop(img).unsqueeze(0))
		if torch.cuda.is_available():
			input_img=input_img.cuda()
		x = model(input_img).last_hidden_state[:,0,:]
		feats = x.detach().cpu().numpy()
		file_name = p + '_' + format(i+1, '07')
		np.save(os.path.join(save_dir, file_name), feats)
