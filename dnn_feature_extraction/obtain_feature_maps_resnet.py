"""
Obtain ViT features of training and test images in Things-EEG.

using huggingface pretrained ResNet model

"""

import argparse
from torchvision import models
import torch.nn as nn
import numpy as np
import torch
from torch.autograd import Variable as V
from torchvision import transforms as trn
import os
from PIL import Image
import requests

import torchvision.transforms as transforms
import torchvision.datasets as datasets
from transformers import AutoImageProcessor, ViTForImageClassification, ViTFeatureExtractor, ViTModel
# from transformers import CLIPProcessor, CLIPModel
from transformers import AutoImageProcessor, ResNetForImageClassification
gpus = [6]
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, gpus))


# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--pretrained', default=True, type=bool)
parser.add_argument('--project_dir', default='/home/songyonghao/Documents/Data/Things-EEG2/', type=str)
args = parser.parse_args()

print('Extract feature maps ResNet <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))

# Set random seed for reproducible results
seed = 20200220
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
# torch.use_deterministic_algorithms(True)


# =============================================================================
# Select the layers of interest and import the model
# =============================================================================

model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")

# model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
model = model.cuda()
model = nn.DataParallel(model, device_ids=[i for i in range(len(gpus))])


# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# image = Image.open(requests.get(url, stream=True).raw)

# inputs = processor(text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True)

# outputs = model(**inputs)
# logits_per_image = outputs.logits_per_image # this is the image-text similarity score
# probs = logits_per_image.softmax(dim=1) # we can take the softmax to get the label probabilities


# =============================================================================
# Define the image preprocessing
# =============================================================================
# centre_crop = trn.Compose([
# 	trn.Resize((224, 224)),
# 	trn.ToTensor(),
# 	# trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# 	trn.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
# ])
# processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")


# =============================================================================
# Load the images and extract the corresponding feature maps
# =============================================================================
# Extract the feature maps of (1) training images, (2) test images,
# (3) ILSVRC-2012 validation images, (4) ILSVRC-2012 test images.

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
		'full_feature_maps', 'resnet', 'pretrained-'+str(args.pretrained), p)
	if os.path.isdir(save_dir) == False:
		os.makedirs(save_dir)

	# Extract and save the feature maps
	# * better to use a dataloader
	for i, image in enumerate(image_list):
		img = Image.open(image).convert('RGB')
		inputs = processor(images=img, return_tensors="pt")
		# inputs.data['pixel_values'].cuda()
		x = model(**inputs).logits[0]
		feats = x.detach().cpu().numpy()
		# for f, feat in enumerate(x):
		# 	feats[model.feat_list[f]] = feat.data.cpu().numpy()
		file_name = p + '_' + format(i+1, '07')
		np.save(os.path.join(save_dir, file_name), feats)
