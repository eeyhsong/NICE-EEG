"""
Package ResNet features for center images

"""

import argparse
import torch.nn as nn
import numpy as np
import torch
import os
from PIL import Image
from transformers import AutoImageProcessor, ResNetForImageClassification

gpus = [3]
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, gpus))

parser = argparse.ArgumentParser()
parser.add_argument('--pretrained', default=True, type=bool)
parser.add_argument('--project_dir', default='/home/Data/Things-EEG2/Image_set', type=str)
args = parser.parse_args()

print('Extract feature maps ResNet of images for center <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))

# Set random seed for reproducible results
seed = 20200220
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")
model = model.cuda()
model = nn.DataParallel(model, device_ids=[i for i in range(len(gpus))])

processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")

img_set_dir = os.path.join(args.project_dir, 'image_set/center_images/')
condition_list = os.listdir(img_set_dir)
condition_list.sort()

all_centers = []

for cond in condition_list:
    one_cond_dir = os.path.join(args.project_dir, 'image_set/center_images/', cond)
    cond_img_list = os.listdir(one_cond_dir)
    cond_img_list.sort()
    cond_center = []
    for img in cond_img_list:
        img_path = os.path.join(one_cond_dir, img)
        img = Image.open(img_path).convert('RGB')
        inputs = processor(images=img, return_tensors="pt")
        x = model(**inputs).logits[0]

        # cond_center.append(outputs.detach().cpu().numpy())
    # cond_center = np.mean(cond_center, axis=0)
    # all_centers.append(np.squeeze(cond_center))
        cond_center.append(x.detach().cpu().numpy())
    all_centers.append(np.array(cond_center))

# all_centers = np.array(all_centers)
# print(all_centers.shape)
np.save(os.path.join(args.project_dir, 'center_all_image_resnet.npy'), all_centers)
