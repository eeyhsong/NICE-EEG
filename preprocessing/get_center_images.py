import os
import shutil
import numpy as np

things_path = '/home/Data/THINGS/Images/'
things_eeg_test_images_path = '/home/Data/Things-EEG2/Image_set/image_set/test_images/'
things_eeg_center_images_path = '/home/Data/Things-EEG2/Image_set/image_set/center_images/'

things_list = os.listdir(things_path)[6:]
things_list.sort()
test_list = os.listdir(things_eeg_test_images_path)
test_list.sort()
# center_list = os.listdir(things_eeg_center_images_path)
for i in range(len(test_list)):
    shutil.copytree(things_path+test_list[i][6:], things_eeg_center_images_path+test_list[i][6:])
    os.rename(things_eeg_center_images_path+test_list[i][6:], things_eeg_center_images_path+test_list[i])
    test_img = os.listdir(things_eeg_test_images_path+test_list[i])
    os.unlink(things_eeg_center_images_path+test_list[i]+'/'+test_img[0])

print('ttt')