# import os
# import pickle
# import numpy as np

# num_sub = 10

# root = '/home/songyonghao/Documents/Data/Things-EEG2/Preprocessed_data/'

# for sub_idx in range(num_sub):
#     training_data = np.load(os.path.join(root, 'sub-'+format(sub_idx+1,'02'), 'preprocessed_eeg_training.npy'), allow_pickle=True)
#     training_data['preprocessed_eeg_data'] = np.mean(training_data['preprocessed_eeg_data'], 1)
    
#     save_eeg_train = open(os.path.join(root, 'trainR4_testR80', 'sub-'+format(sub_idx+1,'02')+'_training.npy'), 'wb')
#     pickle.dump(training_data, save_eeg_train, protocol=4)
# 	# save_eeg.close()
#     del training_data

#     test_data = np.load(os.path.join(root, 'sub-'+format(sub_idx+1,'02'), 'preprocessed_eeg_test.npy'), allow_pickle=True)
#     test_data['preprocessed_eeg_data'] = np.mean(test_data['preprocessed_eeg_data'], 1)

#     save_eeg_test = open(os.path.join(root, 'trainR4_testR80', 'sub-'+format(sub_idx+1,'02')+'_test.npy'), 'wb')
#     pickle.dump(test_data, save_eeg_test, protocol=4)
#     # save_eeg.close()
#     del test_data
# print('111')

import os
import pickle
import numpy as np

num_sub = 10

root = '/home/songyonghao/Documents/Data/Things-EEG2/Preprocessed_data/'

for sub_idx in range(num_sub):
    training_data = np.load(os.path.join(root, 'sub-'+format(sub_idx+1,'02'), 'preprocessed_eeg_training.npy'), allow_pickle=True)
    # training_data['preprocessed_eeg_data'] = np.mean(training_data['preprocessed_eeg_data'], 1)
    training_data['preprocessed_eeg_data'] = np.concatenate((training_data['preprocessed_eeg_data']))

    save_eeg_train = open(os.path.join(root, 'trainAll_testR80', 'sub-'+format(sub_idx+1,'02')+'_training.npy'), 'wb')
    pickle.dump(training_data, save_eeg_train, protocol=4)
	# save_eeg.close()
    del training_data

    # test_data = np.load(os.path.join(root, 'sub-'+format(sub_idx+1,'02'), 'preprocessed_eeg_test.npy'), allow_pickle=True)
    # test_data['preprocessed_eeg_data'] = np.mean(test_data['preprocessed_eeg_data'], 1)

    # save_eeg_test = open(os.path.join(root, 'trainR4_testR80', 'sub-'+format(sub_idx+1,'02')+'_test.npy'), 'wb')
    # pickle.dump(test_data, save_eeg_test, protocol=4)
    # # save_eeg.close()
    # del test_data
print('111')

