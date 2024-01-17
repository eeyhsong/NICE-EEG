import numpy as np
import matplotlib.pyplot as plt

new_order = np.array([83, 1, 103, 138, 36, 139, 148, 140, 143, 104, 
                     105, 2, 149, 84, 37, 106, 85, 38, 150, 107, 
                     108, 39, 109, 31, 86, 151, 40, 152, 41, 153, 
                     87, 42, 3, 4, 110, 154, 155, 43, 5, 111, 
                     156, 112, 113, 114, 157, 32, 44, 45, 158, 46,
                     47, 159, 33, 48, 49, 160, 50, 34, 88, 51, 
                     52, 115, 6, 53, 7, 144, 161, 162, 8, 9, 
                     54, 10, 55, 163, 89, 11, 116, 117, 164, 118, 
                     56, 57, 119, 90, 91, 12, 13, 14, 15, 120, 
                     58, 121, 122, 165, 166, 141, 16, 167, 168, 92, 
                     59, 169, 170, 142, 123, 17, 171, 172, 60, 18, 
                     19, 173, 61, 124, 93, 125, 174, 175, 126, 62, 
                     176, 63, 64, 65, 66, 177, 20, 178, 21, 179, 
                     67, 68, 22, 127, 69, 23, 24, 180, 128, 70, 
                     71, 25, 72, 26, 129, 181, 73, 74, 145, 27, 
                     130, 28, 182, 94, 183, 184, 75, 76, 77, 95, 
                     29, 185, 186, 96, 97, 187, 146, 131, 188, 132,
                     133, 98, 134, 78, 99, 189, 190, 191, 192, 193, 
                     194, 195, 35, 79, 135, 136, 196, 100, 197, 30, 
                     101, 137, 147, 198, 80, 81, 102, 199, 82, 200])


y_true_all = []
y_pred_all = []
for sub_idx in range(10):
    for run_idx in range(20):
        y_true = np.arange(200)
        y_pred = np.load('./pic/y_pred_arange/sub%d_pred%d.npy' % (sub_idx+1, run_idx))
        y_pred = y_pred[:, 0]
        y_pred_all.append(y_pred)
        y_true_all.append(np.arange(200))

y_true_all = np.concatenate(y_true_all)
y_pred_all = np.concatenate(y_pred_all)


# draw confusion matrix with y_true and y_pred, normalized
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_true_all, y_pred_all)
# cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
plt.figure(figsize=(14, 12))
plt.imshow(cm, cmap='Reds')
plt.title('Confusion matrix', fontsize=20)
# scale the colorbar
plt.colorbar(shrink=0.8)
x_ticks = [35, 82, 102, 137, 147, 200]
x_ticklabels = ['animal', 'food', 'vehicle', 'tool', 'sport', 'others']
plt.xticks(x_ticks, x_ticklabels, fontsize=16)
plt.xlabel('Predicted', fontsize=20)
plt.yticks(x_ticks, x_ticklabels, fontsize=16, rotation=90)
plt.ylabel('True', fontsize=20)
# plt.tight_layout()
plt.savefig('./pic/Conf/confusion_matrix.png', dpi=300)
print('111')
