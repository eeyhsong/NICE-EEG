import mne
import numpy as np
import scipy.io
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties 
# font = FontProperties(fname=r"simsun.ttc", size=8) 

results = []
for sub in range(10):
    tmp = np.load('./pic/similarity/sub%d_sim.npy' % (sub+1))
    results.append(tmp)
results = np.array(results)
results = np.mean(results, axis=0)
results = (results - np.min(results)) / (np.max(results) - np.min(results))

plt.figure(figsize=(14, 12))
plt.imshow(results, cmap='GnBu')  # spectral_r, ocean_r, GnBu

plt.plot([0, 35], [35, 35], color='black', alpha=0.3)
plt.plot([35, 35], [0, 35], color='black', alpha=0.3)
plt.plot([0, 0], [0, 35], color='black', alpha=0.3)
plt.plot([0, 35], [0, 0], color='black', alpha=0.3)

plt.plot([35, 82], [82, 82], color='black', alpha=0.3)
plt.plot([82, 82], [35, 82], color='black', alpha=0.3)
plt.plot([35, 35], [35, 82], color='black', alpha=0.3)
plt.plot([35, 82], [35, 35], color='black', alpha=0.3)


plt.plot([82, 102], [102, 102], color='black', alpha=0.3)
plt.plot([102, 102], [82, 102], color='black', alpha=0.3)
plt.plot([82, 82], [82, 102], color='black', alpha=0.3)
plt.plot([82, 102], [82, 82], color='black', alpha=0.3)

plt.plot([102, 137], [137, 137], color='black', alpha=0.3)
plt.plot([137, 137], [102, 137], color='black', alpha=0.3)
plt.plot([102, 102], [102, 137], color='black', alpha=0.3)
plt.plot([102, 137], [102, 102], color='black', alpha=0.3)

plt.plot([137, 200], [200, 200], color='black', alpha=0.3)
plt.plot([200, 200], [137, 200], color='black', alpha=0.3)
plt.plot([137, 137], [137, 200], color='black', alpha=0.3)
plt.plot([137, 200], [137, 137], color='black', alpha=0.3)


plt.colorbar(shrink=0.8)

x_ticks = [35, 82, 102, 137, 200]
x_ticklabels = ['animal', 'food', 'vehicle', 'tool', 'others']
plt.xlim(0, 200)
plt.ylim(200, 0)
plt.xticks(x_ticks, x_ticklabels, fontsize=18)
plt.yticks(x_ticks, x_ticklabels, fontsize=18, rotation=90)


plt.title('Similarity', size=20)
plt.tight_layout()
# plt.show()
plt.savefig('./pic/heatmap1.png', dpi=300)

print('the end')
