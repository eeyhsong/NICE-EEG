import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

delta = np.array(delta) * 100
theta = np.array(theta) * 100
alpha = np.array(alpha) * 100
beta = np.array(beta) * 100
gamma = np.array(gamma) * 100


data = [delta, theta, alpha, beta, gamma]
top = np.max(data[-1])

fig, ax = plt.subplots(figsize=(8, 4))

# sns.set_style("whitegrid")
# sns.violinplot(data=data, palette="Set3", bw=.2, cut=1, linewidth=1)
sns.boxplot(data=data, palette='Greens', showmeans=True, width=0.5, whis=100, meanprops={'marker':'^', 'markerfacecolor':'blue', 'markeredgecolor':'black', 'markersize':'5'})

# ax.plot([2, 2, 5, 5], [top+0.8, top+1, top+1, top+0.8], 'k-', lw=1.5)
# ax.plot([3, 3, 5, 5], [top+1.8, top+2, top+2, top+1.8], 'k-', lw=1.5)
# ax.plot([4, 4, 5, 5], [top+2.8, top+3, top+3, top+2.8], 'k-', lw=1.5)
# plt.text((2+5)*.5, top+1, '*', ha='center', va='bottom', color='k', fontsize=12)
# plt.text((3+5)*.5, top+2, '*', ha='center', va='bottom', color='k', fontsize=12)
# plt.text((4+5)*.5, top+3, '**', ha='center', va='bottom', color='k', fontsize=12)


# ax.set_ylim(0, 18)
ax.set_xticklabels(['delta', 'theta', 'alpha', 'beta', 'gamma'], fontsize=12)
ax.set_xlabel('Frequency band', fontsize=12)
ax.set_ylabel('Accuracy (%)', fontsize=12)

# save the figure
plt.savefig('./pic/frequency_band.svg', dpi=300)