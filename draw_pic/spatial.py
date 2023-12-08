import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns


frontral = np.array(frontral) * 100 
central = np.array(central) * 100
parietal = np.array(parietal) * 100
occipital = np.array(occipital) * 100
temporal = np.array(temporal) * 100
all = np.array(all) * 100

data = [frontral, central, parietal, temporal, occipital, all]
top = np.max(data[-1])

fig, ax = plt.subplots(figsize=(8, 7))

# sns.set_style("whitegrid")
# sns.violinplot(data=data, palette="Set3", bw=.2, cut=1, linewidth=1)
sns.boxplot(data=data, palette='Blues', showmeans=True, width=0.5, whis=10, meanprops={'marker':'^', 'markerfacecolor':'green', 'markeredgecolor':'black', 'markersize':'5'})

ax.plot([2, 2, 5, 5], [top+0.8, top+1, top+1, top+0.8], 'k-', lw=1.5)
ax.plot([3, 3, 5, 5], [top+1.8, top+2, top+2, top+1.8], 'k-', lw=1.5)
ax.plot([4, 4, 5, 5], [top+2.8, top+3, top+3, top+2.8], 'k-', lw=1.5)
plt.text((2+5)*.5, top+1, '*', ha='center', va='bottom', color='k', fontsize=12)
plt.text((3+5)*.5, top+2, '*', ha='center', va='bottom', color='k', fontsize=12)
plt.text((4+5)*.5, top+3, '**', ha='center', va='bottom', color='k', fontsize=12)


ax.set_ylim(5, 25)
ax.set_xticklabels(['Frontal', 'Central', 'Parietal', 'Temporal', 'Occipital', 'All'], fontsize=12)
ax.set_xlabel('Brain region (ablated)', fontsize=12)
ax.set_ylabel('Accuracy (%)', fontsize=12)

# save the figure
plt.savefig('./pic/spatial_region.svg', dpi=300)