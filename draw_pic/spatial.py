import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

frontral = [0.12, 0.135, 0.135, 0.145, 0.09, 0.125, 0.15, 0.22, 0.115, 0.16]
central = [0.13, 0.11, 0.135, 0.155, 0.08, 0.12, 0.14, 0.155, 0.155, 0.15]
parietal = [0.105, 0.095, 0.115, 0.14, 0.08, 0.11, 0.14, 0.17, 0.125, 0.14]
occipital = [0.085, 0.08, 0.115, 0.115, 0.06, 0.1, 0.115, 0.16, 0.06, 0.11]
temporal = [0.125, 0.08, 0.13, 0.135, 0.07, 0.125, 0.13, 0.155, 0.11, 0.13]
all = [0.123, 0.104, 0.131, 0.164, 0.08, 0.141, 0.152, 0.2, 0.133, 0.149]

p_frontral = stats.wilcoxon(all, frontral, alternative='greater', )[1] * 5
p_central = stats.wilcoxon(all, central, alternative='greater')[1] * 5 
p_parietal = stats.wilcoxon(all, parietal, alternative='greater')[1] * 5
p_occipital = stats.wilcoxon(all, occipital, alternative='greater')[1] * 5
p_temporal = stats.wilcoxon(all, temporal, alternative='greater')[1] * 5

# t_frontral = stats.ttest_ind(all, frontral)[1]
# t_central = stats.ttest_ind(all, central)[1]
# t_parietal = stats.ttest_ind(all, parietal)[1]
# t_occipital = stats.ttest_ind(all, occipital)[1]
# t_temporal = stats.ttest_ind(all, temporal)[1]

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
plt.savefig('./pic/Conf/spatial_region.svg', dpi=300)