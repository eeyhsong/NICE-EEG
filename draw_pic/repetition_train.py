import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats


sig1 = stats.wilcoxon(r1, c1, alternative='two-sided')[1]
sig2 = stats.wilcoxon(r2, c2, alternative='two-sided')[1]
sig3 = stats.wilcoxon(r3, c3, alternative='two-sided')[1]
sig4 = stats.wilcoxon(r4, c4, alternative='two-sided')[1]

sig11 = stats.wilcoxon(r2, r1, alternative='greater')[1]
sig12 = stats.wilcoxon(r3, r2, alternative='greater')[1]
sig13 = stats.wilcoxon(r4, r3, alternative='greater')[1]

sig21 = stats.wilcoxon(c2, c1, alternative='greater')[1]
sig22 = stats.wilcoxon(c3, c2, alternative='greater')[1]
sig23 = stats.wilcoxon(c4, c3, alternative='greater')[1]


y1 = np.array(repeti)*100
y2 = np.array(condi)*100
std1 = np.array(repeti_std)*100
std2 = np.array(condi_std)*100


# Create figure and axis objects
fig, ax = plt.subplots(figsize=(7, 6))
x = np.arange(4)    
# Plot lines and fill between with standard deviation as shadow

ax.plot(x, y1, '.-', label='All repetitions')
plt.rcParams.update({'font.size': 14})
ax.fill_between(x, y1-std1, y1+std1, alpha=0.2)
# ax.errorbar(x, y1, yerr=std1, fmt='.-', label='All conditions')
ax.plot(x, y2, '.-', label='All conditions')
plt.rcParams.update({'font.size': 14})
ax.fill_between(x, y2-std2, y2+std2, alpha=0.2)
# ax.errorbar(x, y2, yerr=std2, fmt='.-', label='All repetitions')

top = 15
ax.plot([0, 0, 1, 1], [top+0.8, top+1, top+1, top+0.8], 'k-', lw=1.5)
ax.plot([1, 1, 2, 2], [top+1.8, top+2, top+2, top+1.8], 'k-', lw=1.5)
ax.plot([2, 2, 3, 3], [top+2.8, top+3, top+3, top+2.8], 'k-', lw=1.5)


ax.hlines(0.5, 0, 3, colors = "grey", linestyles = "dashed", zorder = 1)

plt.text((0+1)*.5-0.05, top+1, '**', ha='center', va='bottom', color='#2077B4', fontsize=12)
plt.text((0+1)*.5+0.05, top+1, '**', ha='center', va='bottom', color='#FF7F0E', fontsize=12)
plt.text((1+2)*.5-0.025, top+2, '*', ha='center', va='bottom', color='#2077B4', fontsize=12)
plt.text((1+2)*.5+0.025, top+2, '*', ha='center', va='bottom', color='#FF7F0E', fontsize=12)
plt.text((2+3)*.5, top+3, '*', ha='center', va='bottom', color='#2077B4', fontsize=12)

# plt.text((1+2)*.5, top+2, '*', ha='center', va='bottom', color='k', fontsize=12)
# plt.text((2+3)*.5, top+3, '**', ha='center', va='bottom', color='k', fontsize=12)


# ax.vlines([1, 6], 0, 20, colors = "grey", linestyles = "dashed", linewidth=0.5, alpha=0.5)
# ax.vlines(6, 0, 20, colors = "grey", linestyles = "dashed", label='t=200 ms', linewidth=0.5, alpha=0.5)

# x_pt = 6
# y_pt = 15.75

# # add a text label at the point of interest
# ax.text(x_pt, y_pt, '(600, 15.75)', color='black')
# ax.text(1, 0.9, '(100, 0.9)', color='black')
# plt.annotate(f'y={y_pt:.2f}', xy=(x_pt, y_pt), xytext=(x_pt+0.5, y_pt+0.5),
#              arrowprops=dict(facecolor='black', shrink=0.05))

# Set labels and legend
ax.set_xlabel('Amount of training data (%)', fontsize=14)
ax.set_ylabel('Accuracy (%)', fontsize=14)
ax.set_xticks(x, ['25', '50', '75', '100'])
# ax.set_xticks(x)
# ax.set_xlim([0, 10])
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
ax.set_ylim([0, 22])
ax.legend(loc=9, ncol=2, fontsize=14, frameon=False)


# Show plot
# plt.savefig('./pic/Conf/time_window.png', dpi=300)
plt.tight_layout()
plt.savefig('./pic/repetition_train.svg', dpi=300)