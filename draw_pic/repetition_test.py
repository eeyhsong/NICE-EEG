import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

mean_repeti = np.array(mean_repeti) * 100
std_repeti = np.array(std_repeti) * 100

labels = ['1', '5', '10', '15', '20', '25', '30', '35', '40', '45', '50', '55', '60', '65', '70', '75', '80']
# create figure and axis objects
fig, ax = plt.subplots(figsize=(8, 5))

# create bar chart
bars = ax.bar(labels, mean_repeti, yerr=std_repeti, align='center', ecolor='black', capsize=4, alpha=0.8)
for i in range(len(bars)):
    bars[i].set_color(plt.cm.Blues((i+5)/(len(bars)+12)))

ax.hlines(0.5, -0.5, 16.5, colors = "grey", linestyles = "dashed")
# set labels and title
ax.set_xlabel('Repeat Times', fontsize=14)
ax.set_ylabel('Accuracy (%)', fontsize=14)
ax.set_xlim(-0.5, 16.5)
# ax.set_title('Bar Chart with Error Bars')
plt.tight_layout()
plt.savefig('./pic/repetition_test.svg', dpi=300)