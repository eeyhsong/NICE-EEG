import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

mean_repeti = [0.0275, 0.0744, 0.0986, 0.1132, 0.1155, 0.129, 0.133, 0.13075, 0.132, 0.1365, 0.1345, 0.1525, 0.1435, 0.1395, 0.138, 0.1355, 0.137]
std_repeti = [0.006264751, 0.015853738, 0.023334193, 0.02975007, 0.024749985, 0.034676473, 0.027608372, 0.031025303, 0.032951985, 0.033997549, 0.038976489, 0.039033461, 0.036442497, 0.029292585, 0.040428263, 0.029949031, 0.036530049]

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
plt.savefig('./pic/Conf/repetition_test.svg', dpi=300)