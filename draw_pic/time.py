import numpy as np
import matplotlib.pyplot as plt

back = [0.009, 0.039, 0.086, 0.1365, 0.1385, 0.1575, 0.1475, 0.1415, 0.1385, 0.133]
back_std = [0.0052, 0.0182, 0.0327, 0.0242, 0.0285, 0.0242, 0.0358, 0.0368, 0.0343, 0.0334]
forward = [0.138, 0.1375, 0.1045, 0.0715, 0.0335, 0.0145, 0.0105, 0.007, 0.0055, 0.0045]
forward_std = [0.0308, 0.0355, 0.0317, 0.0261, 0.0075, 0.0093, 0.0085, 0.0042, 0.006, 0.0037]
seg = [0.0115, 0.0315, 0.041, 0.038, 0.0215, 0.0085, 0.007, 0.0075, 0.005, 0.0035]
seg_std = [0.0058, 0.0127, 0.0149, 0.0181, 0.0100, 0.0058, 0.0089, 0.0035, 0.0041, 0.0041]


y1 = np.array(back)*100
y2 = np.array(forward)*100
y3 = np.array(seg)*100
std1 = np.array(back_std)*100
std2 = np.array(forward_std)*100
std3 = np.array(seg_std)*100

# Create figure and axis objects
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(1, 11)    
# Plot lines and fill between with standard deviation as shadow
ax.plot(x, y1, '.-', label='[0, t]')
ax.fill_between(x, y1-std1, y1+std1, alpha=0.2)
ax.plot(x-1, y2, '.-', label='[t, 1000]')
ax.fill_between(x-1, y2-std2, y2+std2, alpha=0.2)
ax.plot(x, y3, '.-', label='[t, t+100]')
ax.fill_between(x, y3-std3, y3+std3, alpha=0.2)

ax.hlines(0.5, 0, 10, colors = "grey", linestyles = "dashed", label='Chance level')
ax.vlines([1, 6], 0, 20, colors = "grey", linestyles = "dashed", linewidth=0.5, alpha=0.5)
# ax.vlines(6, 0, 20, colors = "grey", linestyles = "dashed", label='t=200 ms', linewidth=0.5, alpha=0.5)

x_pt = 6
y_pt = 15.75

# # add a text label at the point of interest
ax.text(x_pt, y_pt, '(600, 15.75)', color='black')
ax.text(1, 0.9, '(100, 0.9)', color='black')
# plt.annotate(f'y={y_pt:.2f}', xy=(x_pt, y_pt), xytext=(x_pt+0.5, y_pt+0.5),
#              arrowprops=dict(facecolor='black', shrink=0.05))

# Set labels and legend
ax.set_xlabel('t (ms)', fontsize=12)
ax.set_ylabel('Accuracy (%)', fontsize=12)
ax.set_xticks(x, ['100', '200', '300', '400', '500', '600', '700', '800', '900', '1000'])
# ax.set_xticks(x)
ax.set_xlim([0, 10])
ax.set_ylim([0, 20])
ax.legend()


plt.savefig('./pic/time_window.svg', dpi=300)