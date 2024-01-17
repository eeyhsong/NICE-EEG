 # # Frequency spectrum
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import signal
from scipy.fftpack import fft, fftfreq
from scipy.signal import butter, lfilter, freqz
from matplotlib import gridspec 


train_data = []
train_label = []

# sub 1 7 10 is good!
nSub = 1
eeg_data_path = '/home/Data/Things-EEG2/Preprocessed_data_250Hz/'

train_data = np.load(eeg_data_path + '/sub-' + format(nSub, '02') + '/preprocessed_eeg_training.npy', allow_pickle=True)
train_data = train_data['preprocessed_eeg_data']
# train_data = train_data[:, 0:4, :, :]
train_data = np.mean(train_data, axis=1)
train_data = np.expand_dims(train_data, axis=1)

test_data = np.load(eeg_data_path + '/sub-' + format(nSub, '02') + '/preprocessed_eeg_test.npy', allow_pickle=True)
test_data = test_data['preprocessed_eeg_data']
test_data = np.mean(test_data, axis=1)
test_data = np.expand_dims(test_data, axis=1)


Parietal_ch = [37, 38, 39, 40, 41, 42, 43, 46, 47, 48, 49, 50, 51, 52, 53, 54] # 16
Occipital_ch = [55, 56, 57, 58, 59, 60, 61, 62] # 8
Temporal_ch = [15, 16, 17, 23, 24, 25, 26, 27, 33, 34, 35, 36, 44, 45] # 14

# time-frequency map
from scipy import signal
fs = 250  
t = np.linspace(0, 1, fs, endpoint=False)
x = np.mean(np.squeeze(train_data), axis=0)

x1 = x[Occipital_ch]
x1 = np.mean(x1, axis=0)
f1, t1, Sxx1 = signal.spectrogram(x1, fs, nperseg=50, scaling='spectrum')

x2 = x[Temporal_ch]
x2 = np.mean(x2, axis=0)
f2, t2, Sxx2 = signal.spectrogram(x2, fs, nperseg=50, scaling='spectrum')

x3 = x[Parietal_ch]
x3 = np.mean(x3, axis=0)
f3, t3, Sxx3 = signal.spectrogram(x3, fs, nperseg=50, scaling='spectrum')


Sxx1 = np.log10(Sxx1)
Sxx1 -= np.max(Sxx1)
Sxx1 /= np.abs(np.min(Sxx1))
Sxx2 = np.log10(Sxx2)
Sxx2 -= np.max(Sxx2)
Sxx2 /= np.abs(np.min(Sxx2))
Sxx3 = np.log10(Sxx3)
Sxx3 -= np.max(Sxx3)
Sxx3 /= np.abs(np.min(Sxx3))
# Sxx -= np.max(Sxx)
# Sxx /= np.abs(np.min(Sxx))   


# Set up the figure and gridspec
fig = plt.figure(figsize=(20, 6))

gs = gridspec.GridSpec(1, 4, width_ratios=[1, 1, 1, 0.1])

# Plot the first group of data
ax1 = plt.subplot(gs[0])
im1 = ax1.pcolormesh(t1, f1, Sxx1, cmap='PiYG')
ax1.set_title('Occipital', fontsize=16)
ax1.set_xlabel('Time (ms)', fontsize=14)
ax1.set_xticklabels([0, 200, 400, 600, 800])

ax1.set_ylabel('Frequency (Hz)', fontsize=16)
ax1.set_ylim([0, 100])
ax1.tick_params(labelsize=14)


# Plot the second group of data
ax2 = plt.subplot(gs[1])
im2 = ax2.pcolormesh(t2, f2, Sxx2, cmap='PiYG')
ax2.set_title('Temporal', fontsize=16)
ax2.set_xlabel('Time (ms)', fontsize=14)
ax2.set_ylim([0, 100])
ax2.tick_params(labelsize=14)
ax2.set_xticklabels([0, 200, 400, 600, 800])

# Plot the third group of data
ax3 = plt.subplot(gs[2])
im3 = ax3.pcolormesh(t3, f3, Sxx3, cmap='PiYG')
ax3.set_title('Parietal', fontsize=16)
ax3.set_xlabel('Time (s)', fontsize=14)
ax3.set_ylim([0, 100])
ax3.tick_params(labelsize=14)
ax3.set_xticklabels([0, 200, 400, 600, 800])

# Add a colorbar to the right of the third plot
cax = plt.subplot(gs[:, -1])
fig.colorbar(im3, cax=cax)

# # Add some padding between the subplots
# plt.subplots_adjust(wspace=0.4)

plt.savefig('./pic/Conf/time_freq.svg', dpi=300)



# fig, ax = plt.subplots(figsize=(6, 6))
# im = ax.pcolormesh(t, f, Sxx, cmap='PiYG')
# ax.set_xlabel('Time (s)')
# ax.set_ylabel('Frequency (Hz)')
# ax.set_ylim([0, 100])
# cbar = fig.colorbar(im)
# cbar.set_label('Power [dB]')
# # plt.show()
# plt.savefig('./pic/Conf/time_freq_occipital.png', dpi=300)


# # band-pass filter
# import signal
# # # EEG rythm band
# # # theta 4-8 Hz
# # # alpha 8-12 Hz
# # # beta 12-30 Hz
# # # low gamma 32-45 Hz
# # # high gamma 55-95 Hz
# # a band pass fitler from 4 to 8 Hz with sample rate 1000 Hz

# from scipy.signal import butter, filtfilt

# # Define the filter parameters
# lowcut = 8  # Hz
# highcut = 12  # Hz
# fs = 250  # Hz
# order = 4

# # Create the filter coefficients
# nyquist = 0.5 * fs
# low = lowcut / nyquist
# high = highcut / nyquist
# b, a = butter(order, [low, high], btype='band')

# # Generate a test signal
# t = np.linspace(0, 1, fs, endpoint=False)
# # signal = np.sin(2 * np.pi * 10 * t) + np.sin(2 * np.pi * 20 * t) + np.sin(2 * np.pi * 50 * t)
# signal = np.mean(np.squeeze(train_data), axis=0)
# # signal = signal[60]
# # signal = np.mean(signal, axis=0)

# # Filter the signal
# filtered_signal = filtfilt(b, a, signal)

# # Plot the results
# fig, axs = plt.subplots(2, 1, figsize=(10, 6))
# axs[0].plot(signal.transpose())
# axs[0].set_title('Original Signal')
# axs[1].plot(filtered_signal.transpose())
# axs[1].set_title('Filtered Signal')
# plt.savefig('bandpass.png')

# # Plot the frequency response of the filter
# w, h = scipy.signal.freqz(b, a)
# # Plot frequency response
# fig, (ax1, ax2) = plt.subplots(2, 1)
# fig.suptitle('Filter Frequency Response')
# ax1.plot((fs * 0.5 / np.pi) * w, abs(h))
# ax1.set_ylabel('Amplitude')
# ax1.set_xlabel('Frequency (Hz)')
# ax1.set_ylim([0, 1.1])
# ax1.grid(True)

# # Plot phase
# ax2.plot((fs * 0.5 / np.pi) * w, np.angle(h))
# ax2.set_ylabel('Phase (radians)')
# ax2.set_xlabel('Frequency (Hz)')
# ax2.set_ylim([-np.pi, np.pi])
# ax2.grid(True)

# # Just Amplitude
# fig, ax = plt.subplots(figsize=(10, 6))
# ax.plot(0.5*fs*w/np.pi, np.abs(h))
# ax.set_title('Frequency Response')
# ax.set_xlabel('Frequency (Hz)')
# ax.set_ylabel('Amplitude')
# ax.grid(True)
# plt.savefig('bp_freq_response.png')