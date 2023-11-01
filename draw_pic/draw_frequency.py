 # # Frequency spectrum
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import signal
from scipy.fftpack import fft, fftfreq
from scipy.signal import butter, lfilter, freqz


# fs = 250  # sampling frequency
# fs_len = 250
# t = np.arange(0, 1, 1/fs)
# x = np.mean(np.squeeze(train_data), axis=0)
# x = np.mean(x, axis=0)
# # x = x[60]

# # Compute the frequency spectrum
# X = np.fft.fft(x)
# freq = np.fft.fftfreq(fs_len, 1/fs)
# freq = freq[:fs_len//2]  # Only plot positive frequencies
# X_mag = np.abs(X[:fs_len//2]) / fs_len  # Magnitude spectrum

# # Plot the frequency spectrum
# plt.figure()
# plt.plot(freq, X_mag)
# plt.xlabel('Frequency (Hz)')
# plt.ylabel('Magnitude')
# plt.savefig('data_freq.png')
# # plt.show()


Parietal_ch = [37, 38, 39, 40, 41, 42, 43, 46, 47, 48, 49, 50, 51, 52, 53, 54] # 16
Occipital_ch = [55, 56, 57, 58, 59, 60, 61, 62] # 8
Temporal_ch = [15, 16, 17, 23, 24, 25, 26, 27, 33, 34, 35, 36, 44, 45] # 14

# # time-frequency map
# from scipy import signal
# fs = 250  
# t = np.linspace(0, 1, fs, endpoint=False)
# x = np.mean(np.squeeze(train_data), axis=0)
# x = x[Occipital_ch]
# x = np.mean(x, axis=0)

# f, t, Sxx = signal.spectrogram(x, fs, nperseg=50, scaling='spectrum')

# Sxx = np.log10(Sxx)
# Sxx -= np.max(Sxx)
# Sxx /= np.abs(np.min(Sxx))         

# fig, ax = plt.subplots()
# im = ax.pcolormesh(t, f, Sxx, cmap='viridis')
# ax.set_xlabel('Time (s)')
# ax.set_ylabel('Frequency (Hz)')
# ax.set_ylim([0, 100])
# cbar = fig.colorbar(im)
# cbar.set_label('Power [dB]')
# # plt.show()
# plt.savefig('time_freq_occipital.png')


# band-pass filter
import signal
# # EEG rythm band
# # theta 4-8 Hz
# # alpha 8-12 Hz
# # beta 12-30 Hz
# # low gamma 32-45 Hz
# # high gamma 55-95 Hz
# a band pass fitler from 4 to 8 Hz with sample rate 1000 Hz

from scipy.signal import butter, filtfilt

# Define the filter parameters
lowcut = 8  # Hz
highcut = 12  # Hz
fs = 250  # Hz
order = 4

# Create the filter coefficients
nyquist = 0.5 * fs
low = lowcut / nyquist
high = highcut / nyquist
b, a = butter(order, [low, high], btype='band')

# Generate a test signal
t = np.linspace(0, 1, fs, endpoint=False)
# signal = np.sin(2 * np.pi * 10 * t) + np.sin(2 * np.pi * 20 * t) + np.sin(2 * np.pi * 50 * t)
signal = np.mean(np.squeeze(train_data), axis=0)
# signal = signal[60]
# signal = np.mean(signal, axis=0)

# Filter the signal
filtered_signal = filtfilt(b, a, signal)

# Plot the results
fig, axs = plt.subplots(2, 1, figsize=(10, 6))
axs[0].plot(signal.transpose())
axs[0].set_title('Original Signal')
axs[1].plot(filtered_signal.transpose())
axs[1].set_title('Filtered Signal')
plt.savefig('bandpass.png')

# Plot the frequency response of the filter
w, h = scipy.signal.freqz(b, a)
# Plot frequency response
fig, (ax1, ax2) = plt.subplots(2, 1)
fig.suptitle('Filter Frequency Response')
ax1.plot((fs * 0.5 / np.pi) * w, abs(h))
ax1.set_ylabel('Amplitude')
ax1.set_xlabel('Frequency (Hz)')
ax1.set_ylim([0, 1.1])
ax1.grid(True)

# Plot phase
ax2.plot((fs * 0.5 / np.pi) * w, np.angle(h))
ax2.set_ylabel('Phase (radians)')
ax2.set_xlabel('Frequency (Hz)')
ax2.set_ylim([-np.pi, np.pi])
ax2.grid(True)

plt.savefig('bp_freq_response.png')