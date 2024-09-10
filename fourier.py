import numpy as np
import matplotlib.pyplot as plt

# Define the signal
#t = np.linspace(0, 2*np.pi, 1000)  # Time values
#signal = np.sin(5*t) + 0.5*np.sin(10*t) + 0.2*np.sin(20*t)  # Example signal

filename = "A00019"
filepath = "/home/saptarshi/Research/CF_Explanation/ISI_2D_to_1D_data/" + str(filename) + ".npy"
signal = np.load(filepath)



import numpy as np
import matplotlib.pyplot as plt

# Given array representing the 1D signal
#signal = np.array([1, 2, 3, 4, 5, 4, 3, 2, 1])  # Example signal

# Sampling rate
sampling_rate = 300 # Example sampling rate (samples per second)

# Compute the time array based on the sampling rate
total_time = len(signal) / sampling_rate
t = np.linspace(0, total_time, len(signal), endpoint=False)

# Compute the Fourier transform
fourier_transform = np.fft.fft(signal)
frequencies = np.fft.fftfreq(len(signal), 1 / sampling_rate)

# Plot the signal
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(t, signal)
plt.title('Original Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

# Plot the Fourier transform
plt.subplot(2, 1, 2)
plt.plot(frequencies, np.abs(fourier_transform))
plt.title('Fourier Transform')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.xlim(0, 50)  # Limit the x-axis to 80 Hz
plt.ylim(0, 150) 
plt.tight_layout()
# Save the plot
plt.savefig('fourier_transform_plot.png')

# Show the plot
plt.show()
