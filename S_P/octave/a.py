import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd

# Parameters
fs = 44100  # Sampling frequency
t = np.arange(0, 1, 1/fs)  # Time vector for 1 second
f = 1  # Frequency of sine wave (1 Hz)

# Generate sine wave
x = np.sin(2 * np.pi * f * t)

# Plot the sine wave
plt.plot(t, x)
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.title('1 Hz Sine Wave')
plt.show()

# Play the sound
sd.play(x, fs)
sd.wait()  # Wait until sound is done playing