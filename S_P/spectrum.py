import numpy as np
import matplotlib.pyplot as plt

fs = 8  # サンプリング周波数
N = 8   # サンプル数
t = np.arange(N) / fs
x = np.sin(2 * np.pi * 1 * t)  # 1Hzの正弦波

# DFT 計算
X = np.fft.fft(x)
freqs = np.fft.fftfreq(N, d=1/fs)  # 周波数軸

# 振幅スペクトルをプロット
plt.stem(freqs, np.abs(X), use_line_collection=True)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.title("DFT of 1Hz Sine Wave (fs=8Hz)")
plt.grid()
plt.show()
