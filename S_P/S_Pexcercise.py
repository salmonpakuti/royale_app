import numpy as np
import matplotlib.pyplot as plt

# パラメータ設定
fs = 100  # サンプリング周波数 (Hz)
t = np.arange(0, 10, 1/fs)  # 時間軸 (10秒間)

# 信号の生成
x = (1.3) * np.sin(2 * np.pi * 15 * t) + (1.7) * np.sin(2 * np.pi * 40 * (t - 2))

# FFTの計算
N = len(x)  # サンプリング数
X = np.fft.fft(x)  # FFT
frequencies = np.fft.fftfreq(N, d=1/fs)  # 周波数軸
amplitudes = np.abs(X) / (N / 2)  # 振幅（スケーリング）

# 正の周波数成分のみを抽出
positive_frequencies = frequencies[:N // 2]
positive_amplitudes = amplitudes[:N // 2]

# プロット
plt.figure(figsize=(10, 6))
plt.plot(positive_frequencies, positive_amplitudes)
plt.xlabel('Frequency (Hz)', fontsize=12)
plt.ylabel('Amplitude', fontsize=12)
plt.title('{\bf Periodogram}', fontsize=14)
plt.grid(True)
plt.show()
