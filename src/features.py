"""Turn signal windows into model inputs."""

import numpy as np
from scipy import stats

from cwru_files import TARGET_SAMPLING_RATE_HZ


def normalize_windows(windows):
    """Z-score each window on its own (zero mean, unit standard deviation)."""
    means = windows.mean(axis=1, keepdims=True)
    standard_deviations = windows.std(axis=1, keepdims=True)
    return (windows - means) / (standard_deviations + 1e-12)


def fft_spectrum_features(windows):
    """Log magnitude of the one-sided FFT of each normalized window.

    A Hann taper reduces leakage at the window edges. The DC bin is dropped,
    leaving 1024 bins from about 5.9 Hz to 6 kHz.
    """
    normalized = normalize_windows(windows)
    taper = np.hanning(windows.shape[1])
    magnitudes = np.abs(np.fft.rfft(normalized * taper, axis=1))
    return np.log1p(magnitudes[:, 1:])


def handcrafted_features(windows):
    """The four features from the original notebook: RMS, kurtosis,
    FFT peak frequency (Hz), and spectral entropy."""
    feature_rows = []
    frequencies_hz = np.fft.rfftfreq(windows.shape[1], d=1 / TARGET_SAMPLING_RATE_HZ)
    for window in windows:
        rms = np.sqrt(np.mean(window ** 2))
        kurtosis = stats.kurtosis(window)
        magnitudes = np.abs(np.fft.rfft(window))
        peak_frequency = frequencies_hz[np.argmax(magnitudes[1:]) + 1]
        power = magnitudes ** 2
        power_fraction = power / np.sum(power)
        spectral_entropy = -np.sum(power_fraction * np.log2(power_fraction + 1e-12))
        feature_rows.append([rms, kurtosis, peak_frequency, spectral_entropy])
    return np.array(feature_rows)
