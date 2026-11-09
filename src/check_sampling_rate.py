"""Check the sampling rate of each recording from the data itself.

Shaft-speed lines (1x, 2x, 3x RPM/60) move with motor speed. With the correct
sampling rate they sit exactly at integer multiples of the shaft frequency and
stand well above the nearby spectrum. With a rate that is off by 4x they do not.
A strong line with a ratio near 1.000, 2.000, 3.000 marks the correct rate.

Files 98 and 99 store no RPM, so the rated speed for their load is used.
"""

import numpy as np
from scipy.io import loadmat

from cwru_files import RECORDINGS
from load_data import RAW_DATA_DIR, read_drive_end_signal

CANDIDATE_RATES_HZ = [12000, 48000]
RATED_RPM_BY_LOAD = {0: 1797, 1: 1772, 2: 1750, 3: 1730}
FILES_TO_CHECK = [97, 98, 99, 100, 105, 108, 118, 121, 130, 133]


def motor_speed_rpm(file_number, motor_load):
    mat_contents = loadmat(RAW_DATA_DIR / f"{file_number}.mat")
    rpm_key = f"X{file_number:03d}RPM"
    if rpm_key in mat_contents:
        return float(np.ravel(mat_contents[rpm_key])[0])
    return float(RATED_RPM_BY_LOAD[motor_load])


def strongest_line_near(magnitudes, frequencies, target_hz):
    """Return (frequency of strongest bin within 0.5% of target, its prominence)."""
    bin_width_hz = frequencies[1] - frequencies[0]
    near_target = np.abs(frequencies - target_hz) <= max(0.005 * target_hz, 1.5 * bin_width_hz)
    surroundings = (np.abs(frequencies - target_hz) > 0.03 * target_hz) & (
        np.abs(frequencies - target_hz) <= 0.15 * target_hz
    )
    best_index = np.where(near_target)[0][np.argmax(magnitudes[near_target])]
    prominence = magnitudes[best_index] / np.median(magnitudes[surroundings])
    return frequencies[best_index], prominence


def main():
    load_by_file = {file_number: load for file_number, _, _, load in RECORDINGS}
    for file_number in FILES_TO_CHECK:
        rpm = motor_speed_rpm(file_number, load_by_file[file_number])
        shaft_hz = rpm / 60
        signal = read_drive_end_signal(file_number)
        magnitudes = np.abs(np.fft.rfft(signal * np.hanning(len(signal))))
        report_parts = []
        for rate_hz in CANDIDATE_RATES_HZ:
            frequencies = np.fft.rfftfreq(len(signal), d=1 / rate_hz)
            line_texts = []
            for order in (1, 2, 3):
                line_hz, prominence = strongest_line_near(magnitudes, frequencies, order * shaft_hz)
                line_texts.append(f"{line_hz / shaft_hz:.3f} ({prominence:5.1f}x)")
            report_parts.append(f"{rate_hz // 1000} kHz: " + "  ".join(line_texts))
        print(f"{file_number:>4}.mat {rpm:5.0f} rpm, {len(signal):6d} samples | " + " | ".join(report_parts))


if __name__ == "__main__":
    main()
