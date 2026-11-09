"""Read CWRU .mat files, resample to a common rate, and cut them into windows."""

from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.signal import decimate

from cwru_files import (
    CLASS_NAMES,
    FAULT_SAMPLING_RATE_HZ,
    NORMAL_SAMPLING_RATE_HZ,
    RECORDINGS,
    TARGET_SAMPLING_RATE_HZ,
)

RAW_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
WINDOW_SIZE = 2048
HOP_SIZE = 1024


def read_drive_end_signal(file_number):
    """Return the drive-end accelerometer signal stored in one .mat file.

    Some files hold signals from more than one recording (99.mat also has
    X098_DE_time), so the variable name must match the file number.
    """
    mat_contents = loadmat(RAW_DATA_DIR / f"{file_number}.mat")
    expected_key = f"X{file_number:03d}_DE_time"
    if expected_key not in mat_contents:
        drive_end_keys = [key for key in mat_contents if key.endswith("_DE_time")]
        raise KeyError(f"{file_number}.mat has no {expected_key}; found {drive_end_keys}")
    return mat_contents[expected_key].flatten().astype(np.float64)


def resample_to_target_rate(signal, original_rate_hz):
    if original_rate_hz == TARGET_SAMPLING_RATE_HZ:
        return signal
    factor = original_rate_hz // TARGET_SAMPLING_RATE_HZ
    return decimate(signal, factor, ftype="fir", zero_phase=True)


def split_into_windows(signal):
    starts = range(0, len(signal) - WINDOW_SIZE + 1, HOP_SIZE)
    return np.array([signal[start:start + WINDOW_SIZE] for start in starts])


def load_all_windows():
    """Return windows plus per-window labels and recording metadata."""
    all_windows = []
    labels = []
    file_numbers = []
    motor_loads = []
    fault_diameters = []

    for file_number, class_name, fault_diameter, motor_load in RECORDINGS:
        signal = read_drive_end_signal(file_number)
        if class_name == "Normal":
            signal = resample_to_target_rate(signal, NORMAL_SAMPLING_RATE_HZ)
        else:
            signal = resample_to_target_rate(signal, FAULT_SAMPLING_RATE_HZ)

        windows = split_into_windows(signal)
        all_windows.append(windows)
        labels += [CLASS_NAMES.index(class_name)] * len(windows)
        file_numbers += [file_number] * len(windows)
        motor_loads += [motor_load] * len(windows)
        fault_diameters += [fault_diameter] * len(windows)

    return {
        "windows": np.concatenate(all_windows),
        "labels": np.array(labels),
        "file_numbers": np.array(file_numbers),
        "motor_loads": np.array(motor_loads),
        "fault_diameters": np.array(fault_diameters),
    }
