"""Download the CWRU recordings listed in cwru_files.py into data/raw."""

import time
import urllib.request
from pathlib import Path

from cwru_files import DOWNLOAD_URL, RECORDINGS

RAW_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def download_file(url, destination, max_attempts=3):
    for attempt in range(1, max_attempts + 1):
        try:
            temporary_path = destination.with_suffix(".part")
            urllib.request.urlretrieve(url, temporary_path)
            temporary_path.replace(destination)
            return True
        except Exception as error:
            print(f"    attempt {attempt} failed: {error}")
            time.sleep(2 * attempt)
    return False


def main():
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    failed_files = []

    for file_number, class_name, fault_diameter, motor_load in RECORDINGS:
        destination = RAW_DATA_DIR / f"{file_number}.mat"
        if destination.exists() and destination.stat().st_size > 0:
            print(f"{destination.name}: already downloaded")
            continue

        url = DOWNLOAD_URL.format(file_number=file_number)
        print(f"{destination.name}: downloading ({class_name}, {fault_diameter} in, {motor_load} HP)")
        if not download_file(url, destination):
            failed_files.append(destination.name)
        time.sleep(0.3)

    total_megabytes = sum(path.stat().st_size for path in RAW_DATA_DIR.glob("*.mat")) / 1e6
    print(f"\nFiles in {RAW_DATA_DIR}: {len(list(RAW_DATA_DIR.glob('*.mat')))} ({total_megabytes:.1f} MB)")
    if failed_files:
        print("Failed:", ", ".join(failed_files), "- run the script again to retry.")


if __name__ == "__main__":
    main()
