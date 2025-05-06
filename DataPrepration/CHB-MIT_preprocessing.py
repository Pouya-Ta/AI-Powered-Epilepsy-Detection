import os
import mne
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, iirnotch
from scipy.signal import welch
from scipy import signal
import re
import warnings

warnings.filterwarnings("ignore", message="Scaling factor is not defined")
random.seed(42)

# Define paths
data_folder = r"C:\Data"  # Change to your raw data folder
preprocessed_folder = r"C:\Preprocessed data"  # Preprocessed data storage
plots_folder = (
    r"C:\Plots"
)
seizure_summary_csv_path = r"C:\seizure_summary.csv"  # Global CSV summary file

os.makedirs(preprocessed_folder, exist_ok=True)
os.makedirs(plots_folder, exist_ok=True)


def band_filter(data, low_freq, high_freq, sfreq):
    """Applies a bandpass filter."""
    nyquist = sfreq / 2
    low, high = low_freq / nyquist, high_freq / nyquist
    b, a = butter(N=4, Wn=[low, high], btype="bandpass")
    return filtfilt(b, a, data, axis=1)


def notch_filter(data, sfreq):
    """Applies a notch filter at 60 Hz."""
    b, a = iirnotch(60, 30, sfreq)
    return filtfilt(b, a, data, axis=1)


def soft_thresholding(X, threshold):
    return np.sign(X) * np.maximum(np.abs(X) - threshold, 0)


def preprocess_eeg(raw_data, fs):
    """Applies bandpass and notch filters."""
    filtered_data = band_filter(raw_data, 0.5, 80, fs)
    centered_data = filtered_data - np.mean(filtered_data, axis=1, keepdims=True)
    cleaned_data = notch_filter(centered_data, fs)
    f, t, Zxx = signal.stft(cleaned_data, fs=fs, nperseg=2 * fs)

    mad_noise = np.median(np.abs(Zxx), axis=(2), keepdims=True)
    threshold = mad_noise * (0.2 + f[None, :, None] / 100)
    # Apply frequency-adaptive thresholding
    Zxx_denoised = soft_thresholding(Zxx, threshold)
    _, eeg_denoised = signal.istft(Zxx_denoised, fs=fs, nperseg=2 * fs)

    eeg_denoised = eeg_denoised[:, : cleaned_data.shape[1]]
    return eeg_denoised


def create_windows(data, fs, window_size):
    """Splits data into fixed-size windows."""
    step = int(window_size * fs)
    return [data[:, i : i + step] for i in range(0, data.shape[1] - step + 1, step)]


def plot_window(window, fs, ch_names, save_path):
    """Plots a given EEG window and saves it."""
    time = np.linspace(0, window.shape[1] / fs, window.shape[1])
    plt.figure(figsize=(10, 6))
    for i, channel in enumerate(window):
        plt.plot(time, channel, label=ch_names[i])
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (µV)")
    plt.title(os.path.basename(save_path))
    plt.legend(
        loc="upper right", bbox_to_anchor=(1.15, 1.0), fontsize="small", frameon=True
    )
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def read_summary_csv(csv_path):
    """
    Reads CSV summary file and extracts seizure information.

    Expected CSV format:
      File Name,Seizure Start Time,Seizure End Time
    """
    df = pd.read_csv(csv_path)
    seizure_info = {}
    for _, row in df.iterrows():
        file_name = row["File_name"]
        start_time = int(row["Seizure_start"])
        end_time = int(row["Seizure_stop"])
        if file_name in seizure_info:
            seizure_info[file_name].append((start_time, end_time))
        else:
            seizure_info[file_name] = [(start_time, end_time)]
    return seizure_info


# Modify segment_eeg_with_prediction_logic to return count:
def segment_eeg_with_prediction_logic(
    file_name, preprocessed_data, fs, patient_folder, ch_names, seizure_info
):
    total_preictal_windows = 0
    if file_name not in seizure_info or not seizure_info[file_name]:
        print(f"Info: {file_name} has no seizures, skipping preictal segmentation.")
        return total_preictal_windows
    window_size_sec = 10  # Use 20-second windows
    patient_preprocessed_folder = os.path.join(preprocessed_folder, patient_folder, 'preictal')
    os.makedirs(patient_preprocessed_folder, exist_ok=True)


    for seizure_start, seizure_end in seizure_info[file_name]:
        # PREICTAL: 10–30 min before seizure onset
        preictal_start = max(0, seizure_start - 900)  # 20 (900) mins before seizure
        preictal_end = max(0, seizure_start - 300)  # 10 (300) mins before seizure
        if preictal_end > preictal_start:
            preictal_data = preprocessed_data[
                :, preictal_start * fs : preictal_end * fs
            ]
            preictal_windows = create_windows(preictal_data, fs, window_size_sec)
        else:
            preictal_windows = []
            print(
                f"Warning: For {file_name}, no valid preictal window found (preictal_end <= preictal_start)."
            )

        # Save preictal windows only
        for i, window in enumerate(preictal_windows):
            save_path = os.path.join(
                patient_preprocessed_folder, f"{file_name}_preictal_{i}.npy"
            )
            np.save(save_path, window)
            plot_window(window, fs, ch_names, save_path.replace(".npy", "_plot.png"))

        total_preictal_windows += len(preictal_windows)
        print(total_preictal_windows)
        print(f'{filename} Saved')

    return total_preictal_windows


def segment_interictal_only(
    file_name, preprocessed_data, fs, patient_folder, ch_names, target_count=None
):
    """Creates interictal windows matching number of preictal windows."""
    """If target_count is provided and the number of windows exceeds it, random undersampling is applied;
    otherwise, all windows are used.
    """
    print('Before window')
    window_size_sec = 10
    all_windows = create_windows(preprocessed_data, fs, window_size_sec)

    # Random undersampling to match preictal count
    if target_count is not None and len(all_windows) > target_count:
        windows = random.sample(all_windows, target_count)
    else:
        windows = all_windows
    print('Presaving')    
    patient_preprocessed_folder = os.path.join(preprocessed_folder, patient_folder, 'interictal')
    os.makedirs(patient_preprocessed_folder, exist_ok=True)

    for i, window in enumerate(windows):
        save_path = os.path.join(
            patient_preprocessed_folder, f"{file_name}_interictal_{i}.npy"
        )
        np.save(save_path, window)
        plot_window(window, fs, ch_names, save_path.replace(".npy", "_plot.png"))
    print(f'{filename} Saved')


# Load the global seizure information from the CSV file
global_seizure_info = read_summary_csv(seizure_summary_csv_path)


# Sorting folders numerically
def extract_number(folder_name):
    match = re.search(r"\d+", folder_name)  # Extract numeric part
    return int(match.group()) if match else float("inf")


sorted_folders = sorted(os.listdir(data_folder), key=extract_number)
preictal_count = {}
for patient_folder in sorted_folders:
    print('------------------------------------------------------------')
    print(f"Processing folder: {patient_folder}")
    patient_path = os.path.join(data_folder, patient_folder)

    if not os.path.isdir(patient_path):
        print(f"Skipping {patient_folder}: Not a directory")
        continue
    patient_preprocessed_folder = os.path.join(preprocessed_folder, patient_folder)
    print('Preictal windows creation started')
    # Process each .edf file in the patient folder
    for filename in os.listdir(patient_path):
        if not filename.endswith(".edf"):
            continue

        file_path = os.path.join(patient_path, filename)

        try:
            raw = mne.io.read_raw_edf(file_path, preload=True)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue

        if filename in global_seizure_info and global_seizure_info[filename]:
            raw.drop_channels([ch for ch in raw.info["ch_names"] if ch == "-"])
            eeg_data = raw.get_data()
            fs = int(raw.info["sfreq"])
            ch_names = raw.ch_names
            preprocessed_data = preprocess_eeg(eeg_data, fs)
            preictal_count[f"{filename}"] = []
            preictal_count_one = segment_eeg_with_prediction_logic(
                    filename,
                    preprocessed_data,
                    fs,
                    patient_folder,
                    ch_names,
                    global_seizure_info)
            preictal_count[f"{filename}"].append(preictal_count_one)
    print('---------------------------------------------------------')
    print('Interictal windows creation started')
    # Process each .edf file in the patient folder
    for filename in os.listdir(patient_path):
        if not filename.endswith(".edf"):
            continue

        file_path = os.path.join(patient_path, filename)

        try:
            raw = mne.io.read_raw_edf(file_path, preload=True)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue
        print(preictal_count)
        if filename not in global_seizure_info or not global_seizure_info[filename]:
            raw.drop_channels([ch for ch in raw.info["ch_names"] if ch == "-"])
            eeg_data = raw.get_data()
            fs = int(raw.info["sfreq"])
            ch_names = raw.ch_names
            preprocessed_data = preprocess_eeg(eeg_data, fs)
            try:
                first_key = list(preictal_count.keys())[0]
                interictal_count = preictal_count[first_key]
                if interictal_count[0] > 0:
                    segment_interictal_only(
                        filename,
                        preprocessed_data,
                        fs,
                        patient_folder,
                        ch_names,
                        target_count=interictal_count[0],
                    )
                    del preictal_count[first_key]
                    interictal_count.pop(0)
            except IndexError:
                print("Skip the file, already made the interictal file")


print("Preprocessing, segmentation, and plotting complete!")
