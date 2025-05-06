import os
import numpy as np
import pandas as pd
import pywt
from scipy.stats import skew, kurtosis, entropy
from scipy.signal import spectrogram, welch
import re


# Define paths
preprocessed_folder = r"C:\Preprocessed data"
output_folder = (
    r"C:\CSVs_New"
)
os.makedirs(output_folder, exist_ok=True)

# EEG frequency bands
bands = {
    "Delta": (0.5, 4),
    "Theta": (4, 8),
    "Alpha": (8, 13),
    "Beta": (13, 30),
    "Gamma": (30, 100),
}


# Extract channel names from text file
def extract_channel_names(summary_path):
    """
    Extracts EEG channel names from a CHB-MIT summary text file.
    Preserves the order and includes dummy channels ("-").
    """
    channel_names = []
    capture = False

    with open(summary_path, "r") as file:
        for line in file:
            line = line.strip()
            if line.startswith("Channels in EDF Files"):
                capture = True  # Start capturing channel names
                continue
            if capture:
                if line.startswith("File Name"):  # Stop capturing at next section
                    break
                match = re.match(r"Channel \d+: (.+)", line)
                if match:
                    channel_names.append(match.group(1))  # Extract channel name

    return channel_names


# Hjorth Parameters
def hjorth_parameters(signal):
    activity = np.var(signal)
    mobility = np.sqrt(np.var(np.diff(signal)) / activity)
    complexity = np.sqrt(np.var(np.diff(np.diff(signal))) / np.var(np.diff(signal)))
    return activity, mobility, complexity


# Zero Crossing Rate
def zero_crossing_rate(signal):
    return np.sum(np.diff(np.sign(signal)) != 0) / len(signal)


# Frequency Domain Features (Welch PSD)
def compute_frequency_features(signal, fs):
    freqs, psd = welch(signal, fs=fs, nperseg=1024)
    mean_freq = np.sum(freqs * psd) / np.sum(psd)
    median_freq = freqs[np.where(np.cumsum(psd) >= np.sum(psd) / 2)[0][0]]
    peak_freq = freqs[np.argmax(psd)]
    psd_norm = psd / np.sum(psd)
    spec_entropy = entropy(psd_norm)
    band_powers = {
        band: np.sum(psd[(freqs >= low) & (freqs <= high)]) / np.sum(psd)
        for band, (low, high) in bands.items()
    }
    return mean_freq, median_freq, peak_freq, spec_entropy, band_powers


# Time-Frequency Features (STFT)
def compute_stft_features(signal, fs):
    f, t, Sxx = spectrogram(signal, fs=fs, nperseg=256)
    Sxx_norm = Sxx / np.sum(Sxx)
    spec_entropy = np.mean(entropy(Sxx_norm, axis=0))
    mean_freq = np.mean(np.sum(f[:, None] * Sxx, axis=0) / np.sum(Sxx, axis=0))
    median_freq = np.mean(
        f[np.argmax(np.cumsum(Sxx, axis=0) >= np.sum(Sxx, axis=0) / 2, axis=0)]
    )
    peak_freq = np.mean(f[np.argmax(Sxx, axis=0)])
    return mean_freq, median_freq, peak_freq, spec_entropy


def compute_wavelet_energy(signal, wavelet="db4", level=4):
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    return [np.sum(np.square(c)) for c in coeffs[1:]]


# Extract channel names from the summary file
def extract_channel_names(summary_path):
    """Extracts channel names from the summary text file."""
    channel_names = []
    with open(summary_path, "r") as file:
        lines = file.readlines()
        capture = False
        for line in lines:
            if line.startswith("Channels in EDF Files"):
                capture = True
            if capture and line.startswith("Channel"):
                parts = line.split(":")
                if len(parts) > 1:
                    channel_name = parts[1].strip()
                    channel_names.append(channel_name)
            if capture and line.startswith("File Name:"):
                # Stop capturing once the next file is reached
                break
    return channel_names


# For each patient folder
for patient in os.listdir(preprocessed_folder):
    patient_path = os.path.join(preprocessed_folder, patient)
    if os.path.isdir(patient_path):
        save_path = os.path.join(output_folder, patient)
        os.makedirs(save_path, exist_ok=True)

        # Find the summary text file for the patient
        summary_files = [
            f for f in os.listdir(patient_path) if f.endswith("-summary.txt")
        ]
        if not summary_files:
            print(f"Warning: No summary text file found for {patient}. Skipping...")
            continue

        # Extract channel names from the summary text
        summary_path = os.path.join(
            patient_path, summary_files[0]
        )  # Assuming one summary per patient
        channel_names = extract_channel_names(summary_path)
        channel_names.pop(0)
        print(channel_names)

        # Gather list of .npy files and determine max channels across files
        npy_files = [f for f in os.listdir(patient_path) if f.endswith(".npy")]
        if not npy_files:
            print(f"No .npy files found in {patient_path}.")
            continue
        # Determine maximum channels across EEG files
        max_channels = 0
        for file in npy_files:
            data = np.load(os.path.join(patient_path, file))
            max_channels = max(max_channels, data.shape[0])

        # Ensure that channel names match the max channels
        if len(channel_names) < max_channels:
            for i in range(len(channel_names), max_channels):
                channel_names.append(f"Ch{i+1}")  # Fallback naming
        elif len(channel_names) > max_channels:
            channel_names = channel_names[:max_channels]  # Trim excess names

        # List to collect all rows (one per window) for the patient
        patient_features = []

        # Process each .npy file (each representing one window)
        for file in npy_files:
            file_path = os.path.join(patient_path, file)
            window_data = np.load(file_path)  # Expected shape: (channels, samples)
            fs = 256  # Adjust sample rate if needed

            # Start the row with the window (file) name
            row_features = [file]
            n_channels = window_data.shape[0]

            # Compute features for each channel in this file
            for i in range(n_channels):
                signal = window_data[i]
                # Time-domain features
                mean_val = np.mean(signal)
                variance = np.var(signal)
                std_dev = np.std(signal)
                rms = np.sqrt(np.mean(signal**2))
                skewness = skew(signal)
                kurt = kurtosis(signal)
                peak_to_peak = np.ptp(signal)
                zcr = zero_crossing_rate(signal)
                activity, mobility, complexity = hjorth_parameters(signal)

                # Frequency-domain features
                mean_freq, median_freq, peak_freq, spec_entropy, band_powers = (
                    compute_frequency_features(signal, fs)
                )

                # Time-frequency features (STFT)
                mean_tf, median_tf, peak_tf, spec_entropy_tf = compute_stft_features(
                    signal, fs
                )

                # Wavelet energy features (4 levels)
                wavelet_energy = compute_wavelet_energy(signal)

                # Combine features for this channel (28 features)
                channel_features = (
                    [
                        mean_val,
                        variance,
                        std_dev,
                        rms,
                        skewness,
                        kurt,
                        peak_to_peak,
                        zcr,
                        activity,
                        mobility,
                        complexity,
                        mean_freq,
                        median_freq,
                        peak_freq,
                        spec_entropy,
                    ]
                    + list(band_powers.values())
                    + [mean_tf, median_tf, peak_tf, spec_entropy_tf]
                    + wavelet_energy
                )

                row_features.extend(channel_features)

            # If this file has fewer channels than max_channels, pad with NaNs for the missing channels
            if n_channels < max_channels:
                missing_channels = max_channels - n_channels
                row_features.extend([np.nan] * (missing_channels * 28))

            # Now each row should have: 1 + (max_channels * 28) features
            patient_features.append(row_features)

        # If no features were extracted, warn and skip CSV creation.
        if not patient_features:
            print(f"No feature data extracted for patient {patient}.")
            continue

        # Generate column names dynamically
        columns = ["Window_Name"]
        feature_types = (
            [
                "Mean",
                "Variance",
                "STD",
                "RMS",
                "Skewness",
                "Kurtosis",
                "Peak-to-Peak",
                "ZCR",
                "Activity",
                "Mobility",
                "Complexity",
                "Mean Frequency",
                "Median Frequency",
                "Peak Frequency",
                "Spectral Entropy",
            ]
            + list(bands.keys())
            + [
                "Mean Frequency (T-F)",
                "Median Frequency (T-F)",
                "Peak Frequency (T-F)",
                "Spectral Entropy (T-F)",
            ]
            + [f"Wavelet Energy L{lvl}" for lvl in range(1, 5)]
        )

        for ch in channel_names:
            for feature in feature_types:
                columns.append(f"{feature}_channel_{ch}")

        expected_total_columns = 1 + max_channels * 28
        if len(columns) != expected_total_columns:
            print("Warning: Column count does not match expected feature count.")

        # Create DataFrame from the collected rows and save as CSV.
        final_df = pd.DataFrame(patient_features, columns=columns)
        output_file = os.path.join(save_path, f"{patient}_features.csv")
        final_df.to_csv(output_file, index=False)
        print(f"Saved: {output_file}")
print("Feature extraction completed successfully!")
