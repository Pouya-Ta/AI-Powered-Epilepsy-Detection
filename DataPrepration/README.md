# EEG Preprocessing & Feature Extraction

This directory contains two Python scripts to preprocess CHB-MIT EEG recordings and extract key features.

---

## Scripts

1. **CHB-MIT_preprocessing.py**
   - **Purpose:**
     - Load raw `.edf` files
     - Apply bandpass (0.5–80 Hz) and 60 Hz notch filters
     - Perform STFT-based denoising
     - Segment data into fixed-length preictal and interictal windows
   - **Usage:**
     ```bash
     python CHB-MIT_preprocessing.py \
       --input-folder /path/to/raw_edf/ \
       --output-folder /path/to/preprocessed/ \
       --window-size 10  # seconds
     ```
   - **Outputs:**
     - `preictal/` and `interictal/` folders with `.npy` arrays `(channels, samples)`
     - `seizure_summary.csv` listing file names, labels, and timestamps

2. **chb_mit_featureExtraction_newVersion.py**
   - **Purpose:**
     - Read `.npy` windows from preprocessing
     - Compute time-domain (Hjorth parameters), frequency-domain (band powers), and time-frequency (wavelet energies) features
     - Export features to CSV per patient
   - **Usage:**
     ```bash
     python chb_mit_featureExtraction_newVersion.py \
       --preprocessed-folder /path/to/preprocessed/ \
       --output-folder /path/to/features/ \
       --wavelet db4 \
       --levels 4
     ```
   - **Outputs:**
     - CSV files `patientXX_features.csv`, each row = one 10 s window, columns = extracted features

---

## Dependencies
Install required packages:
```bash
pip install numpy scipy pandas mne pywt
````
