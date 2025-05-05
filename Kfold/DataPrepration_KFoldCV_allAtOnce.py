import os
import pandas as pd
from sklearn.model_selection import KFold
# Define main directory and output directory
data_dir = r"C:\Users\Pouya\Documents\university\Bachelor - AUT\Final Project\Data Set\CSV_D"  # Change this if needed
output_dir = r"C:\Users\Pouya\Documents\university\Bachelor - AUT\Final Project\Data Set\Nowrous\K-FOLD\After Data Prepration K-fold"
os.makedirs(output_dir, exist_ok=True)

print("Output directory created (if not existed):", output_dir)

# Get list of patient folders
patient_folders = sorted(os.listdir(data_dir))
print("Found patient folders:", patient_folders)

data_list = []

# Identify common columns by reading the first file of each patient
common_columns = None
for folder in patient_folders:
    folder_path = os.path.join(data_dir, folder)
    if os.path.isdir(folder_path):  # Ensure it's a directory
        csv_files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]
        if csv_files:
            sample_df = pd.read_csv(os.path.join(folder_path, csv_files[0]))
            if common_columns is None:
                common_columns = set(sample_df.columns)
            else:
                common_columns.intersection_update(sample_df.columns)
            print(
                f"Processed {folder}: Found common columns in sample file {csv_files[0]}"
            )

# Convert to list
common_columns = list(common_columns)
if "Window_Name" not in common_columns:
    common_columns.append("Window_Name")  # Ensure "Window_Name" is included
print("Final common columns across all patients:", common_columns)

# Process each patient
for idx, folder in enumerate(patient_folders):
    folder_path = os.path.join(data_dir, folder)
    if os.path.isdir(folder_path):
        csv_files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]
        print(
            f"Processing patient {idx + 1}/{len(patient_folders)}: {folder}, found {len(csv_files)} files"
        )
        for csv_file in csv_files:
            file_path = os.path.join(folder_path, csv_file)
            df = pd.read_csv(file_path)
            print(f"Reading file: {csv_file}, shape: {df.shape}")

            # Keep only common columns
            df = df[common_columns]

            # Debugging: Print unique Window_Name values
            if "Window_Name" in df.columns:
                print("Unique Window_Name values:", df["Window_Name"].unique())
            else:
                print("Error: 'Window_Name' column not found in the dataframe!")

            # Label the data based on "Window_Name"
            def assign_label(name):
                name = str(name).lower()
                if "preictal" in name:
                    return -1
                elif "ictal" in name:
                    return 1
                elif "interictal" in name:
                    return 0
                return None  # Handle unexpected cases

            df["Label"] = df["Window_Name"].apply(assign_label)

            # Debugging: Print label counts
            print("Label distribution:", df["Label"].value_counts())

            # Ensure "Window_Name" is retained in the final dataset
            df = df.dropna(subset=["Label"])  # Remove any rows with invalid labels

            # Remove interictal data
            interictal_count = (df["Label"] == 0).sum()
            df = df[df["Label"] != 0]
            print(
                f"Removed {interictal_count} interictal records, new shape: {df.shape}"
            )

            # Ensure "Label" is the first column if it exists
            if "Label" in df.columns:
                cols = [
                    col for col in df.columns if col != "Label"
                ] + ["Label"] 
                df = df[cols]

            # Ensure "Window_Name" is the first column if it exists
            if "Window_Name" in df.columns:
                cols = ["Window_Name"] + [
                    col for col in df.columns if col != "Window_Name"
                ]
                df = df[cols]
                        # ---------------------------
            # Perform random undersampling
            # ---------------------------
            preictal_df = df[df["Label"] == -1]
            ictal_df = df[df["Label"] == 1]

            print(
                f"Before balancing: preictal={len(preictal_df)}, ictal={len(ictal_df)}"
            )

            if len(preictal_df) > 0 and len(ictal_df) > len(preictal_df):
                ictal_sampled_df = ictal_df.sample(
                    n=len(preictal_df), random_state=42
                )
                df = pd.concat(
                    [preictal_df, ictal_sampled_df], ignore_index=True
                )
                df = df.sample(frac=1, random_state=42).reset_index(
                    drop=True
                )  # Shuffle
                print(
                    f"After balancing: preictal={len(preictal_df)}, ictal={len(ictal_sampled_df)}"
                )
            else:
                print(
                    "Skipping balancing due to insufficient preictal or ictal data."
                )
            # Collect all patient data into a dictionary
            if "patient_data" not in locals():
                patient_data = {}

            patient_data[folder] = df
            df.interpolate(method='linear')
            # Collect all patient data into a dictionary
            data_list.append(df)

# Combine all data into one DataFrame
data_all = pd.concat(data_list, ignore_index=True)

# === K-Fold Cross Validation ===
k_folds = 5
kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

for fold_idx, (train_index, test_index) in enumerate(kf.split(data_all)):
    train_df = data_all.iloc[train_index]
    test_df = data_all.iloc[test_index]

    train_file_path = os.path.join(output_dir, f"train_fold_{fold_idx+1}.csv")
    test_file_path = os.path.join(output_dir, f"test_fold_{fold_idx+1}.csv")

    train_df.to_csv(train_file_path, index=False)
    test_df.to_csv(test_file_path, index=False)

    print(f"Saved K-Fold split {fold_idx+1}: train -> {train_file_path}, test -> {test_file_path}")

