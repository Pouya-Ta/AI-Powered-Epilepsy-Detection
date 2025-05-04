import pandas as pd
import os

train_dir = r"C:\Users\Pouya\Documents\university\Bachelor - AUT\Final Project\Data Set\Nowrous\After Model-Based"
test_dir = r"C:\Users\Pouya\Documents\university\Bachelor - AUT\Final Project\Data Set\Nowrous\After Data Prepration"
output_dir = r"C:\Users\Pouya\Documents\university\Bachelor - AUT\Final Project\Data Set\Nowrous\After Test-Modification"

os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(train_dir):
    if filename.endswith("_model_based.csv"):
        patient_id = filename.replace("train_patient_", "").replace(
            "_model_based.csv", ""
        )
        train_file = os.path.join(train_dir, filename)
        test_file = os.path.join(test_dir, f"test_patient_{patient_id}.csv")

        if not os.path.exists(test_file):
            print(f"Test file for {patient_id} not found, skipping.")
            continue

        # Load train and test files
        train_df = pd.read_csv(train_file)
        test_df = pd.read_csv(test_file)

        # Identify columns to keep from train
        selected_columns = train_df.columns
        test_df = test_df[selected_columns]

        # Output filenames
        train_out = os.path.join(output_dir, f"final_train_{patient_id}.csv")
        test_out = os.path.join(output_dir, f"final_test_{patient_id}.csv")

        train_df.to_csv(train_out, index=False)
        test_df.to_csv(test_out, index=False)

        print(f"Saved final train/test files for {patient_id}")
