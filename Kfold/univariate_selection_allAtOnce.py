import os
import pandas as pd

# === Config ===
base_dir = r"C:\Users\Pouya\Documents\university\Bachelor - AUT\Final Project\Data Set\Nowrous\K-FOLD"

train_dir = os.path.join(base_dir, "After Data Prepration K-fold")
score_dir = os.path.join(base_dir, "After Feature Scoring K-fold")
output_dir = os.path.join(base_dir, "After Univariate K-fold")
top_n = 150

os.makedirs(output_dir, exist_ok=True)

# === Loop through all train_patient CSVs ===
for filename in os.listdir(train_dir):
    if filename.startswith("train_") and filename.endswith(".csv"):
        patient_id = filename.replace("train_", "").replace(".csv", "")
        train_file_path = os.path.join(train_dir, filename)
        feature_scores_path = os.path.join(
            score_dir, f"feature_scores_{patient_id}.csv"
        )
        output_path = os.path.join(
            output_dir, f"train_{patient_id}_univariate_features.csv"
        )

        print(f"\nProcessing: {filename}")
        print(f"Loading feature scores from: {feature_scores_path}")

        # Load scoring results
        feature_scores_df = pd.read_csv(feature_scores_path)
        feature_scores_df = feature_scores_df.sort_values(
            by="Vote_Score", ascending=False
        )
        selected_features = feature_scores_df.head(top_n)["Feature"].tolist()

        print(f"Top {top_n} features selected.")

        # Load train dataset
        df = pd.read_csv(train_file_path)

        # Validate selected features exist
        selected_features = [f for f in selected_features if f in df.columns]
        print(f"Final selected features after validation: {len(selected_features)}")

        # Add label + ID (assuming label = last column, ID = first)
        final_df = df[selected_features + [df.columns[-1], df.columns[0]]]

        # Ensure "Window_Name" is the first column if it exists
        if "Window_Name" in final_df.columns:
            cols = ["Window_Name"] + [
                col for col in final_df.columns if col != "Window_Name"
            ]
            final_df = final_df[cols]

        # Save output
        final_df.to_csv(output_path, index=False)
        print(f"Saved selected features to: {output_path}")
