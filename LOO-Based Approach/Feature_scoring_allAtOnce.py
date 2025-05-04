import os
import pandas as pd
import numpy as np
from sklearn.feature_selection import f_classif, mutual_info_classif

# Statistical tests in order to be applied as afunction here
def fisher_score(X, y):
    """Computes Fisher Score for each feature."""
    unique_classes = np.unique(y)
    num_features = X.shape[1]
    scores = np.zeros(num_features)

    overall_mean = np.mean(X, axis=0)

    numerator = np.zeros(num_features)
    denominator = np.zeros(num_features)

    for c in unique_classes:
        class_mask = y == c
        class_mean = np.mean(X[class_mask], axis=0)
        class_variance = np.var(X[class_mask], axis=0)
        class_count = np.sum(class_mask)

        numerator += class_count * (class_mean - overall_mean) ** 2
        denominator += class_count * class_variance

    scores = numerator / (denominator + 1e-10)  # Avoid division by zero
    return scores


# --- Configurable paths ---
input_dir = r"C:\Users\Pouya\Documents\university\Bachelor - AUT\Final Project\Data Set\Nowrous\K-FOLD\After Data Prepration K-fold"
output_dir = r"C:\Users\Pouya\Documents\university\Bachelor - AUT\Final Project\Data Set\Nowrous\K-FOLD\After Feature Scoring K-fold"
os.makedirs(output_dir, exist_ok=True)

print(f"Scanning input directory for train_patient_*.csv files in: {input_dir}")

# --- Process each LOOCV train set ---
for filename in os.listdir(input_dir):
    if filename.startswith("train_patient_") and filename.endswith(".csv"):
        patient_id = filename.replace("train_", "").replace(".csv", "")
        train_file_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, f"feature_scores_{patient_id}.csv")

        print(f"\nProcessing {filename}...")

        df = pd.read_csv(train_file_path)
        print(f"Loaded shape: {df.shape}")

        # Prepare data
        feature_columns = df.columns[1:-1]  # Assuming 1st col = ID, last = label
        labels = df.iloc[:, -1]

        df[feature_columns] = df[feature_columns].apply(pd.to_numeric, errors="coerce")
        df[feature_columns] = df[feature_columns].fillna(df[feature_columns].mean())
        df[feature_columns] = df[feature_columns].fillna(0)

        if df[feature_columns].isnull().any().any():
            print(f"Warning: NaNs still present in {filename}")
            continue

        # --- Scoring ---
        f_values, _ = f_classif(df[feature_columns], labels)
        fisher_values = fisher_score(df[feature_columns].values, labels.values)
        mi_values = mutual_info_classif(
            df[feature_columns], labels, discrete_features="auto"
        )

        num_features = len(feature_columns)

        score_df = pd.DataFrame(
            {
                "Feature": feature_columns,
                "f_classif": f_values,
                "fisher": fisher_values,
                "mutual_info": mi_values,
            }
        )

        for method in ["f_classif", "fisher", "mutual_info"]:
            score_df[f"{method}_rank"] = score_df[method].rank(
                ascending=False, method="min"
            )
            score_df[f"{method}_vote"] = num_features - score_df[f"{method}_rank"]

        score_df["Vote_Score"] = score_df[
            [f"{m}_vote" for m in ["f_classif", "fisher", "mutual_info"]]
        ].sum(axis=1)
        score_df = score_df.sort_values(by="Vote_Score", ascending=False)

        feature_score_df = score_df[["Feature", "Vote_Score"]].copy()

        # --- Save result ---
        feature_score_df.to_csv(output_path, index=False)
        print(f"Saved feature scores to: {output_path}")
