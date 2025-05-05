import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

input_dir = r"C:\Users\Pouya\Documents\university\Bachelor - AUT\Final Project\Data Set\Nowrous\K-FOLD\After Univariate K-fold"
output_dir = r"C:\Users\Pouya\Documents\university\Bachelor - AUT\Final Project\Data Set\Nowrous\K-FOLD\After Model Based K-fold"

top_k = 20

os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(input_dir):
    if filename.endswith("_univariate_features.csv"):
        input_path = os.path.join(input_dir, filename)
        output_filename = filename.replace(
            "_univariate_features.csv", "_model_based.csv"
        )
        output_path = os.path.join(output_dir, output_filename)

        df = pd.read_csv(input_path)
        name_col = df.columns[0]
        label_col = df.columns[-1]

        X = df.drop(columns=[name_col, label_col])
        y = df[label_col]

        model = RandomForestClassifier(n_estimators=200, random_state=42)
        model.fit(X, y)

        importances = model.feature_importances_
        features_importances = list(zip(X.columns, importances))
        features_importances.sort(key=lambda x: x[1], reverse=True)

        top_features = [feat for feat, _ in features_importances[:top_k]]
        reduced_df = df[[name_col] + top_features + [label_col]]

        # Ensure "Window_Name" is the first column if it exists
        if "Window_Name" in reduced_df.columns:
            cols = ["Window_Name"] + [
                col for col in reduced_df.columns if col != "Window_Name"
            ]
            reduced_df = reduced_df[cols]
            
        reduced_df.to_csv(output_path, index=False)
        print(f"{filename}: top {top_k} features saved to {output_filename}")
