import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# === Directories ===
data_dir = r"C:\Data" # Input folder (the data after test modification)
New_dir = r"C:\DT_LOO" # The output folder (which contains some other folders)
report_dir = os.path.join(New_dir, "DecisionTree_Reports")
cm_plot_dir = os.path.join(New_dir, "DecisionTree_Plots", "Confusion_Matrix")
acc_plot_dir = os.path.join(New_dir, "DecisionTree_Plots", "Accuracy_Comparison")
tree_plot_dir = os.path.join(New_dir, "DecisionTree_Plots", "Tree_Visualization")
output_csv = os.path.join(New_dir, "decision_tree_summary_results.csv")

os.makedirs(report_dir, exist_ok=True)
os.makedirs(cm_plot_dir, exist_ok=True)
os.makedirs(acc_plot_dir, exist_ok=True)
os.makedirs(tree_plot_dir, exist_ok=True)

results = []

# === Main loop through patients and depths ===
depth_values = [5, 10, 15, 20, 25, 30, 35, 40]

for filename in os.listdir(data_dir):
    if filename.startswith("final_train_") and filename.endswith(".csv"):
        patient_id = filename.replace("final_train_", "").replace(".csv", "")
        train_path = os.path.join(data_dir, filename)
        test_path = os.path.join(data_dir, f"final_test_{patient_id}.csv")

        if not os.path.exists(test_path):
            print(f"Missing test for {patient_id}, skipping.")
            continue

        df_train = pd.read_csv(train_path)
        df_test = pd.read_csv(test_path)

        label_col = df_train.columns[-1]
        name_col = df_train.columns[0]

        df_train = df_train[df_train[label_col] != 0]
        df_test = df_test[df_test[label_col] != 0]

        X_train = df_train.drop(columns=[name_col, label_col])
        y_train = df_train[label_col]
        X_test = df_test.drop(columns=[name_col, label_col])
        y_test = df_test[label_col]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        for depth in depth_values:
            model = DecisionTreeClassifier(random_state=42, max_depth=depth)
            model.fit(X_train_scaled, y_train)

            y_pred_test = model.predict(X_test_scaled)
            y_pred_train = model.predict(X_train_scaled)

            test_acc = accuracy_score(y_test, y_pred_test)
            train_acc = accuracy_score(y_train, y_pred_train)
            precision = precision_score(
                y_test, y_pred_test, pos_label=1, zero_division=0
            )
            recall = recall_score(y_test, y_pred_test, pos_label=1, zero_division=0)
            f1 = f1_score(y_test, y_pred_test, pos_label=1, zero_division=0)
            overfit = train_acc - test_acc
            report = classification_report(
                y_test, y_pred_test, target_names=["Preictal (-1)", "Interctal (1)"]
            )

            results.append(
                {
                    "Patient": patient_id,
                    "Depth": depth,
                    "Accuracy": test_acc,
                    "Precision": precision,
                    "Recall": recall,
                    "F1_Score": f1,
                    "Train_Accuracy": train_acc,
                    "Overfitting_Gap": overfit,
                    "Features_Used": X_train.shape[1],
                }
            )

            # Save report
            report_path = os.path.join(
                report_dir, f"report_{patient_id}_depth{depth}.txt"
            )
            with open(report_path, "w") as f:
                f.write(
                    f"Patient: {patient_id}\nModel: Decision Tree\nDepth: {depth}\n"
                )
                f.write(f"Train Accuracy: {train_acc:.4f}\n")
                f.write(f"Test Accuracy: {test_acc:.4f}\n")
                f.write(f"Overfitting Gap: {overfit:.4f}\n\n")
                f.write("Classification Report:\n")
                f.write(report)

            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred_test, labels=[-1, 1])
            plt.figure(figsize=(5, 4))
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Oranges",
                xticklabels=["Preictal (-1)", "Interctal (1)"],
                yticklabels=["Preictal (-1)", "Interctal (1)"],
            )
            plt.title(f"Confusion Matrix: {patient_id} | Depth {depth}")
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.tight_layout()
            plt.savefig(
                os.path.join(
                    cm_plot_dir, f"confusion_matrix_{patient_id}_depth{depth}.png"
                )
            )
            plt.close()

            # Accuracy bar chart
            plt.figure()
            plt.bar(["Train", "Test"], [train_acc, test_acc], color=["green", "red"])
            plt.ylim(0, 1)
            plt.ylabel("Accuracy")
            plt.title(f"Accuracy Comparison - {patient_id} | Depth {depth}")
            plt.tight_layout()
            plt.savefig(
                os.path.join(acc_plot_dir, f"accuracy_{patient_id}_depth{depth}.png")
            )
            plt.close()

            # Tree visualization
            plt.figure(figsize=(16, 6))
            plot_tree(
                model,
                filled=True,
                feature_names=X_train.columns,
                class_names=["-1", "1"],
                rounded=True,
            )
            plt.title(f"Decision Tree - {patient_id} | Depth {depth}")
            plt.savefig(
                os.path.join(tree_plot_dir, f"tree_{patient_id}_depth{depth}.png")
            )
            plt.close()

            print(
                f" Finished {patient_id} | Depth {depth}: Test Acc = {test_acc:.2f}, F1 = {f1:.2f}"
            )


# === Save summary CSV ===
df_results = pd.DataFrame(results)
df_results = df_results.sort_values("Patient")
df_results.to_csv(output_csv, index=False)
print(f"\n Summary saved to: {output_csv}")
