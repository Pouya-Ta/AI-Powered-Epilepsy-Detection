import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.preprocessing import StandardScaler

data_dir = r"C:\Users\Pouya\Documents\university\Bachelor - AUT\Final Project\Data Set\Nowrous\After Test-Modification"
New_dir = r"C:\Users\Pouya\Documents\university\Bachelor - AUT\Final Project\Data Set\Nowrous\KNN"
output_csv = os.path.join(New_dir, "loo_knn_results.csv")
plot_dir = os.path.join(New_dir, "K_vs_Accuracy_Plots")
report_dir = os.path.join(New_dir, "KNN_Reports")
os.makedirs(plot_dir, exist_ok=True)
os.makedirs(report_dir, exist_ok=True)

results = []

for filename in os.listdir(data_dir):
    if filename.startswith("final_train_") and filename.endswith(".csv"):
        patient_id = filename.replace("final_train_", "").replace(".csv", "")
        train_path = os.path.join(data_dir, filename)
        test_path = os.path.join(data_dir, f"final_test_{patient_id}.csv")

        if not os.path.exists(test_path):
            print(f"Test file missing for {patient_id}, skipping...")
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

        k_accuracies = []

        for k in range(1, 21):
            model = KNeighborsClassifier(n_neighbors=k)
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            acc = accuracy_score(y_test, y_pred)
            k_accuracies.append((k, acc))

        best_k, best_acc = max(k_accuracies, key=lambda x: x[1])
        best_model = KNeighborsClassifier(n_neighbors=best_k)
        best_model.fit(X_train_scaled, y_train)

        y_pred_test = best_model.predict(X_test_scaled)
        y_pred_train = best_model.predict(X_train_scaled)

        test_accuracy = accuracy_score(y_test, y_pred_test)
        train_accuracy = accuracy_score(y_train, y_pred_train)
        precision = precision_score(y_test, y_pred_test, pos_label=1, zero_division=0)
        recall = recall_score(y_test, y_pred_test, pos_label=1, zero_division=0)
        f1 = f1_score(y_test, y_pred_test, pos_label=1, zero_division=0)
        report = classification_report(
            y_test, y_pred_test, target_names=["Ictal (-1)", "Preictal (1)"]
        )

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred_test, labels=[-1, 1])
        plt.figure(figsize=(5, 4))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Ictal (-1)", "Preictal (1)"],
            yticklabels=["Ictal (-1)", "Preictal (1)"],
        )
        plt.title(f"Confusion Matrix: {patient_id}")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"confusion_matrix_{patient_id}.png"))
        plt.close()

        # Plot k vs accuracy
        plt.figure()
        ks, accs = zip(*k_accuracies)
        plt.plot(ks, accs, marker="o")
        plt.xticks(ks)
        plt.title(f"K vs Accuracy for {patient_id}")
        plt.xlabel("K (Number of Neighbors)")
        plt.ylabel("Accuracy")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"k_vs_accuracy_{patient_id}.png"))
        plt.close()

        # Save classification report to file
        report_path = os.path.join(report_dir, f"report_{patient_id}.txt")
        with open(report_path, "w") as f:
            f.write(f"Patient ID: {patient_id}\n")
            f.write(f"Best K: {best_k}\n")
            f.write(f"Train Accuracy: {train_accuracy:.4f}\n")
            f.write(f"Test Accuracy: {test_accuracy:.4f}\n")
            f.write(
                f"Overfitting Gap (Train - Test): {train_accuracy - test_accuracy:.4f}\n"
            )
            f.write(f"Feature Count: {X_train.shape[1]}\n")
            f.write("\nClassification Report:\n")
            f.write(report)

        results.append(
            {
                "Patient": patient_id,
                "Best_k": best_k,
                "Accuracy": test_accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1_Score": f1,
                "Train_Accuracy": train_accuracy,
                "Overfitting_Gap": train_accuracy - test_accuracy,
            }
        )

        print(
            f"Done: {patient_id} | k={best_k} | Test Acc={test_accuracy:.4f} | Overfit={train_accuracy - test_accuracy:.2f}"
        )

# Save master CSV
results_df = pd.DataFrame(results)
results_df = results_df.sort_values("Patient")
results_df.to_csv(output_csv, index=False)

print(f"\nAll results saved to:\n→ {output_csv}\n→ {report_dir}")