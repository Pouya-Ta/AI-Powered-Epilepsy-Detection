import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
from sklearn.svm import SVC
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


data_dir = r"C:\Users\Pouya\Documents\university\Bachelor - AUT\Final Project\Data Set\Nowrous\After Test-Modification"
New_dir = r"C:\Users\Pouya\Documents\university\Bachelor - AUT\Final Project\Data Set\Nowrous\SVM"
output_csv = os.path.join(New_dir, "svm_summary_results.csv")
report_dir = os.path.join(New_dir, "SVM_Reports")
conf_matrix_dir = os.path.join(New_dir, "SVM_Plots", "Confusion_Matrix")
acc_plot_dir = os.path.join(New_dir, "SVM_Plots", "Accuracy_Comparison")

os.makedirs(report_dir, exist_ok=True)
os.makedirs(conf_matrix_dir, exist_ok=True)
os.makedirs(acc_plot_dir, exist_ok=True)


results = []

kernels = ["linear", "poly", "rbf"]


def plot_svm_decision_boundary_2d(X, y, model, patient_id, kernel, save_path):
    h = 0.02  # step size for mesh grid
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(7, 6))
    cmap_light = ListedColormap(["#FFBBBB", "#BBBBFF"])
    plt.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.5)

    # Plot data points
    plt.scatter(
        X[y == -1, 0],
        X[y == -1, 1],
        color="red",
        label="Ictal (-1)",
        edgecolor="white",
        s=30,
    )
    plt.scatter(
        X[y == 1, 0],
        X[y == 1, 1],
        color="blue",
        label="Preictal (1)",
        edgecolor="white",
        s=30,
    )

    # Plot support vectors
    if hasattr(model, "support_vectors_"):
        plt.scatter(
            model.support_vectors_[:, 0],
            model.support_vectors_[:, 1],
            s=100,
            facecolors="none",
            edgecolors="black",
            label="Support Vectors",
        )

    plt.title(f"SVM Decision Boundary (2D PCA) - {patient_id} ({kernel})")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


for filename in os.listdir(data_dir):
    if filename.startswith("final_train_") and filename.endswith(".csv"):
        patient_id = filename.replace("final_train_", "").replace(".csv", "")
        train_path = os.path.join(data_dir, filename)
        test_path = os.path.join(data_dir, f"final_test_{patient_id}.csv")

        if not os.path.exists(test_path):
            print(f"Missing test file for {patient_id}, skipping.")
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

        for kernel in kernels:
            model = SVC(kernel=kernel, probability=False, random_state=42)
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
            overfit_gap = train_acc - test_acc
            report = classification_report(
                y_test, y_pred_test, target_names=["Ictal (-1)", "Preictal (1)"]
            )

            results.append(
                {
                    "Patient": patient_id,
                    "Kernel": kernel,
                    "Accuracy": test_acc,
                    "Precision": precision,
                    "Recall": recall,
                    "F1_Score": f1,
                    "Train_Accuracy": train_acc,
                    "Overfitting_Gap": overfit_gap,
                    "Features_Used": X_train.shape[1],
                }
            )

            # Save text report
            report_file = os.path.join(report_dir, f"report_{patient_id}_{kernel}.txt")
            with open(report_file, "w") as f:
                f.write(f"Patient: {patient_id}\nKernel: {kernel}\n")
                f.write(f"Train Accuracy: {train_acc:.4f}\n")
                f.write(f"Test Accuracy: {test_acc:.4f}\n")
                f.write(f"Overfitting Gap: {overfit_gap:.4f}\n")
                f.write("\nClassification Report:\n")
                f.write(report)

            # Save confusion matrix
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
            plt.title(f"Confusion Matrix: {patient_id} ({kernel})")
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.tight_layout()
            plt.savefig(
                os.path.join(
                    conf_matrix_dir, f"confusion_matrix_{patient_id}_{kernel}.png"
                )
            )
            plt.close()

            # Save accuracy bar plot
            plt.figure()
            plt.bar(
                ["Train", "Test"], [train_acc, test_acc], color=["skyblue", "salmon"]
            )
            plt.ylim(0, 1)
            plt.title(f"Train vs Test Accuracy: {patient_id} ({kernel})")
            plt.ylabel("Accuracy")
            plt.tight_layout()
            plt.savefig(
                os.path.join(
                    acc_plot_dir, f"accuracy_comparison_{patient_id}_{kernel}.png"
                )
            )
            plt.close()

            print(f"Finished {patient_id} | kernel={kernel} | test acc={test_acc:.4f}")
            # 2D PCA Decision Boundary Plot
            if X_train.shape[1] >= 2:
                decision_plot_dir = os.path.join(
                    New_dir, "SVM_Plots", "Decision_Boundary_2D"
                )
                os.makedirs(decision_plot_dir, exist_ok=True)

                # Reduce to 2D via PCA
                pca_2d = PCA(n_components=2)
                X_train_2d = pca_2d.fit_transform(X_train_scaled)

                # Train visual SVM model
                model_2d = SVC(kernel=kernel)
                model_2d.fit(X_train_2d, y_train)

                save_path = os.path.join(
                    decision_plot_dir, f"decision_boundary_{patient_id}_{kernel}.png"
                )
                plot_svm_decision_boundary_2d(
                    X_train_2d, y_train.values, model_2d, patient_id, kernel, save_path
                )


# Save CSV summary
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(["Patient", "Kernel"])
results_df.to_csv(output_csv, index=False)

print(f"\n✅ All done. Results saved to:\n→ {output_csv}\n→ {report_dir}")