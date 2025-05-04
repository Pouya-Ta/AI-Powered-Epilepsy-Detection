import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from sklearn.preprocessing import StandardScaler, MinMaxScaler

data_dir = r"C:\Users\Pouya\Documents\university\Bachelor - AUT\Final Project\Data Set\Nowrous\After Test-Modification"
New_dir = r"C:\Users\Pouya\Documents\university\Bachelor - AUT\Final Project\Data Set\Nowrous\NB"
output_csv = os.path.join(New_dir, "naive_bayes_summary_results.csv")
report_dir = os.path.join(New_dir, "NB_Reports")
conf_matrix_dir = os.path.join(New_dir, "NB_Plots", "Confusion_Matrix")
acc_plot_dir = os.path.join(New_dir, "NB_Plots", "Accuracy_Comparison")

os.makedirs(report_dir, exist_ok=True)
os.makedirs(conf_matrix_dir, exist_ok=True)
os.makedirs(acc_plot_dir, exist_ok=True)

results = []

models = {
    "gaussian": GaussianNB(),
    "multinomial": MultinomialNB(),
    "bernoulli": BernoulliNB(),
}

for filename in os.listdir(data_dir):
    if filename.startswith("final_train_") and filename.endswith(".csv"):
        patient_id = filename.replace("final_train_", "").replace(".csv", "")
        train_path = os.path.join(data_dir, filename)
        test_path = os.path.join(data_dir, f"final_test_{patient_id}.csv")

        if not os.path.exists(test_path):
            print(f"Test file missing for {patient_id}, skipping.")
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

        for model_name, model in models.items():
            # GaussianNB → StandardScaler | Multinomial/Bernoulli → MinMax [0,1]
            if model_name == "gaussian":
                scaler = StandardScaler()
            else:
                scaler = MinMaxScaler()

            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            model.fit(X_train_scaled, y_train)

            y_pred_test = model.predict(X_test_scaled)
            y_pred_train = model.predict(X_train_scaled)

            test_acc = accuracy_score(y_test, y_pred_test)
            train_acc = accuracy_score(y_train, y_pred_train)
            from sklearn.metrics import precision_score, recall_score, f1_score

            # Fix UndefinedMetricWarning by setting zero_division=0
            precision = precision_score(y_test, y_pred_test, pos_label=1, zero_division=0)
            recall = recall_score(y_test, y_pred_test, pos_label=1, zero_division=0)
            f1 = f1_score(y_test, y_pred_test, pos_label=1, zero_division=0)
            
            overfit_gap = train_acc - test_acc
            report = classification_report(
                y_test, y_pred_test, target_names=["Ictal (-1)", "Preictal (1)"]
            )

            results.append(
                {
                    "Patient": patient_id,
                    "Model": model_name,
                    "Accuracy": test_acc,
                    "Precision": precision,
                    "Recall": recall,
                    "F1_Score": f1,
                    "Train_Accuracy": train_acc,
                    "Overfitting_Gap": overfit_gap,
                    "Features_Used": X_train.shape[1],
                }
            )

            # Save classification report
            report_file = os.path.join(
                report_dir, f"report_{patient_id}_{model_name}.txt"
            )
            with open(report_file, "w") as f:
                f.write(f"Patient: {patient_id}\nNaive Bayes Type: {model_name}\n")
                f.write(f"Train Accuracy: {train_acc:.4f}\n")
                f.write(f"Test Accuracy: {test_acc:.4f}\n")
                f.write(f"Overfitting Gap: {overfit_gap:.4f}\n")
                f.write("\nClassification Report:\n")
                f.write(report)

            # Confusion Matrix
            cm = confusion_matrix(y_test, y_pred_test, labels=[-1, 1])
            plt.figure(figsize=(5, 4))
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Purples",
                xticklabels=["Ictal (-1)", "Preictal (1)"],
                yticklabels=["Ictal (-1)", "Preictal (1)"],
            )
            plt.title(f"Confusion Matrix: {patient_id} ({model_name})")
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.tight_layout()
            plt.savefig(
                os.path.join(
                    conf_matrix_dir, f"confusion_matrix_{patient_id}_{model_name}.png"
                )
            )
            plt.close()

            # Accuracy Bar Chart
            plt.figure()
            plt.bar(
                ["Train", "Test"], [train_acc, test_acc], color=["lightgreen", "tomato"]
            )
            plt.ylim(0, 1)
            plt.title(f"Train vs Test Accuracy: {patient_id} ({model_name})")
            plt.ylabel("Accuracy")
            plt.tight_layout()
            plt.savefig(
                os.path.join(
                    acc_plot_dir, f"accuracy_comparison_{patient_id}_{model_name}.png"
                )
            )
            plt.close()

            print(
                f"Done NB: {patient_id} | model={model_name} | test_acc={test_acc:.4f}"
            )

# Save summary CSV
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(["Patient", "Model"])
results_df.to_csv(output_csv, index=False)

print(
    f"\n✅ All Naive Bayes evaluations complete. Results saved to:\n→ {output_csv}\n→ {report_dir}"
)
