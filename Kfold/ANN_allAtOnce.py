import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import plot_model

# === Config ===
data_dir = r"C:\Users\Pouya\Documents\university\Bachelor - AUT\Final Project\Data Set\Pre\code\Data"
New_dir = r"C:\Users\Pouya\Documents\university\Bachelor - AUT\Final Project\Data Set\Pre\code\ANN_LOO"
report_dir = os.path.join(New_dir, "ANN_TF_Reports")
cm_plot_dir = os.path.join(New_dir, "ANN_TF_Plots", "Confusion_Matrix")
acc_plot_dir = os.path.join(New_dir, "ANN_TF_Plots", "Accuracy_Comparison")
loss_plot_dir = os.path.join(New_dir, "ANN_TF_Plots", "Loss_Curve")
output_csv = os.path.join(New_dir, "ann_tf_summary_results.csv")

os.makedirs(report_dir, exist_ok=True)
os.makedirs(cm_plot_dir, exist_ok=True)
os.makedirs(acc_plot_dir, exist_ok=True)
os.makedirs(loss_plot_dir, exist_ok=True)

results = []
activations = ["relu", "sigmoid", "tanh", "softmax"]

# === Loop through patients ===
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
        y_train = df_train[label_col].replace(-1, 0).values  # convert labels to 0 and 1
        X_test = df_test.drop(columns=[name_col, label_col])
        y_test = df_test[label_col].replace(-1, 0).values

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        for act_fn in activations:
            model = Sequential(
                [
                    Dense(64, activation=act_fn, input_shape=(X_train.shape[1],)),
                    Dense(32, activation=act_fn),
                    Dense(1, activation="sigmoid"),
                ]
            )

            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss="binary_crossentropy",
                metrics=["accuracy"],
            )

            es = EarlyStopping(
                patience=10, restore_best_weights=True, monitor="val_loss"
            )

            history = model.fit(
                X_train_scaled,
                y_train,
                validation_split=0.2,
                epochs=100,
                batch_size=32,
                callbacks=[es],
                verbose=0,
            )

            y_pred_prob = model.predict(X_test_scaled).flatten()
            y_pred_test = (y_pred_prob > 0.5).astype(int)
            y_pred_train = (model.predict(X_train_scaled).flatten() > 0.5).astype(int)

            train_acc = accuracy_score(y_train, y_pred_train)
            test_acc = accuracy_score(y_test, y_pred_test)
            precision = precision_score(y_test, y_pred_test, zero_division=0)
            recall = recall_score(y_test, y_pred_test, zero_division=0)
            f1 = f1_score(y_test, y_pred_test, zero_division=0)
            overfit_gap = train_acc - test_acc
            report = classification_report(
                y_test, y_pred_test, target_names=["Preictal (-1)", "Ictal (1)"]
            )

            results.append(
                {
                    "Patient": patient_id,
                    "Activation": act_fn,
                    "Accuracy": test_acc,
                    "Precision": precision,
                    "Recall": recall,
                    "F1_Score": f1,
                    "Train_Accuracy": train_acc,
                    "Overfitting_Gap": overfit_gap,
                    "Features_Used": X_train.shape[1],
                }
            )

            # === Save classification report ===
            report_path = os.path.join(report_dir, f"report_{patient_id}_{act_fn}.txt")
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(f"Patient: {patient_id}\nActivation: {act_fn}\n")
                f.write(f"Train Accuracy: {train_acc:.4f}\n")
                f.write(f"Test Accuracy: {test_acc:.4f}\n")
                f.write(f"Overfitting Gap: {overfit_gap:.4f}\n")
                f.write(f"Final Training Loss: {history.history['loss'][-1]:.4f}\n")
                f.write("\nModel Architecture:\n")
                model.summary(print_fn=lambda x: f.write(x + "\n"))
                f.write("\nClassification Report:\n")
                f.write(report)

            # === Loss curve ===
            plt.figure()
            plt.plot(history.history["loss"], label="Train Loss")
            plt.plot(history.history["val_loss"], label="Val Loss")
            plt.title(f"Loss Curve - {patient_id} ({act_fn})")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.tight_layout()
            plt.savefig(
                os.path.join(loss_plot_dir, f"loss_curve_{patient_id}_{act_fn}.png")
            )
            plt.close()

            # === Accuracy bar plot ===
            plt.figure()
            plt.bar(["Train", "Test"], [train_acc, test_acc], color=["green", "orange"])
            plt.ylim(0, 1)
            plt.ylabel("Accuracy")
            plt.title(f"Train vs Test Accuracy - {patient_id} ({act_fn})")
            plt.tight_layout()
            plt.savefig(
                os.path.join(acc_plot_dir, f"accuracy_{patient_id}_{act_fn}.png")
            )
            plt.close()

            # === Confusion matrix ===
            cm = confusion_matrix(y_test, y_pred_test, labels=[0, 1])
            plt.figure(figsize=(5, 4))
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=["Preictal (-1)", "Interctal (1)"],
                yticklabels=["Preictal (-1)", "Interctal (1)"],
            )
            plt.title(f"Confusion Matrix - {patient_id} ({act_fn})")
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.tight_layout()
            plt.savefig(
                os.path.join(cm_plot_dir, f"confusion_matrix_{patient_id}_{act_fn}.png")
            )
            plt.close()

            print(
                f"✅ {patient_id} | Activation={act_fn} | Test Acc={test_acc:.2f} | F1={f1:.2f}"
            )

# === Save all results ===
pd.DataFrame(results).sort_values(["Patient", "Activation"]).to_csv(
    output_csv, index=False
)
print(f"\n📊 ANN TensorFlow results saved to: {output_csv}")
