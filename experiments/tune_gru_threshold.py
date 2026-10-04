import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

model = tf.keras.models.load_model("data/processed/gru_model.keras")

X_val = np.load("data/processed/X_val_seq.npy").astype("float32")

y_val = np.load("data/processed/y_val_seq.npy").astype(int)


probabilities = model.predict(X_val, verbose=0).ravel()


results = []

thresholds = np.arange(0.30, 0.71, 0.01)


for threshold in thresholds:
    predictions = (probabilities >= threshold).astype(int)

    accuracy = accuracy_score(y_val, predictions)

    balanced_accuracy = balanced_accuracy_score(y_val, predictions)

    precision = precision_score(y_val, predictions, zero_division=0)

    recall = recall_score(y_val, predictions, zero_division=0)

    f1 = f1_score(y_val, predictions, zero_division=0)

    results.append(
        {
            "threshold": threshold,
            "accuracy": accuracy,
            "balanced_accuracy": balanced_accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    )


best_result = max(results, key=lambda x: x["balanced_accuracy"])


print("\n===== BEST THRESHOLD =====")

print(f"Threshold: {best_result['threshold']:.2f}")

print(f"Accuracy: {best_result['accuracy']:.4f}")

print(f"Balanced Accuracy: {best_result['balanced_accuracy']:.4f}")

print(f"Precision: {best_result['precision']:.4f}")

print(f"Recall: {best_result['recall']:.4f}")

print(f"F1: {best_result['f1']:.4f}")


best_predictions = (probabilities >= best_result["threshold"]).astype(int)


cm = confusion_matrix(y_val, best_predictions)


print("\nConfusion Matrix:")

print(cm)


print("\n===== TOP 10 THRESHOLDS =====")

sorted_results = sorted(results, key=lambda x: x["balanced_accuracy"], reverse=True)


for result in sorted_results[:10]:
    print(
        f"Threshold "
        f"{result['threshold']:.2f}"
        f" | Balanced Acc "
        f"{result['balanced_accuracy']:.4f}"
        f" | Accuracy "
        f"{result['accuracy']:.4f}"
        f" | F1 "
        f"{result['f1']:.4f}"
    )
