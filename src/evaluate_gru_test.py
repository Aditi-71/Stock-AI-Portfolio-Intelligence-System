import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

THRESHOLD = 0.39


model = tf.keras.models.load_model("data/processed/gru_model.keras")


X_test = np.load("data/processed/X_test_seq.npy").astype("float32")

y_test = np.load("data/processed/y_test_seq.npy").astype(int)


print("\nTest shape:")
print(X_test.shape)


probabilities = model.predict(X_test, verbose=0).ravel()


predictions = (probabilities >= THRESHOLD).astype(int)


accuracy = accuracy_score(y_test, predictions)

balanced_accuracy = balanced_accuracy_score(y_test, predictions)

precision = precision_score(y_test, predictions, zero_division=0)

recall = recall_score(y_test, predictions, zero_division=0)

f1 = f1_score(y_test, predictions, zero_division=0)

roc_auc = roc_auc_score(y_test, probabilities)

brier = brier_score_loss(y_test, probabilities)

cm = confusion_matrix(y_test, predictions)


always_up_accuracy = y_test.mean()


print("\n===== FINAL GRU TEST RESULTS =====")

print(f"Locked threshold:   {THRESHOLD:.2f}")

print(f"Accuracy:           {accuracy:.4f}")

print(f"Balanced Accuracy:  {balanced_accuracy:.4f}")

print(f"Precision:          {precision:.4f}")

print(f"Recall:             {recall:.4f}")

print(f"F1 Score:           {f1:.4f}")

print(f"ROC-AUC:            {roc_auc:.4f}")

print(f"Brier Score:        {brier:.4f}")

print(f"Always-UP Accuracy: {always_up_accuracy:.4f}")


print("\nConfusion Matrix:")

print(cm)


unique, counts = np.unique(predictions, return_counts=True)


print("\nPrediction counts:")

for label, count in zip(unique, counts):
    print(f"Class {label}: {count}")


print("\nProbability statistics:")

print(f"Minimum: {probabilities.min():.4f}")

print(f"Mean:    {probabilities.mean():.4f}")

print(f"Maximum: {probabilities.max():.4f}")


print("\nActual test distribution:")

print(f"DOWN: {(y_test == 0).sum()}")

print(f"UP:   {(y_test == 1).sum()}")
