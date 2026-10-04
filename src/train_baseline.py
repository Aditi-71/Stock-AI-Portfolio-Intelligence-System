from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

data_dir = Path("data/processed")

X_train = pd.read_parquet(data_dir / "X_train_scaled.parquet")
y_train = pd.read_parquet(data_dir / "y_train_scaled.parquet")["target"]

X_val = pd.read_parquet(data_dir / "X_val_scaled.parquet")
y_val = pd.read_parquet(data_dir / "y_val_scaled.parquet")["target"]

X_test = pd.read_parquet(data_dir / "X_test_scaled.parquet")
y_test = pd.read_parquet(data_dir / "y_test_scaled.parquet")["target"]


model = LogisticRegression(max_iter=1000, random_state=42)


print("Training Logistic Regression...")

model.fit(X_train, y_train)


val_predictions = model.predict(X_val)

val_probabilities = model.predict_proba(X_val)[:, 1]


accuracy = accuracy_score(y_val, val_predictions)

precision = precision_score(y_val, val_predictions)

recall = recall_score(y_val, val_predictions)

f1 = f1_score(y_val, val_predictions)

roc_auc = roc_auc_score(y_val, val_probabilities)

cm = confusion_matrix(y_val, val_predictions)


print()
print("===== VALIDATION RESULTS =====")

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"ROC-AUC:   {roc_auc:.4f}")

print()
print("Confusion Matrix:")
print(cm)


test_predictions = model.predict(X_test)

test_probabilities = model.predict_proba(X_test)[:, 1]


test_accuracy = accuracy_score(y_test, test_predictions)

test_precision = precision_score(y_test, test_predictions)

test_recall = recall_score(y_test, test_predictions)

test_f1 = f1_score(y_test, test_predictions)

test_roc_auc = roc_auc_score(y_test, test_probabilities)


print()
print("===== TEST RESULTS =====")

print(f"Accuracy:  {test_accuracy:.4f}")
print(f"Precision: {test_precision:.4f}")
print(f"Recall:    {test_recall:.4f}")
print(f"F1 Score:  {test_f1:.4f}")
print(f"ROC-AUC:   {test_roc_auc:.4f}")
