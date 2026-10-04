from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

data_dir = Path("data/processed")

X_train = pd.read_parquet(data_dir / "X_train_clean.parquet")

y_train = pd.read_parquet(data_dir / "y_train_clean.parquet")["target"].astype(int)

X_val = pd.read_parquet(data_dir / "X_val_clean.parquet")

y_val = pd.read_parquet(data_dir / "y_val_clean.parquet")["target"].astype(int)

X_test = pd.read_parquet(data_dir / "X_test_clean.parquet")

y_test = pd.read_parquet(data_dir / "y_test_clean.parquet")["target"].astype(int)


model = RandomForestClassifier(
    n_estimators=300, max_depth=8, min_samples_leaf=10, random_state=42, n_jobs=-1
)


print("Training Random Forest...")

model.fit(X_train, y_train)


val_predictions = model.predict(X_val)

val_probabilities = model.predict_proba(X_val)[:, 1]


print()
print("===== VALIDATION RESULTS =====")

print(f"Accuracy:  {accuracy_score(y_val, val_predictions):.4f}")

print(f"Precision: {precision_score(y_val, val_predictions):.4f}")

print(f"Recall:    {recall_score(y_val, val_predictions):.4f}")

print(f"F1 Score:  {f1_score(y_val, val_predictions):.4f}")

print(f"ROC-AUC:   {roc_auc_score(y_val, val_probabilities):.4f}")

print()
print("Confusion Matrix:")

print(confusion_matrix(y_val, val_predictions))


val_prediction_df = pd.DataFrame(
    {"actual": y_val.values, "prediction": val_predictions, "probability_up": val_probabilities},
    index=X_val.index,
)

val_prediction_df.index.name = "Date"

val_output = data_dir / "random_forest_val_predictions.parquet"

val_prediction_df.to_parquet(val_output)


test_predictions = model.predict(X_test)

test_probabilities = model.predict_proba(X_test)[:, 1]


print()
print("===== TEST RESULTS =====")

print(f"Accuracy:  {accuracy_score(y_test, test_predictions):.4f}")

print(f"Precision: {precision_score(y_test, test_predictions):.4f}")

print(f"Recall:    {recall_score(y_test, test_predictions):.4f}")

print(f"F1 Score:  {f1_score(y_test, test_predictions):.4f}")

print(f"ROC-AUC:   {roc_auc_score(y_test, test_probabilities):.4f}")

print()
print("Confusion Matrix:")

print(confusion_matrix(y_test, test_predictions))


test_prediction_df = pd.DataFrame(
    {"actual": y_test.values, "prediction": test_predictions, "probability_up": test_probabilities},
    index=X_test.index,
)

test_prediction_df.index.name = "Date"

test_output = data_dir / "random_forest_test_predictions.parquet"

test_prediction_df.to_parquet(test_output)


print()
print("Prediction files saved:")

print(val_output)
print(test_output)
