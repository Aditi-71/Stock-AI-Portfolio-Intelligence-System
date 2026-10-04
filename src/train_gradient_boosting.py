import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

X_train = pd.read_parquet("data/processed/X_train_clean.parquet")

y_train = pd.read_parquet("data/processed/y_train_clean.parquet")["target"].astype(int)

X_val = pd.read_parquet("data/processed/X_val_clean.parquet")

y_val = pd.read_parquet("data/processed/y_val_clean.parquet")["target"].astype(int)


model = GradientBoostingClassifier(
    n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42
)


print("\nTraining Gradient Boosting...")

model.fit(X_train, y_train)


predictions = model.predict(X_val)

probabilities = model.predict_proba(X_val)[:, 1]


accuracy = accuracy_score(y_val, predictions)

precision = precision_score(y_val, predictions, zero_division=0)

recall = recall_score(y_val, predictions, zero_division=0)

f1 = f1_score(y_val, predictions, zero_division=0)

roc_auc = roc_auc_score(y_val, probabilities)

cm = confusion_matrix(y_val, predictions)


print("\n===== VALIDATION RESULTS =====")

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"ROC-AUC:   {roc_auc:.4f}")

print("\nConfusion Matrix:")
print(cm)


feature_importance = pd.DataFrame(
    {"feature": X_train.columns, "importance": model.feature_importances_}
)

feature_importance = feature_importance.sort_values(by="importance", ascending=False)

print("\n===== TOP 30 IMPORTANT FEATURES =====")

print(feature_importance.head(30).to_string(index=False))
