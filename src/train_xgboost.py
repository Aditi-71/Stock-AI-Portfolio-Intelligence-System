import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

X_train = pd.read_parquet("data/processed/X_train_clean.parquet")

y_train = pd.read_parquet("data/processed/y_train_clean.parquet")["target"].astype(int)

X_val = pd.read_parquet("data/processed/X_val_clean.parquet")

y_val = pd.read_parquet("data/processed/y_val_clean.parquet")["target"].astype(int)


model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.03,
    max_depth=3,
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1,
)


print("\nTraining XGBoost...")

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


importance_df = pd.DataFrame({"feature": X_train.columns, "importance": model.feature_importances_})

importance_df = importance_df.sort_values(by="importance", ascending=False)

print("\n===== TOP 30 IMPORTANT FEATURES =====")

print(importance_df.head(30).to_string(index=False))
