import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.inspection import permutation_importance
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


split_point = int(len(X_train) * 0.80)

X_inner_train = X_train.iloc[:split_point]

y_inner_train = y_train.iloc[:split_point]

X_inner_val = X_train.iloc[split_point:]

y_inner_val = y_train.iloc[split_point:]


print("\nInner training rows:")
print(len(X_inner_train))

print("\nInner validation rows:")
print(len(X_inner_val))


selection_model = GradientBoostingClassifier(
    n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42
)

print("\nTraining feature-selection model...")

selection_model.fit(X_inner_train, y_inner_train)


print("\nCalculating permutation importance...")

result = permutation_importance(
    selection_model,
    X_inner_val,
    y_inner_val,
    scoring="roc_auc",
    n_repeats=10,
    random_state=42,
    n_jobs=-1,
)


importance_df = pd.DataFrame(
    {
        "feature": X_train.columns,
        "importance": result.importances_mean,
        "std": result.importances_std,
    }
)

importance_df = importance_df.sort_values("importance", ascending=False)


selected_features = importance_df.head(30)["feature"].tolist()


print("\n===== SELECTED 30 FEATURES =====")

for feature in selected_features:
    print(feature)


final_model = GradientBoostingClassifier(
    n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42
)


print("\nTraining Gradient Boosting with 30 selected features...")

final_model.fit(X_train[selected_features], y_train)


predictions = final_model.predict(X_val[selected_features])

probabilities = final_model.predict_proba(X_val[selected_features])[:, 1]


accuracy = accuracy_score(y_val, predictions)

precision = precision_score(y_val, predictions, zero_division=0)

recall = recall_score(y_val, predictions, zero_division=0)

f1 = f1_score(y_val, predictions, zero_division=0)

roc_auc = roc_auc_score(y_val, probabilities)

cm = confusion_matrix(y_val, predictions)


print("\n===== VALIDATION RESULTS (30 FEATURES) =====")

print(f"Accuracy:  {accuracy:.4f}")

print(f"Precision: {precision:.4f}")

print(f"Recall:    {recall:.4f}")

print(f"F1 Score:  {f1:.4f}")

print(f"ROC-AUC:   {roc_auc:.4f}")

print("\nConfusion Matrix:")

print(cm)


pd.Series(selected_features, name="feature").to_csv(
    "data/processed/selected_features.csv", index=False
)

print("\nSelected features saved to:")

print("data/processed/selected_features.csv")
