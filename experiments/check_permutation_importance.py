import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.inspection import permutation_importance

X_train = pd.read_parquet("data/processed/X_train_clean.parquet")

y_train = pd.read_parquet("data/processed/y_train_clean.parquet")["target"].astype(int)

X_val = pd.read_parquet("data/processed/X_val_clean.parquet")

y_val = pd.read_parquet("data/processed/y_val_clean.parquet")["target"].astype(int)


model = GradientBoostingClassifier(
    n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42
)

print("\nTraining Gradient Boosting...")

model.fit(X_train, y_train)


print("\nCalculating permutation importance...")

result = permutation_importance(
    model, X_val, y_val, scoring="roc_auc", n_repeats=10, random_state=42, n_jobs=-1
)


importance_df = pd.DataFrame(
    {"feature": X_val.columns, "importance": result.importances_mean, "std": result.importances_std}
)

importance_df = importance_df.sort_values(by="importance", ascending=False)


print("\n===== TOP 30 PERMUTATION FEATURES =====")

print(importance_df.head(30).to_string(index=False))

print("\nFeatures with positive importance:")

print((importance_df["importance"] > 0).sum())

print(f"\nTotal features: {len(importance_df)}")
