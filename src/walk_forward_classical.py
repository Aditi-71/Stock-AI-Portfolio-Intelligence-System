import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

X = pd.read_parquet("data/processed/X_train_clean.parquet")

y = pd.read_parquet("data/processed/y_train_clean.parquet")["target"].astype(int)


print("\nDataset:")
print("Rows:", len(X))
print("Features:", X.shape[1])


tscv = TimeSeriesSplit(n_splits=5)


models = {
    "Logistic Regression": Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=300, max_depth=8, min_samples_leaf=10, random_state=42, n_jobs=-1
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42
    ),
    "XGBoost": XGBClassifier(
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
    ),
}


all_results = []


for model_name, model in models.items():
    print()
    print("=" * 60)
    print(model_name)
    print("=" * 60)

    fold_number = 1

    for train_index, val_index in tscv.split(X):
        X_train = X.iloc[train_index]
        X_val = X.iloc[val_index]

        y_train = y.iloc[train_index]
        y_val = y.iloc[val_index]

        model.fit(X_train, y_train)

        predictions = model.predict(X_val)

        probabilities = model.predict_proba(X_val)[:, 1]

        accuracy = accuracy_score(y_val, predictions)

        f1 = f1_score(y_val, predictions, zero_division=0)

        auc = roc_auc_score(y_val, probabilities)

        always_up_accuracy = y_val.mean()

        train_start = X_train.index.min()
        train_end = X_train.index.max()

        val_start = X_val.index.min()
        val_end = X_val.index.max()

        print(f"\nFold {fold_number}")

        print(f"Train: {train_start.date()} to {train_end.date()}")

        print(f"Validation: {val_start.date()} to {val_end.date()}")

        print(f"Accuracy: {accuracy:.4f}")

        print(f"Always-UP accuracy: {always_up_accuracy:.4f}")

        print(f"F1: {f1:.4f}")

        print(f"ROC-AUC: {auc:.4f}")

        all_results.append(
            {
                "model": model_name,
                "fold": fold_number,
                "accuracy": accuracy,
                "always_up_accuracy": always_up_accuracy,
                "f1": f1,
                "roc_auc": auc,
            }
        )

        fold_number += 1


results = pd.DataFrame(all_results)


summary = results.groupby("model").agg(
    {
        "accuracy": ["mean", "std"],
        "f1": ["mean", "std"],
        "roc_auc": ["mean", "std"],
        "always_up_accuracy": "mean",
    }
)


print()
print("=" * 60)
print("WALK-FORWARD SUMMARY")
print("=" * 60)

print(summary.round(4))


results.to_csv("data/processed/walk_forward_classical_results.csv", index=False)


print("\nSaved results to:")

print("data/processed/walk_forward_classical_results.csv")
