import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

data = pd.read_parquet("data/processed/model_data_with_sentiment.parquet")

data = data.sort_index()


print("\nDataset shape:")
print(data.shape)

print("\nDate range:")
print(data.index.min(), "to", data.index.max())


y = data["target"].astype(int)


sentiment_features = [
    "AAPL_sentiment_mean",
    "AAPL_sentiment_std",
    "AAPL_positive_ratio",
    "AAPL_negative_ratio",
    "AAPL_neutral_ratio",
    "AAPL_sentiment_balance",
    "AAPL_article_count",
    "AAPL_log_article_count",
    "AAPL_has_news",
]


base_features = [
    column for column in data.columns if (column != "target" and column not in sentiment_features)
]


X_without_sentiment = data[base_features]


X_with_sentiment = data.drop(columns=["target"])


print("\nFeatures WITHOUT sentiment:", X_without_sentiment.shape[1])

print("Features WITH sentiment:", X_with_sentiment.shape[1])


tscv = TimeSeriesSplit(n_splits=5)


results = []


def evaluate_model(model_name, X, y):

    print()
    print("=" * 65)
    print(model_name)
    print("=" * 65)

    fold_number = 1

    for train_index, val_index in tscv.split(X):
        X_train = X.iloc[train_index]

        X_val = X.iloc[val_index]

        y_train = y.iloc[train_index]

        y_val = y.iloc[val_index]

        model = RandomForestClassifier(
            n_estimators=300, max_depth=8, min_samples_leaf=10, random_state=42, n_jobs=-1
        )

        model.fit(X_train, y_train)

        predictions = model.predict(X_val)

        probabilities = model.predict_proba(X_val)[:, 1]

        accuracy = accuracy_score(y_val, predictions)

        f1 = f1_score(y_val, predictions, zero_division=0)

        roc_auc = roc_auc_score(y_val, probabilities)

        always_up_accuracy = y_val.mean()

        train_start = X_train.index.min()

        train_end = X_train.index.max()

        val_start = X_val.index.min()

        val_end = X_val.index.max()

        print(f"\nFold {fold_number}")

        print(f"Train: {train_start.date()} to {train_end.date()}")

        print(f"Validation: {val_start.date()} to {val_end.date()}")

        print(f"Accuracy: {accuracy:.4f}")

        print(f"Always-UP Accuracy: {always_up_accuracy:.4f}")

        print(f"F1: {f1:.4f}")

        print(f"ROC-AUC: {roc_auc:.4f}")

        results.append(
            {
                "model": model_name,
                "fold": fold_number,
                "accuracy": accuracy,
                "f1": f1,
                "roc_auc": roc_auc,
                "always_up_accuracy": always_up_accuracy,
            }
        )

        fold_number += 1


evaluate_model("RF WITHOUT SENTIMENT", X_without_sentiment, y)


evaluate_model("RF WITH SENTIMENT", X_with_sentiment, y)


results_df = pd.DataFrame(results)


summary = results_df.groupby("model").agg(
    {
        "accuracy": ["mean", "std"],
        "f1": ["mean", "std"],
        "roc_auc": ["mean", "std"],
        "always_up_accuracy": "mean",
    }
)


print()
print("=" * 70)
print("SENTIMENT A/B TEST SUMMARY")
print("=" * 70)

print(summary.round(4))


auc_summary = results_df.groupby("model")["roc_auc"].mean()


without_auc = auc_summary["RF WITHOUT SENTIMENT"]

with_auc = auc_summary["RF WITH SENTIMENT"]


difference = with_auc - without_auc


print()
print("===== SENTIMENT IMPACT =====")

print(f"Without sentiment ROC-AUC: {without_auc:.4f}")

print(f"With sentiment ROC-AUC:    {with_auc:.4f}")

print(f"Difference:                {difference:+.4f}")


if difference > 0:
    print("\nSentiment improved average walk-forward ROC-AUC.")

elif difference < 0:
    print("\nSentiment reduced average walk-forward ROC-AUC.")

else:
    print("\nNo measurable ROC-AUC change.")


results_df.to_csv("data/processed/sentiment_rf_comparison.csv", index=False)


print("\nSaved results to:")

print("data/processed/sentiment_rf_comparison.csv")
