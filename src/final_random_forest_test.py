from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

data_dir = Path("data/processed")

X_train = pd.read_parquet(data_dir / "X_train_clean.parquet")

y_train = pd.read_parquet(data_dir / "y_train_clean.parquet")["target"].astype(int)

X_val = pd.read_parquet(data_dir / "X_val_clean.parquet")

y_val = pd.read_parquet(data_dir / "y_val_clean.parquet")["target"].astype(int)

X_test = pd.read_parquet(data_dir / "X_test_clean.parquet")

y_test = pd.read_parquet(data_dir / "y_test_clean.parquet")["target"].astype(int)


X_final_train = pd.concat([X_train, X_val])

y_final_train = pd.concat([y_train, y_val])


print("\nFinal training rows:")
print(len(X_final_train))

print("\nFinal training date range:")
print(X_final_train.index.min(), "to", X_final_train.index.max())


model = RandomForestClassifier(
    n_estimators=300, max_depth=8, min_samples_leaf=10, random_state=42, n_jobs=-1
)


print("\nTraining final Random Forest...")

model.fit(X_final_train, y_final_train)


test_probabilities = model.predict_proba(X_test)[:, 1]

test_predictions = (test_probabilities >= 0.50).astype(int)


result = pd.DataFrame(
    {"actual": y_test.values, "prediction": test_predictions, "probability_up": test_probabilities},
    index=X_test.index,
)

result.index.name = "Date"

output_file = data_dir / "final_rf_test_predictions.parquet"

result.to_parquet(output_file)


print("\nSaved:")
print(output_file)

print("\nPrediction counts:")
print(result["prediction"].value_counts())

print("\nProbability statistics:")
print(result["probability_up"].describe())
