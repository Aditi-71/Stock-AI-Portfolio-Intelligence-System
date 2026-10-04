from pathlib import Path

import pandas as pd

input_file = Path("data/processed/model_data.parquet")

data = pd.read_parquet(input_file)


data = data.sort_index()


n = len(data)

train_end = int(n * 0.70)
val_end = int(n * 0.85)


train = data.iloc[:train_end].copy()

validation = data.iloc[train_end:val_end].copy()

test = data.iloc[val_end:].copy()


X_train = train.drop(columns=["target"])
y_train = train["target"]

X_val = validation.drop(columns=["target"])
y_val = validation["target"]

X_test = test.drop(columns=["target"])
y_test = test["target"]


print("Chronological split completed!")
print()

print("Total rows:", len(data))

print()
print("Training set:")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)
print("Date range:", X_train.index.min(), "to", X_train.index.max())

print()
print("Validation set:")
print("X_val:", X_val.shape)
print("y_val:", y_val.shape)
print("Date range:", X_val.index.min(), "to", X_val.index.max())

print()
print("Test set:")
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)
print("Date range:", X_test.index.min(), "to", X_test.index.max())


output_dir = Path("data/processed")
output_dir.mkdir(parents=True, exist_ok=True)

X_train.to_parquet(output_dir / "X_train.parquet")
y_train.to_frame().to_parquet(output_dir / "y_train.parquet")

X_val.to_parquet(output_dir / "X_val.parquet")
y_val.to_frame().to_parquet(output_dir / "y_val.parquet")

X_test.to_parquet(output_dir / "X_test.parquet")
y_test.to_frame().to_parquet(output_dir / "y_test.parquet")


print()
print("Saved train/validation/test datasets to:")
print(output_dir)
