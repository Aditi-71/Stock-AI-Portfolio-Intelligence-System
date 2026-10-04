from pathlib import Path

import pandas as pd

data_dir = Path("data/processed")

X_train = pd.read_parquet(data_dir / "X_train.parquet")
y_train = pd.read_parquet(data_dir / "y_train.parquet")["target"]

X_val = pd.read_parquet(data_dir / "X_val.parquet")
y_val = pd.read_parquet(data_dir / "y_val.parquet")["target"]

X_test = pd.read_parquet(data_dir / "X_test.parquet")
y_test = pd.read_parquet(data_dir / "y_test.parquet")["target"]


train_mask = X_train.notna().all(axis=1)
val_mask = X_val.notna().all(axis=1)
test_mask = X_test.notna().all(axis=1)

X_train = X_train.loc[train_mask]
y_train = y_train.loc[train_mask]

X_val = X_val.loc[val_mask]
y_val = y_val.loc[val_mask]

X_test = X_test.loc[test_mask]
y_test = y_test.loc[test_mask]


print("Missing-value handling completed!")
print()

print("Training:")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print()
print("Validation:")
print("X_val:", X_val.shape)
print("y_val:", y_val.shape)

print()
print("Test:")
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)

print()
print("Remaining missing values:")
print("Train:", X_train.isna().sum().sum())
print("Validation:", X_val.isna().sum().sum())
print("Test:", X_test.isna().sum().sum())


X_train.to_parquet(data_dir / "X_train_clean.parquet")
y_train.to_frame().to_parquet(data_dir / "y_train_clean.parquet")

X_val.to_parquet(data_dir / "X_val_clean.parquet")
y_val.to_frame().to_parquet(data_dir / "y_val_clean.parquet")

X_test.to_parquet(data_dir / "X_test_clean.parquet")
y_test.to_frame().to_parquet(data_dir / "y_test_clean.parquet")

print()
print("Saved cleaned datasets to:")
print(data_dir)
