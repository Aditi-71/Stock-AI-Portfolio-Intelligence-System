from pathlib import Path

import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

data_dir = Path("data/processed")

X_train = pd.read_parquet(data_dir / "X_train_clean.parquet")
y_train = pd.read_parquet(data_dir / "y_train_clean.parquet")["target"]

X_val = pd.read_parquet(data_dir / "X_val_clean.parquet")
y_val = pd.read_parquet(data_dir / "y_val_clean.parquet")["target"]

X_test = pd.read_parquet(data_dir / "X_test_clean.parquet")
y_test = pd.read_parquet(data_dir / "y_test_clean.parquet")["target"]


scaler = StandardScaler()


X_train_scaled = scaler.fit_transform(X_train)


X_val_scaled = scaler.transform(X_val)

X_test_scaled = scaler.transform(X_test)


X_train_scaled = pd.DataFrame(X_train_scaled, index=X_train.index, columns=X_train.columns)

X_val_scaled = pd.DataFrame(X_val_scaled, index=X_val.index, columns=X_val.columns)

X_test_scaled = pd.DataFrame(X_test_scaled, index=X_test.index, columns=X_test.columns)


X_train_scaled.to_parquet(data_dir / "X_train_scaled.parquet")
X_val_scaled.to_parquet(data_dir / "X_val_scaled.parquet")
X_test_scaled.to_parquet(data_dir / "X_test_scaled.parquet")

y_train.to_frame().to_parquet(data_dir / "y_train_scaled.parquet")
y_val.to_frame().to_parquet(data_dir / "y_val_scaled.parquet")
y_test.to_frame().to_parquet(data_dir / "y_test_scaled.parquet")


joblib.dump(scaler, data_dir / "scaler.joblib")


print("Feature scaling completed!")
print()

print("Training:", X_train_scaled.shape)
print("Validation:", X_val_scaled.shape)
print("Test:", X_test_scaled.shape)

print()
print("Scaler fitted ONLY on training data.")

print()
print("Saved scaled datasets and scaler to:")
print(data_dir)
