from pathlib import Path

import numpy as np
import pandas as pd

sequence_length = 60


X_val = pd.read_parquet("data/processed/X_val_scaled.parquet")

X_test = pd.read_parquet("data/processed/X_test_scaled.parquet")

y_test = pd.read_parquet("data/processed/y_test_scaled.parquet")["target"].astype(int)


history = X_val.tail(sequence_length - 1)


combined_X = pd.concat([history, X_test])


X_test_seq = []
y_test_seq = []
test_dates = []


for i in range(len(X_test)):
    start = i
    end = i + sequence_length

    sequence = combined_X.iloc[start:end].values

    X_test_seq.append(sequence)

    y_test_seq.append(y_test.iloc[i])

    test_dates.append(X_test.index[i])


X_test_seq = np.array(X_test_seq)

y_test_seq = np.array(y_test_seq)

test_dates = np.array(test_dates)


print("\nTest sequence creation completed!")

print("\nX_test_seq shape:")

print(X_test_seq.shape)

print("\ny_test_seq shape:")

print(y_test_seq.shape)

print("\nTest date range:")

print(test_dates[0], "to", test_dates[-1])

print("\nSequence length:")

print(X_test_seq.shape[1])

print("\nNumber of features:")

print(X_test_seq.shape[2])


output_dir = Path("data/processed")

np.save(output_dir / "X_test_seq.npy", X_test_seq)

np.save(output_dir / "y_test_seq.npy", y_test_seq)

np.save(output_dir / "test_seq_dates.npy", test_dates)


print("\nSaved test sequences to:")

print(output_dir)
