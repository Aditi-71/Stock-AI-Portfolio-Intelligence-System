from pathlib import Path

import numpy as np
import pandas as pd

sequence_length = 60


X_train = pd.read_parquet("data/processed/X_train_scaled.parquet")

y_train = pd.read_parquet("data/processed/y_train_scaled.parquet")["target"].astype(int)

X_val = pd.read_parquet("data/processed/X_val_scaled.parquet")

y_val = pd.read_parquet("data/processed/y_val_scaled.parquet")["target"].astype(int)


def create_train_sequences(X, y, sequence_length):

    X_sequences = []
    y_sequences = []
    dates = []

    for i in range(sequence_length - 1, len(X)):
        start = i - sequence_length + 1

        end = i + 1

        sequence = X.iloc[start:end].values

        target = y.iloc[i]

        X_sequences.append(sequence)

        y_sequences.append(target)

        dates.append(X.index[i])

    return (np.array(X_sequences), np.array(y_sequences), np.array(dates))


def create_validation_sequences(X_history, X_current, y_current, sequence_length):

    history = X_history.tail(sequence_length - 1)

    combined_X = pd.concat([history, X_current])

    X_sequences = []
    y_sequences = []
    dates = []

    for i in range(len(X_current)):
        start = i

        end = i + sequence_length

        sequence = combined_X.iloc[start:end].values

        target = y_current.iloc[i]

        X_sequences.append(sequence)

        y_sequences.append(target)

        dates.append(X_current.index[i])

    return (np.array(X_sequences), np.array(y_sequences), np.array(dates))


X_train_seq, y_train_seq, train_dates = create_train_sequences(X_train, y_train, sequence_length)


X_val_seq, y_val_seq, val_dates = create_validation_sequences(
    X_train, X_val, y_val, sequence_length
)


print("\nSequence creation completed!")

print("\nTraining sequences:")

print("X shape:", X_train_seq.shape)

print("y shape:", y_train_seq.shape)

print("\nValidation sequences:")

print("X shape:", X_val_seq.shape)

print("y shape:", y_val_seq.shape)

print("\nEach sample contains:")

print(sequence_length, "days ×", X_train.shape[1], "features")

print("\nTraining sequence date range:")

print(train_dates[0], "to", train_dates[-1])

print("\nValidation sequence date range:")

print(val_dates[0], "to", val_dates[-1])


print("\nSanity checks:")

print("Training sequence length:", X_train_seq.shape[1])

print("Validation sequence length:", X_val_seq.shape[1])

print("Number of features:", X_train_seq.shape[2])

print("Validation targets:", len(y_val_seq))

print("Original validation targets:", len(y_val))


output_dir = Path("data/processed")

np.save(output_dir / "X_train_seq.npy", X_train_seq)

np.save(output_dir / "y_train_seq.npy", y_train_seq)

np.save(output_dir / "X_val_seq.npy", X_val_seq)

np.save(output_dir / "y_val_seq.npy", y_val_seq)

np.save(output_dir / "train_seq_dates.npy", train_dates)

np.save(output_dir / "val_seq_dates.npy", val_dates)


print("\nSequences saved to:")

print(output_dir)
