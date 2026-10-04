import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2

SEQUENCE_LENGTH = 60

np.random.seed(42)
tf.random.set_seed(42)


X = pd.read_parquet("data/processed/X_train_clean.parquet")

y = pd.read_parquet("data/processed/y_train_clean.parquet")["target"].astype(int)


print("\nDataset:")
print("Rows:", len(X))
print("Features:", X.shape[1])


def create_train_sequences(X, y, sequence_length):

    X_sequences = []
    y_sequences = []

    for i in range(sequence_length - 1, len(X)):
        start = i - sequence_length + 1

        sequence = X[start : i + 1]

        target = y[i]

        X_sequences.append(sequence)
        y_sequences.append(target)

    return (np.array(X_sequences), np.array(y_sequences))


def create_validation_sequences(X_train, X_val, y_val, sequence_length):

    history = X_train[-(sequence_length - 1) :]

    combined = np.concatenate([history, X_val], axis=0)

    X_sequences = []
    y_sequences = []

    for i in range(len(X_val)):
        sequence = combined[i : i + sequence_length]

        X_sequences.append(sequence)

        y_sequences.append(y_val[i])

    return (np.array(X_sequences), np.array(y_sequences))


def build_lstm_model(sequence_length, number_of_features):

    model = Sequential(
        [
            Input(shape=(sequence_length, number_of_features)),
            LSTM(32, kernel_regularizer=l2(0.0001)),
            Dropout(0.40),
            Dense(16, activation="relu", kernel_regularizer=l2(0.0001)),
            Dropout(0.30),
            Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.AUC(name="auc")],
    )

    return model


tscv = TimeSeriesSplit(n_splits=5)

results = []

fold = 1


for train_index, val_index in tscv.split(X):
    print()
    print("=" * 60)
    print(f"LSTM FOLD {fold}")
    print("=" * 60)

    X_train_raw = X.iloc[train_index]

    X_val_raw = X.iloc[val_index]

    y_train = y.iloc[train_index]

    y_val = y.iloc[val_index]

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train_raw)

    X_val_scaled = scaler.transform(X_val_raw)

    X_train_seq, y_train_seq = create_train_sequences(
        X_train_scaled, y_train.to_numpy(), SEQUENCE_LENGTH
    )

    X_val_seq, y_val_seq = create_validation_sequences(
        X_train_scaled, X_val_scaled, y_val.to_numpy(), SEQUENCE_LENGTH
    )

    X_train_seq = X_train_seq.astype("float32")

    y_train_seq = y_train_seq.astype("float32")

    X_val_seq = X_val_seq.astype("float32")

    y_val_seq = y_val_seq.astype("float32")

    print("Training sequences:", X_train_seq.shape)

    print("Validation sequences:", X_val_seq.shape)

    tf.keras.backend.clear_session()

    np.random.seed(42)
    tf.random.set_seed(42)

    model = build_lstm_model(SEQUENCE_LENGTH, X.shape[1])

    early_stopping = EarlyStopping(
        monitor="val_auc", mode="max", patience=5, restore_best_weights=True, verbose=0
    )

    history = model.fit(
        X_train_seq,
        y_train_seq,
        validation_data=(X_val_seq, y_val_seq),
        epochs=30,
        batch_size=32,
        shuffle=False,
        callbacks=[early_stopping],
        verbose=0,
    )

    probabilities = model.predict(X_val_seq, verbose=0).ravel()

    predictions = (probabilities >= 0.50).astype(int)

    accuracy = accuracy_score(y_val_seq, predictions)

    f1 = f1_score(y_val_seq, predictions, zero_division=0)

    auc = roc_auc_score(y_val_seq, probabilities)

    always_up_accuracy = y_val_seq.mean()

    best_epoch = np.argmax(history.history["val_auc"]) + 1

    train_start = X_train_raw.index.min()

    train_end = X_train_raw.index.max()

    val_start = X_val_raw.index.min()

    val_end = X_val_raw.index.max()

    print(f"Train: {train_start.date()} to {train_end.date()}")

    print(f"Validation: {val_start.date()} to {val_end.date()}")

    print(f"Best epoch: {best_epoch}")

    print(f"Accuracy: {accuracy:.4f}")

    print(f"Always-UP accuracy: {always_up_accuracy:.4f}")

    print(f"F1: {f1:.4f}")

    print(f"ROC-AUC: {auc:.4f}")

    results.append(
        {
            "fold": fold,
            "accuracy": accuracy,
            "always_up_accuracy": always_up_accuracy,
            "f1": f1,
            "roc_auc": auc,
            "best_epoch": best_epoch,
        }
    )

    fold += 1


results_df = pd.DataFrame(results)


print()
print("=" * 60)
print("LSTM WALK-FORWARD SUMMARY")
print("=" * 60)


print("\nMean Accuracy:", round(results_df["accuracy"].mean(), 4))

print("Accuracy Std:", round(results_df["accuracy"].std(), 4))


print("\nMean F1:", round(results_df["f1"].mean(), 4))

print("F1 Std:", round(results_df["f1"].std(), 4))


print("\nMean ROC-AUC:", round(results_df["roc_auc"].mean(), 4))

print("ROC-AUC Std:", round(results_df["roc_auc"].std(), 4))


print("\nMean Always-UP Accuracy:", round(results_df["always_up_accuracy"].mean(), 4))


results_df.to_csv("data/processed/walk_forward_lstm_results.csv", index=False)


print("\nSaved to:")

print("data/processed/walk_forward_lstm_results.csv")
