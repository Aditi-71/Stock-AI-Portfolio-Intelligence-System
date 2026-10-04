import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import GRU, Dense, Dropout, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2

np.random.seed(42)
tf.random.set_seed(42)


X_train = np.load("data/processed/X_train_seq.npy").astype("float32")

y_train = np.load("data/processed/y_train_seq.npy").astype("float32")

X_val = np.load("data/processed/X_val_seq.npy").astype("float32")

y_val = np.load("data/processed/y_val_seq.npy").astype("float32")


print("\nTraining shape:")
print(X_train.shape)

print("\nValidation shape:")
print(X_val.shape)


model = Sequential(
    [
        Input(shape=(X_train.shape[1], X_train.shape[2])),
        GRU(32, kernel_regularizer=l2(0.0001)),
        Dropout(0.40),
        Dense(16, activation="relu", kernel_regularizer=l2(0.0001)),
        Dropout(0.30),
        Dense(1, activation="sigmoid"),
    ]
)


model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
    loss="binary_crossentropy",
    metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
)


print("\n===== GRU MODEL =====")

model.summary()


early_stopping = EarlyStopping(
    monitor="val_auc", mode="max", patience=6, restore_best_weights=True, verbose=1
)


reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=0.00001, verbose=1)


print("\nTraining GRU...")


history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=40,
    batch_size=32,
    shuffle=False,
    callbacks=[early_stopping, reduce_lr],
    verbose=1,
)


probabilities = model.predict(X_val, verbose=0).ravel()


predictions = (probabilities >= 0.50).astype(int)


accuracy = accuracy_score(y_val, predictions)

precision = precision_score(y_val, predictions, zero_division=0)

recall = recall_score(y_val, predictions, zero_division=0)

f1 = f1_score(y_val, predictions, zero_division=0)

roc_auc = roc_auc_score(y_val, probabilities)

cm = confusion_matrix(y_val, predictions)


print("\n===== GRU VALIDATION RESULTS =====")

print(f"Accuracy:  {accuracy:.4f}")

print(f"Precision: {precision:.4f}")

print(f"Recall:    {recall:.4f}")

print(f"F1 Score:  {f1:.4f}")

print(f"ROC-AUC:   {roc_auc:.4f}")

print("\nConfusion Matrix:")

print(cm)


unique, counts = np.unique(predictions, return_counts=True)

print("\nPrediction counts:")

for label, count in zip(unique, counts):
    print(f"Class {label}: {count}")


print("\nProbability statistics:")

print(f"Minimum: {probabilities.min():.4f}")

print(f"Mean:    {probabilities.mean():.4f}")

print(f"Maximum: {probabilities.max():.4f}")


model.save("data/processed/gru_model.keras")


print("\nGRU model saved to:")

print("data/processed/gru_model.keras")
