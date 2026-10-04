"""LSTM, GRU, BiLSTM and positional Transformer regression with Keras."""

from __future__ import annotations

import gc
import os
import random
from functools import lru_cache

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler


@lru_cache(maxsize=1)
def backend():
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
    import tensorflow as tf

    @tf.keras.utils.register_keras_serializable(package="StockPRD")
    class SinusoidalPosition(tf.keras.layers.Layer):
        def __init__(self, length, width, **kwargs):
            super().__init__(**kwargs)
            self.length, self.width = int(length), int(width)
            positions = np.arange(length)[:, None]
            dimensions = np.arange(width)[None, :]
            angles = positions / np.power(10000.0, 2 * (dimensions // 2) / width)
            enc = np.where(dimensions % 2 == 0, np.sin(angles), np.cos(angles))
            self.encoding = tf.constant(enc[None, :, :], dtype=tf.float32)

        def call(self, inputs):
            return inputs + tf.cast(self.encoding, inputs.dtype)

        def get_config(self):
            return {**super().get_config(), "length": self.length, "width": self.width}

    return tf, SinusoidalPosition


def build_deep(name, length, features, config):
    tf, Position = backend()
    tf.keras.backend.clear_session()
    gc.collect()
    seed = config["random_seed"]
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()
    layers = tf.keras.layers
    r = config["regression"]
    inputs = layers.Input((length, features))
    if name in ("lstm", "gru"):
        recurrent = layers.LSTM if name == "lstm" else layers.GRU
        x = recurrent(64, return_sequences=True)(inputs)
        x = layers.Dropout(r["dropout"])(x)
        x = recurrent(32)(x)
        x = layers.Dropout(r["dropout"])(x)
    elif name == "bilstm":
        x = layers.Bidirectional(layers.LSTM(32))(inputs)
        x = layers.Dropout(r["dropout"])(x)
    elif name == "transformer":
        x = layers.Dense(32)(inputs)
        x = Position(length, 32)(x)
        for _ in range(2):
            attention = layers.MultiHeadAttention(num_heads=4, key_dim=8, dropout=0.1)(x, x)
            x = layers.LayerNormalization()(x + attention)
            feedforward = layers.Dense(64, activation="relu")(x)
            feedforward = layers.Dropout(0.1)(feedforward)
            feedforward = layers.Dense(32)(feedforward)
            x = layers.LayerNormalization()(x + feedforward)
        x = layers.GlobalAveragePooling1D()(x)
    else:
        raise ValueError(f"Unknown deep model: {name}")
    x = layers.Dense(16, activation="relu")(x)
    output = layers.Dense(1, activation="linear")(x)
    model = tf.keras.Model(inputs, output, name=name)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=r["learning_rate"]),
        loss=tf.keras.losses.Huber(),
        metrics=["mae"],
    )
    return model


class DeepRegressor:
    def __init__(self, model, x_scaler, y_scaler, history):
        self.model, self.x_scaler, self.y_scaler, self.history = model, x_scaler, y_scaler, history

    def predict_block(self, block):
        pred = self.model.predict(block.sequences(self.x_scaler), verbose=0).reshape(-1, 1)
        return self.y_scaler.inverse_transform(pred).ravel()

    def save(self, directory):
        directory.mkdir(parents=True, exist_ok=True)
        self.model.save(directory / "model.keras")
        joblib.dump(
            {"x_scaler": self.x_scaler, "y_scaler": self.y_scaler}, directory / "scalers.joblib"
        )


def fit_deep(name, train, stopping, config):
    tf, _ = backend()
    xs = StandardScaler().fit(train.scaler_rows)
    ys = StandardScaler().fit(train.y.reshape(-1, 1))
    model = build_deep(name, train.length, train.features.shape[1], config)
    callback = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=config["regression"]["patience"], restore_best_weights=True
    )
    history = model.fit(
        train.sequences(xs),
        ys.transform(train.y[:, None]).astype(np.float32),
        validation_data=(
            stopping.sequences(xs),
            ys.transform(stopping.y[:, None]).astype(np.float32),
        ),
        epochs=config["regression"]["epochs"],
        batch_size=config["regression"]["batch_size"],
        callbacks=[callback],
        shuffle=False,
        verbose=0,
    )
    return DeepRegressor(model, xs, ys, history.history)
