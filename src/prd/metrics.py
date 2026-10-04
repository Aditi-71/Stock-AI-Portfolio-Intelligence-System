"""Regression metrics on reconstructed prices, with explicit direction ties."""

from __future__ import annotations

import numpy as np


def reconstruct_prices(current, predicted_log_return):
    current, r = np.asarray(current, dtype=float), np.asarray(predicted_log_return, dtype=float)
    if current.shape != r.shape or not np.isfinite(r).all() or (current <= 0).any():
        raise ValueError("Invalid price reconstruction inputs.")
    with np.errstate(over="raise", invalid="raise", under="ignore"):
        result = current * np.exp(r)
    if not np.isfinite(result).all() or (result <= 0).any():
        raise ValueError(
            "Invalid predicted price; inspect the model instead of clipping its outputs."
        )
    return result


def regression_metrics(actual, predicted, current) -> dict:
    actual, predicted, current = [np.asarray(x, dtype=float) for x in (actual, predicted, current)]
    if (
        not (actual.shape == predicted.shape == current.shape)
        or actual.ndim != 1
        or not len(actual)
    ):
        raise ValueError("Metric inputs must be equal nonempty one-dimensional arrays.")
    if not all(np.isfinite(x).all() for x in (actual, predicted, current)) or (actual <= 0).any():
        raise ValueError("Metrics require finite predictions and positive actual prices.")
    err = actual - predicted
    sst = np.sum((actual - actual.mean()) ** 2)
    actual_sign = np.sign(actual - current)
    pred_sign = np.sign(predicted - current)
    return {
        "n": len(actual),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mae": float(np.mean(np.abs(err))),
        "mape_pct": float(100 * np.mean(np.abs(err / actual))),
        "r2": float(1 - np.sum(err**2) / sst) if sst > 0 else np.nan,
        "directional_accuracy_pct": float(100 * np.mean(actual_sign == pred_sign)),
        "unchanged_forecast_pct": float(100 * np.mean(pred_sign == 0)),
    }


def predictions_frame(block, predicted_return, ticker, horizon, model, split, radius=None):
    frame = block.rows
    r = np.asarray(predicted_return, dtype=float).reshape(-1)
    if len(r) != len(frame):
        raise ValueError("Prediction count differs from the scoring origins.")
    frame["predicted_return"] = r
    frame["predicted_price"] = reconstruct_prices(frame.current_price.to_numpy(), r)
    frame["ticker"], frame["horizon"], frame["model"], frame["split"] = (
        ticker,
        horizon,
        model,
        split,
    )
    if radius is not None:
        frame["lower_price"] = reconstruct_prices(frame.current_price.to_numpy(), r - radius)
        frame["upper_price"] = reconstruct_prices(frame.current_price.to_numpy(), r + radius)
    return frame.rename_axis("origin_date").reset_index()


def score_frame(frame):
    metrics = regression_metrics(frame.actual_price, frame.predicted_price, frame.current_price)
    if "lower_price" in frame:
        metrics["interval_coverage_pct"] = float(
            100
            * (
                (frame.actual_price >= frame.lower_price)
                & (frame.actual_price <= frame.upper_price)
            ).mean()
        )
    return metrics


def calibration_radius(actual_return, predicted_return, alpha=0.1):
    residual = np.abs(np.asarray(actual_return) - np.asarray(predicted_return))
    if not len(residual) or not np.isfinite(residual).all():
        raise ValueError("Calibration residuals are empty or invalid.")
    level = min(1, np.ceil((len(residual) + 1) * (1 - alpha)) / len(residual))
    return float(np.quantile(residual, level, method="higher"))
