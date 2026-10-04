"""Configuration, paths and provenance shared by the PRD commands."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
REQUIRED_MODELS = (
    "linear",
    "random_forest",
    "xgboost",
    "svr",
    "lstm",
    "gru",
    "bilstm",
    "transformer",
)
DEEP_MODELS = REQUIRED_MODELS[4:]


def load_config(path: str | Path = "config.yaml") -> dict:
    path = Path(path).resolve()
    config = yaml.safe_load(path.read_text())
    config["project_root"] = str(path.parent)
    r = config.setdefault("regression", {})
    defaults = {
        "horizons": [1, 5],
        "cv_splits": 5,
        "min_history_years": 8,
        "price_fill_limit": 1,
        "end_date": "2026-08-26",
        "calendar": "NYSE",
        "macro_lag_days": {"DGS10": 2, "DGS3MO": 2, "CPIAUCSL": 60, "UNRATE": 45},
        "epochs": 200,
        "patience": 12,
        "batch_size": 64,
        "learning_rate": 0.001,
        "dropout": 0.2,
        "n_jobs": 2,
        "interval_alpha": 0.10,
        "historical_test_exposed_through": "2026-08-26",
        "annual_risk_free_rate": 0.04,
        "weight_cap": 0.30,
        "forecast_blend": 0.25,
        "mc_portfolios": 20000,
        "bootstrap_samples": 1000,
        "bootstrap_block": 20,
        "transaction_cost_bps": 10,
        "recommendation_weights": [0.5, 0.2, 0.3],
        "recommendation_thresholds": [-0.10, 0.10],
        "rebalance_band": 0.05,
    }
    for key, value in defaults.items():
        r.setdefault(key, value)
    ratios = [float(config[k]) for k in ("train_ratio", "validation_ratio", "test_ratio")]
    if min(ratios) <= 0 or not np.isclose(sum(ratios), 1):
        raise ValueError("The chronological split ratios must be positive and sum to 1.")
    if len(set(config["universe"])) != len(config["universe"]):
        raise ValueError("Duplicate universe tickers.")
    if not r["horizons"] or any(type(h) is not int or h < 1 for h in r["horizons"]):
        raise ValueError("Horizons must be positive integer trading-session counts.")
    if int(config["sequence_length"]) < 2 or int(r["cv_splits"]) < 2:
        raise ValueError("Sequence length and CV split count must be at least 2.")
    if not 0 < r["interval_alpha"] < 1:
        raise ValueError("interval_alpha must be between 0 and 1.")
    return config


def project_path(config: dict, *parts: str) -> Path:
    return Path(config["project_root"]).joinpath(*parts)


def prepared_dir(config: dict) -> Path:
    return project_path(config, "data", "processed", "prd")


def write_json(path: Path, value) -> None:
    def clean(v):
        if isinstance(v, dict):
            return {str(k): clean(x) for k, x in v.items()}
        if isinstance(v, (list, tuple, np.ndarray)):
            return [clean(x) for x in v]
        if isinstance(v, (pd.Timestamp, Path)):
            return str(v)
        if isinstance(v, np.generic):
            return clean(v.item())
        if isinstance(v, float) and not np.isfinite(v):
            return None
        return v

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(value), indent=2, allow_nan=False) + "\n")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()
