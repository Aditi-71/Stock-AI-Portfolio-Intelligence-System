"""Adjusted-return targets and date-aware, purged, within-split windows."""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from .config import prepared_dir, sha256


def normalise_dates(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    dates = pd.DatetimeIndex(pd.to_datetime(out.index))
    if dates.tz is not None:
        dates = dates.tz_localize(None)
    out.index = dates.normalize()
    out.index.name = "Date"
    return out.sort_index(kind="stable")


def canonical_prices(frame: pd.DataFrame) -> pd.DataFrame:
    """Accept Yahoo's (field, ticker) or (ticker, field) column order."""
    out = normalise_dates(frame)
    if not isinstance(out.columns, pd.MultiIndex) or out.columns.nlevels != 2:
        raise ValueError("Expected Yahoo-style two-level OHLCV columns.")
    if "Adj Close" not in out.columns.get_level_values(0):
        if "Adj Close" in out.columns.get_level_values(1):
            out = out.swaplevel(axis=1)
        else:
            raise ValueError("Adjusted Close is missing. Re-download with auto_adjust=False.")
    return out.sort_index(axis=1)


def regression_targets(prices: pd.DataFrame, ticker: str, horizon: int) -> pd.DataFrame:
    """Target t+h uses positions on the original session grid, never shifted twice."""
    if horizon < 1:
        raise ValueError("horizon must be positive")
    p = prices[ticker].astype(float)
    if (p <= 0).any() or not np.isfinite(p).all():
        raise ValueError("Targets require finite, strictly positive adjusted prices.")
    dates = pd.Series(prices.index, index=prices.index)
    return pd.DataFrame(
        {
            "current_price": p,
            "actual_price": p.shift(-horizon),
            "target_return": np.log(p.shift(-horizon) / p),
            "target_date": dates.shift(-horizon),
        },
        index=prices.index,
    )


@dataclass
class Block:
    features: pd.DataFrame
    targets: pd.DataFrame
    origins: np.ndarray
    lo: int
    hi: int
    length: int

    @property
    def X(self) -> np.ndarray:
        return self.features.iloc[self.origins].to_numpy(dtype=np.float64)

    @property
    def y(self) -> np.ndarray:
        return self.targets.iloc[self.origins]["target_return"].to_numpy(dtype=np.float64)

    @property
    def rows(self) -> pd.DataFrame:
        return self.targets.iloc[self.origins].copy()

    @property
    def scaler_rows(self) -> np.ndarray:
        offsets = np.arange(-self.length + 1, 1)
        positions = np.unique(self.origins[:, None] + offsets)
        return self.features.iloc[positions].to_numpy(dtype=np.float64)

    def sequences(self, scaler=None) -> np.ndarray:
        raw = self.features.to_numpy(dtype=np.float64)
        if scaler is not None:
            raw = scaler.transform(raw)
        offsets = np.arange(-self.length + 1, 1)
        return raw[self.origins[:, None] + offsets].astype(np.float32)


def make_block(features, targets, lo, hi, horizon, length) -> Block:
    if not features.index.equals(targets.index):
        raise ValueError("Feature and target date indices do not match.")
    if not 0 <= lo < hi <= len(features):
        raise ValueError("Invalid chronological block boundaries.")
    origins = np.arange(lo + length - 1, hi - horizon, dtype=int)
    if not len(origins):
        raise ValueError(f"Block [{lo}, {hi}) is too short for L={length}, h={horizon}.")
    rows = targets.iloc[origins]
    if (
        rows.isna().any().any()
        or (rows.target_date >= features.index[hi - 1] + pd.Timedelta(days=1)).any()
    ):
        raise ValueError("A target crosses its permitted split boundary.")
    return Block(features, targets, origins, lo, hi, length)


def split_bounds(n: int, config: dict) -> dict:
    train_end = int(n * config["train_ratio"])
    test_start = int(n * (config["train_ratio"] + config["validation_ratio"]))
    calibration_start = train_end + (test_start - train_end) // 2
    return {
        "train": (0, train_end),
        "validation": (train_end, calibration_start),
        "calibration": (calibration_start, test_start),
        "test": (test_start, n),
    }


def blocks_for(features, prices, ticker, horizon, config) -> dict[str, Block]:
    targets = regression_targets(prices, ticker, horizon)
    return {
        name: make_block(features, targets, lo, hi, horizon, config["sequence_length"])
        for name, (lo, hi) in split_bounds(len(features), config).items()
    }


def walk_forward_blocks(features, targets, horizon, config):
    train_end = split_bounds(len(features), config)["train"][1]
    length = config["sequence_length"]
    for fold, (train, val) in enumerate(
        TimeSeriesSplit(n_splits=config["regression"]["cv_splits"]).split(np.arange(train_end)), 1
    ):
        yield (
            fold,
            make_block(features, targets, 0, int(train[-1]) + 1, horizon, length),
            make_block(features, targets, int(val[0]), int(val[-1]) + 1, horizon, length),
        )


def load_prepared(config):
    path = prepared_dir(config)
    if not (path / "manifest.json").exists():
        raise FileNotFoundError("Prepare the PRD data first: python -m src.prd prepare")
    features = pd.read_parquet(path / "features.parquet")
    prices = pd.read_parquet(path / "adjusted_prices.parquet")
    manifest = json.loads((path / "manifest.json").read_text())
    if not features.index.equals(prices.index):
        raise ValueError("Prepared price and feature dates differ. Re-run prepare.")
    if not np.isfinite(features.to_numpy()).all() or not features.index.is_unique:
        raise ValueError("Prepared features must be finite with unique dates.")
    if manifest["universe"] != config["universe"]:
        raise ValueError("Universe changed after preparation. Re-run prepare.")
    if (
        manifest["sequence_length"] != config["sequence_length"]
        or manifest["horizons"] != config["regression"]["horizons"]
    ):
        raise ValueError("Sequence length or horizons changed after preparation. Re-run prepare.")
    if manifest["macro_lag_days"] != config["regression"]["macro_lag_days"]:
        raise ValueError("Macro lag configuration changed. Re-run prepare.")
    if {k: (v["lo"], v["hi"]) for k, v in manifest["split_bounds"].items()} != split_bounds(
        len(features), config
    ):
        raise ValueError("Split settings changed after preparation. Re-run prepare.")
    for name, digest in manifest["prepared_hashes"].items():
        if sha256(path / name) != digest:
            raise ValueError(f"Prepared file changed without rebuilding its manifest: {name}")
    return features, prices, manifest
