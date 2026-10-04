"""Rebuild audited PRD inputs from the project's existing raw Parquet files."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import prepared_dir, project_path, sha256, write_json
from .data import canonical_prices, normalise_dates, split_bounds


def exchange_sessions(start, end, calendar="NYSE"):
    import pandas_market_calendars as mcal

    return pd.DatetimeIndex(
        mcal.get_calendar(calendar).schedule(start_date=start, end_date=end).index
    ).tz_localize(None)


def clean_prices(raw, sessions, tickers, fill_limit=1):
    prices = canonical_prices(raw)
    duplicate_rows = int(prices.index.duplicated(keep=False).sum())
    duplicate_conflicts = []
    for date, group in prices[prices.index.duplicated(keep=False)].groupby(level=0):
        if len(group.drop_duplicates()) > 1:
            duplicate_conflicts.append(str(date.date()))
    prices = prices[~prices.index.duplicated(keep="last")]
    fields = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    required = [(f, t) for t in tickers for f in fields]
    missing = [x for x in required if x not in prices.columns]
    if missing:
        raise ValueError(f"Missing OHLCV columns: {missing}")
    before = prices.reindex(sessions).loc[:, required].astype(float)
    out = before.copy()
    changes, outliers, violations = [], [], []
    for ticker in tickers:
        cols = [(f, ticker) for f in fields if f != "Volume"]
        missing_rows = out[cols].isna().any(axis=1)
        groups = missing_rows.ne(missing_rows.shift()).cumsum()
        lengths = missing_rows.groupby(groups).transform("sum")
        if (missing_rows & (lengths > fill_limit)).any():
            dates = out.index[missing_rows & (lengths > fill_limit)]
            raise ValueError(
                f"{ticker}: missing-price gap exceeds {fill_limit} session(s): {dates[:8].tolist()}. Inspect raw data."
            )
        out[cols] = out[cols].ffill(limit=fill_limit)
        out[("Volume", ticker)] = out[("Volume", ticker)].fillna(0)
        for date in out.index[before.loc[:, [(f, ticker) for f in fields]].isna().any(axis=1)]:
            changes.append(
                {
                    "date": str(date.date()),
                    "ticker": ticker,
                    "rule": "price_ffill_limited_and_missing_volume_zero",
                }
            )
        p = {f: out[(f, ticker)] for f in fields}
        bad = (
            (p["Low"] > p["High"])
            | (p["Open"] < p["Low"])
            | (p["Open"] > p["High"])
            | (p["Close"] < p["Low"])
            | (p["Close"] > p["High"])
        )
        bad |= (out[cols] <= 0).any(axis=1) | (p["Volume"] < 0)
        violations.extend({"date": str(d.date()), "ticker": ticker} for d in out.index[bad])
        r = np.log(p["Adj Close"]).diff()
        mean, sd = r.shift(1).rolling(63).mean(), r.shift(1).rolling(63).std()
        q1, q3 = r.shift(1).rolling(63).quantile(0.25), r.shift(1).rolling(63).quantile(0.75)
        zflag = ((r - mean) / sd.replace(0, np.nan)).abs() > 5
        iqrflag = (r < q1 - 1.5 * (q3 - q1)) | (r > q3 + 1.5 * (q3 - q1))
        for date in out.index[zflag | iqrflag]:
            outliers.append(
                {
                    "date": str(date.date()),
                    "ticker": ticker,
                    "log_return": float(r.loc[date]),
                    "z_flag": bool(zflag.loc[date]),
                    "iqr_flag": bool(iqrflag.loc[date]),
                    "action": "kept_for_review",
                }
            )
    report = {
        "raw_rows": len(raw),
        "session_rows": len(out),
        "duplicate_rows": duplicate_rows,
        "duplicate_conflict_dates": duplicate_conflicts,
        "filled_rows": len(changes),
        "remaining_missing_cells": int(out.isna().sum().sum()),
        "outliers_flagged": len(outliers),
        "invariant_violations": len(violations),
        "fill_events": changes,
        "outliers": outliers,
        "violations": violations,
    }
    if violations or out.isna().any().any():
        raise ValueError(
            f"Unresolved price quality errors: {len(violations)} OHLC/price violations, {report['remaining_missing_cells']} missing cells. Inspect and correct raw data."
        )
    return out.sort_index(axis=1), report


def align_macro(macro, dates, lags):
    macro = normalise_dates(macro)
    macro = macro[~macro.index.duplicated(keep="last")]
    result = {}
    for name, lag in lags.items():
        if name not in macro:
            raise ValueError(f"Missing macro series: {name}")
        observations = macro[name].dropna().copy()
        observations.index += pd.Timedelta(days=int(lag))
        result[name] = (
            observations.reindex(observations.index.union(dates))
            .sort_index()
            .ffill()
            .reindex(dates)
        )
    return pd.DataFrame(result, index=dates)


def engineer_features(prices, macro, universe, benchmark):
    columns = {}
    returns, above_ma = [], []
    for ticker in universe:
        p, close = prices[("Adj Close", ticker)], prices[("Close", ticker)]
        high = prices[("High", ticker)] * p / close
        low = prices[("Low", ticker)] * p / close
        volume = prices[("Volume", ticker)]
        logp, r = np.log(p), np.log(p).diff()
        local = {}
        for h in (1, 5, 10, 21):
            local[f"ret_{h}d"] = logp.diff(h)
        for w in (10, 20, 50, 200):
            local[f"p_sma{w}"] = p / p.rolling(w).mean()
            local[f"p_ema{w}"] = p / p.ewm(span=w, adjust=False).mean()
        macd = p.ewm(span=12, adjust=False).mean() - p.ewm(span=26, adjust=False).mean()
        local["MACD_pct"] = macd / p
        local["MACD_signal_pct"] = macd.ewm(span=9, adjust=False).mean() / p
        change = p.diff()
        gain, loss = (
            change.clip(lower=0).rolling(14).mean(),
            (-change.clip(upper=0)).rolling(14).mean(),
        )
        local["RSI14"] = (
            (100 - 100 / (1 + gain / loss.replace(0, np.nan)))
            .mask((loss == 0) & (gain > 0), 100)
            .mask((loss == 0) & (gain == 0), 50)
        )
        local["ROC10"] = p.pct_change(10, fill_method=None)
        range14 = high.rolling(14).max() - low.rolling(14).min()
        local["stochastic_k"] = (
            100 * (p - low.rolling(14).min()) / range14.replace(0, np.nan)
        ).mask(range14 == 0, 50)
        local["stochastic_d"] = local["stochastic_k"].rolling(3).mean()
        for w in (20, 60):
            local[f"volatility_{w}d"] = r.rolling(w).std()
        tr = pd.concat([high - low, (high - p.shift()).abs(), (low - p.shift()).abs()], axis=1).max(
            axis=1
        )
        local["ATR14_pct"] = tr.rolling(14).mean() / p
        mid, sd = p.rolling(20).mean(), p.rolling(20).std()
        local["BB_width"] = 4 * sd / mid
        local["BB_position"] = ((p - (mid - 2 * sd)) / (4 * sd).replace(0, np.nan)).mask(
            sd == 0, 0.5
        )
        vmean, vsd = volume.rolling(20).mean(), volume.rolling(20).std()
        local["Volume_z"] = ((volume - vmean) / vsd.replace(0, np.nan)).mask(vsd == 0, 0)
        local["Volume_change"] = (volume / volume.shift().replace(0, np.nan) - 1).mask(
            volume.shift() == 0, 0
        )
        obv = (np.sign(p.diff()).fillna(0) * volume).cumsum()
        local["OBV_change_20d"] = (obv.diff(20) / volume.rolling(20).sum().replace(0, np.nan)).mask(
            volume.rolling(20).sum() == 0, 0
        )
        local["volume_price_divergence"] = local["Volume_z"] * -np.sign(r)
        for lag in (1, 2, 3, 5):
            local[f"ret_lag{lag}"] = r.shift(lag)
            local[f"close_lag{lag}_ratio"] = p.shift(lag) / p
        columns.update({f"{ticker}_{k}": v for k, v in local.items()})
        returns.append(r)
        above_ma.append((p > p.rolling(50).mean()).astype(float))
    features = pd.DataFrame(columns, index=prices.index)
    features["market_return"] = np.log(prices[("Adj Close", benchmark)]).diff()
    features["VIX"] = prices[("Close", "^VIX")]
    features["VIX_change"] = features.VIX.pct_change(fill_method=None)
    features["universe_breadth_up"] = (pd.concat(returns, axis=1) > 0).mean(axis=1)
    features["universe_breadth_above_ma50"] = pd.concat(above_ma, axis=1).mean(axis=1)
    features["day_of_week"] = features.index.dayofweek
    features["month"] = features.index.month
    features["turn_of_month_proxy"] = (
        (features.index.day <= 3) | (features.index.day >= 28)
    ).astype(int)
    features = features.join(macro)
    features["yield_spread"] = features.DGS10 - features.DGS3MO
    return features.replace([np.inf, -np.inf], np.nan)


def prepare(config):
    r = config["regression"]
    raw_path, macro_path = [
        project_path(config, "data", "raw", f"{s}.parquet") for s in ("prices", "macro")
    ]
    if not raw_path.exists() or not macro_path.exists():
        raise FileNotFoundError(
            "Need data/raw/prices.parquet and data/raw/macro.parquet. Copy them from the existing project or run python -m src.prd rebuild."
        )
    raw, macro_raw = pd.read_parquet(raw_path), pd.read_parquet(macro_path)
    raw = canonical_prices(raw)
    start = max(pd.Timestamp(config["start_date"]), raw.index.min())
    end = min(pd.Timestamp(r["end_date"]), raw.index.max())
    if (end - start).days / 365.25 < r["min_history_years"]:
        raise ValueError(f"Price history is shorter than {r['min_history_years']} years.")
    sessions = exchange_sessions(start, end, r["calendar"])
    prices, quality = clean_prices(
        raw.loc[start:end],
        sessions,
        config["universe"] + [config["benchmark"], "^VIX"],
        r["price_fill_limit"],
    )
    macro = align_macro(macro_raw, sessions, r["macro_lag_days"])
    features = engineer_features(prices, macro, config["universe"], config["benchmark"])
    valid = features.notna().all(axis=1)
    if not valid.any():
        raise ValueError(
            "No complete feature rows after warm-up. Check macro availability and price history."
        )
    first = int(np.flatnonzero(valid.to_numpy())[0])
    features = features.iloc[first:]
    if features.isna().any().any():
        bad = features.index[features.isna().any(axis=1)]
        raise ValueError(f"Internal feature gaps cannot be dropped silently: {bad[:8].tolist()}")
    adjusted = prices["Adj Close"].loc[features.index, config["universe"] + [config["benchmark"]]]
    dest = prepared_dir(config)
    dest.mkdir(parents=True, exist_ok=True)
    features.to_parquet(dest / "features.parquet")
    adjusted.to_parquet(dest / "adjusted_prices.parquet")
    prices.to_parquet(dest / "clean_prices.parquet")
    macro.to_parquet(dest / "macro_available.parquet")
    write_json(dest / "quality.json", quality)
    pd.DataFrame(quality["outliers"]).to_csv(dest / "outliers.csv", index=False)
    bounds = split_bounds(len(features), config)
    manifest = {
        "universe": config["universe"],
        "benchmark": config["benchmark"],
        "rows": len(features),
        "features": len(features.columns),
        "first_date": features.index[0],
        "last_date": features.index[-1],
        "warmup_rows_removed": first,
        "horizons": r["horizons"],
        "sequence_length": config["sequence_length"],
        "split_bounds": {
            k: {"lo": lo, "hi": hi, "start": features.index[lo], "end": features.index[hi - 1]}
            for k, (lo, hi) in bounds.items()
        },
        "raw_hashes": {
            str(p.relative_to(project_path(config))): sha256(p) for p in (raw_path, macro_path)
        },
        "prepared_hashes": {
            p.name: sha256(p) for p in (dest / "features.parquet", dest / "adjusted_prices.parquet")
        },
        "macro_lag_days": r["macro_lag_days"],
        "macro_vintage_status": "Current FRED snapshot with observation-date lags; historical release vintages are not verified.",
        "test_status": "historical_exposed"
        if features.index[bounds["test"][0]] <= pd.Timestamp(r["historical_test_exposed_through"])
        else "later_window_requires_exposure_review",
        "direction_ties": "Strict up/down/unchanged sign match from current adjusted price",
        "universe_selection_limitation": "A fixed set of surviving large-cap stocks; results may reflect survivorship/selection bias.",
    }
    write_json(dest / "manifest.json", manifest)
    summary = [
        "# Data quality report",
        "",
        f"Raw rows: {quality['raw_rows']}; exchange sessions: {quality['session_rows']}; model rows: {len(features)}.",
        f"Duplicate records: {quality['duplicate_rows']}; conflicting dates: {len(quality['duplicate_conflict_dates'])}.",
        f"Filled ticker/session rows: {quality['filled_rows']}; retained outlier flags: {quality['outliers_flagged']}.",
        f"Remaining price gaps: {quality['remaining_missing_cells']}; OHLC violations: {quality['invariant_violations']}.",
        f"Feature warm-up removed: {first} initial sessions. No internal sessions were dropped.",
        "",
        "Price gaps are filled only when the full run is at most the configured limit. Longer gaps and OHLC violations stop preparation. Outliers are retained for review; none are automatically winsorised.",
        "Macro observations are shifted by documented calendar-day lags before as-of filling. These lags do not remove revisions in current-vintage FRED history. See quality.json and outliers.csv for detailed audit records.",
    ]
    (dest / "data_quality_report.md").write_text("\n\n".join(summary) + "\n")
    return manifest


def rebuild(config):
    import yfinance as yf
    from pandas_datareader import data as web

    end = pd.Timestamp(config["regression"]["end_date"])
    raw_dir = project_path(config, "data", "raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    tickers = config["universe"] + [config["benchmark"], "^VIX"]
    prices = yf.download(
        tickers,
        start=config["start_date"],
        end=(end + pd.Timedelta(days=1)).date().isoformat(),
        auto_adjust=False,
    )
    if prices.empty:
        raise ValueError("Price provider returned an empty dataset.")
    macro = web.DataReader(
        list(config["regression"]["macro_lag_days"]), "fred", config["start_date"], end
    )
    prices.to_parquet(raw_dir / "prices.parquet")
    macro.to_parquet(raw_dir / "macro.parquet")
    return prepare(config)
