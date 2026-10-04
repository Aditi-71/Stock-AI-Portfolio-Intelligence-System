"""Transparent research signals and an illustrative rebalancing table."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import project_path, write_json


def recommendation_scores(
    forecast_return, volatility, market_volatility, sentiment, horizon, weights, thresholds
):
    w = np.asarray(weights, dtype=float)
    if w.shape != (3,) or (w < 0).any() or not np.isclose(w.sum(), 1):
        raise ValueError("Signal weights must be three nonnegative numbers summing to 1.")
    low, high = thresholds
    if low >= high:
        raise ValueError("Recommendation thresholds must have low < high.")
    forecast = np.tanh((np.asarray(forecast_return) / horizon) / np.maximum(volatility, 1e-8))
    risk = np.asarray(volatility) / np.maximum(np.asarray(volatility) + market_volatility, 1e-8)
    sentiment_values = np.asarray(sentiment, dtype=float)
    missing = ~np.isfinite(sentiment_values)
    contribution = np.where(missing, 0.0, sentiment_values)
    score = w[0] * forecast + w[1] * contribution - w[2] * risk
    return pd.DataFrame(
        {
            "forecast_subscore": forecast,
            "sentiment_subscore": sentiment_values,
            "sentiment_missing": missing,
            "risk_subscore": risk,
            "composite_score": score,
            "recommendation": np.where(
                score >= high, "BUY", np.where(score <= low, "SELL", "HOLD")
            ),
        }
    )


def execute_recommendations(
    adjusted_open, signals, target_weights, start_origin, cost_bps=10, cap=0.3, band=0.05
):
    """Signal at close t; rebalance at next open, then mark at following open."""
    opens = adjusted_open.astype(float)
    n = opens.shape[1]
    if (
        len(target_weights) != n
        or not np.isfinite(opens.to_numpy()).all()
        or (opens <= 0).any().any()
    ):
        raise ValueError("Invalid open prices or target weights.")
    if cost_bps < 0 or cost_bps >= 10000 or band < 0:
        raise ValueError("Invalid cost or rebalancing band.")
    shares, cash, previous_nav = np.zeros(n), 1.0, 1.0
    records, hits = [], []
    start = int(opens.index.get_loc(start_origin)) + 1
    for j in range(start, len(opens) - 1):
        origin, _entry, exit_date = opens.index[j - 1], opens.index[j], opens.index[j + 1]
        price = opens.iloc[j].to_numpy()
        dollars = shares * price
        nav = cash + dollars.sum()
        current = dollars / nav
        recommendation = (
            signals.loc[origin].to_numpy() if origin in signals.index else np.repeat("HOLD", n)
        )
        desired = current.copy()
        desired[recommendation == "BUY"] = np.asarray(target_weights)[recommendation == "BUY"]
        desired[recommendation == "SELL"] = 0.0
        desired[np.abs(desired - current) <= band] = current[np.abs(desired - current) <= band]
        desired = np.clip(desired, 0, cap)
        if desired.sum() > 1:
            desired /= desired.sum()
        after_cost = nav
        fee = cost_bps / 10000
        for _ in range(100):
            next_nav = nav - fee * np.abs(desired * after_cost - dollars).sum()
            if abs(next_nav - after_cost) < 1e-12:
                after_cost = next_nav
                break
            after_cost = next_nav
        turnover = float(np.abs(desired * after_cost - dollars).sum() / nav)
        shares = desired * after_cost / price
        cash = float((1 - desired.sum()) * after_cost)
        next_open = opens.iloc[j + 1].to_numpy()
        next_value = float(cash + shares @ next_open)
        records.append(
            {
                "Date": exit_date,
                "return": next_value / previous_nav - 1,
                "turnover": turnover,
                "fee_fraction": (nav - after_cost) / nav,
                "cash_weight_after_trade": 1 - desired.sum(),
            }
        )
        if origin in signals.index:
            realised = next_open / price - 1
            for i, ticker in enumerate(opens.columns):
                if recommendation[i] != "HOLD":
                    hits.append(
                        {
                            "origin_date": origin,
                            "ticker": ticker,
                            "recommendation": recommendation[i],
                            "realised_open_to_open_return": realised[i],
                            "hit": bool(
                                (realised[i] > 0 and recommendation[i] == "BUY")
                                or (realised[i] < 0 and recommendation[i] == "SELL")
                            ),
                            "always_long_hit": bool(realised[i] > 0),
                        }
                    )
        previous_nav = next_value
    return pd.DataFrame(records).set_index("Date"), pd.DataFrame(hits)


def build_recommendations(config, output, predictions, prices, target_weights, adjusted_open):
    universe = config["universe"]
    origin = pd.to_datetime(predictions.origin_date).min()
    close_returns = prices.pct_change(fill_method=None)
    rolling_vol = close_returns.rolling(60).std()
    daily_path = project_path(config, "data", "processed", "daily_sentiment.parquet")
    daily = pd.read_parquet(daily_path) if daily_path.exists() else None
    all_signals = []
    for date, group in predictions.groupby("origin_date", sort=True):
        now = group.set_index("ticker").reindex(universe)
        if now.predicted_return.isna().any():
            continue
        volatility = rolling_vol.loc[date, universe].to_numpy()
        market_vol = rolling_vol.loc[date, config["benchmark"]]
        sentiment = pd.Series(np.nan, index=universe)
        if daily is not None:
            day = daily[pd.to_datetime(daily.session_date).dt.normalize() == date]
            if not day.empty:
                sentiment.update(day.groupby("ticker").sentiment_mean.mean())
        horizon = int(now.horizon.iloc[0])
        row = recommendation_scores(
            now.predicted_return.to_numpy(),
            volatility,
            market_vol,
            sentiment.to_numpy(),
            horizon,
            config["regression"]["recommendation_weights"],
            config["regression"]["recommendation_thresholds"],
        )
        row.insert(0, "ticker", universe)
        row["origin_date"] = date
        row["predicted_return"] = now.predicted_return.to_numpy()
        all_signals.append(row)
    signal_table = pd.concat(all_signals, ignore_index=True)
    table = signal_table[signal_table.origin_date == origin].copy()
    table["current_weight"] = 0.0
    table["target_weight"] = target_weights
    table["drift"] = table.target_weight - table.current_weight
    band = config["regression"]["rebalance_band"]
    table["rebalance_action"] = np.where(
        table.drift > band, "INCREASE", np.where(table.drift < -band, "REDUCE", "KEEP")
    )
    dest = output / "portfolio"
    table.to_csv(dest / "recommendations.csv", index=False)
    signal_table.to_csv(dest / "signal_history.csv", index=False)
    signals = signal_table.pivot(
        index="origin_date", columns="ticker", values="recommendation"
    ).reindex(columns=universe)
    strategy, hits = execute_recommendations(
        adjusted_open[universe],
        signals,
        target_weights,
        origin,
        config["regression"]["transaction_cost_bps"],
        config["regression"]["weight_cap"],
        band,
    )
    strategy.to_csv(dest / "recommendation_backtest.csv", index_label="Date")
    hits.to_csv(dest / "recommendation_hits.csv", index=False)
    if not hits.empty:
        hits.groupby("ticker").agg(
            actionable_signals=("hit", "size"),
            hit_rate=("hit", "mean"),
            always_long_hit_rate=("always_long_hit", "mean"),
        ).to_csv(dest / "recommendation_hit_rates.csv")
    write_json(
        dest / "recommendation_metadata.json",
        {
            "current_holdings": "Illustrative all-cash starting portfolio; no actual user holdings supplied.",
            "missing_sentiment": "Unknown sentiment contributes zero but remains explicitly missing; it is not classified as neutral or no-news.",
            "weights": config["regression"]["recommendation_weights"],
            "thresholds": config["regression"]["recommendation_thresholds"],
            "evaluation_status": "Chronological next-open strategy, transaction costs, and hit rates produced. BUY restores the initial target weight, SELL targets zero, HOLD retains the position subject to the concentration cap and rebalancing band. Cash earns zero in this simulation.",
            "hit_rate_definition": "Correct sign of the following tradable open-to-open return among BUY/SELL signals; HOLD excluded. Always-long sign accuracy is measured on the same asset/date rows. Financial performance is compared separately with benchmark buy-and-hold in risk_metrics.csv.",
            "disclaimer": "Educational research, not financial advice.",
        },
    )
    return strategy["return"]
