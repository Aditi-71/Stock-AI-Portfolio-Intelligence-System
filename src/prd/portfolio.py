"""Forecast-informed, capped mean-variance portfolios with delayed execution."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

from .config import prepared_dir, write_json
from .data import load_prepared
from .experiment import resolve_run, verify_analysis_inputs
from .maths import risk_metrics, sharpe_bootstrap


def optimise(mu, covariance, annual_rf=0.04, cap=0.3, target=None, objective="sharpe"):
    mu, covariance = np.asarray(mu, dtype=float), np.asarray(covariance, dtype=float)
    n = len(mu)
    if (
        n < 2
        or covariance.shape != (n, n)
        or not np.isfinite(mu).all()
        or not np.isfinite(covariance).all()
    ):
        raise ValueError("Invalid optimiser inputs.")
    if not 0 < cap <= 1 or n * cap < 1 - 1e-10:
        raise ValueError("The per-asset cap makes a fully invested allocation infeasible.")
    rf = 252 * np.expm1(np.log1p(annual_rf) / 252)
    if np.isclose(n * cap, 1):
        weights = np.full(n, 1 / n)
        if target is not None and not np.isclose(weights @ mu, target):
            raise ValueError("Target return is infeasible with fixed equal weights.")
        return weights

    def loss(w):
        variance = float(w @ covariance @ w)
        if variance <= 0:
            return 1e10
        return variance if objective == "variance" else -float((w @ mu - rf) / np.sqrt(variance))

    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    if target is not None:
        constraints.append({"type": "eq", "fun": lambda w: w @ mu - target})
    starts = [np.full(n, 1 / n)]
    if objective == "sharpe":
        starts.append(optimise(mu, covariance, annual_rf, cap, objective="variance"))
    solutions = []
    for initial in starts:
        result = minimize(
            loss,
            initial,
            method="SLSQP",
            bounds=[(0, cap)] * n,
            constraints=constraints,
            options={"maxiter": 2000, "ftol": 1e-12},
        )
        if (
            result.success
            and abs(result.x.sum() - 1) < 1e-7
            and result.x.min() >= -1e-8
            and result.x.max() <= cap + 1e-8
        ):
            if target is None or abs(result.x @ mu - target) < 1e-6:
                solutions.append(result)
    if not solutions:
        raise RuntimeError(
            f"Portfolio solver failed for {objective}, target={target}; no invalid weights were accepted."
        )
    return min(solutions, key=lambda result: result.fun).x


def random_portfolios(mu, covariance, cap=0.3, count=20000, seed=42):
    mu = np.asarray(mu)
    n = len(mu)
    if n * cap < 1 - 1e-10:
        raise ValueError("Infeasible cap")
    if np.isclose(n * cap, 1):
        weights = np.tile(np.full(n, 1 / n), (count, 1))
    else:
        rng, accepted, total = np.random.default_rng(seed), [], 0
        for _ in range(500):
            batch = rng.dirichlet(np.ones(n), size=max(2000, min(20000, 2 * (count - total))))
            batch = batch[batch.max(axis=1) <= cap]
            accepted.append(batch)
            total += len(batch)
            if total >= count:
                break
        if total < count:
            raise RuntimeError(
                "Too few random feasible portfolios for this tight cap; the Monte Carlo check is incomplete."
            )
        weights = np.concatenate(accepted)[:count]
    return pd.DataFrame(
        {
            "expected_return": weights @ mu,
            "volatility": np.sqrt(np.einsum("ij,jk,ik->i", weights, covariance, weights)),
        }
    )


def efficient_frontier(mu, covariance, annual_rf=0.04, cap=0.3, points=40):
    minimum = optimise(mu, covariance, annual_rf, cap, objective="variance")
    greedy, remaining = np.zeros(len(mu)), 1.0
    for i in np.argsort(mu)[::-1]:
        greedy[i] = min(cap, remaining)
        remaining -= greedy[i]
    rows = []
    for target in np.linspace(minimum @ mu, greedy @ mu, points):
        weights = optimise(
            mu, covariance, annual_rf, cap, target=float(target), objective="variance"
        )
        rows.append(
            {
                "expected_return": float(weights @ mu),
                "volatility": float(np.sqrt(weights @ covariance @ weights)),
            }
        )
    return pd.DataFrame(rows)


def fixed_holdings_returns(asset_returns, weights, cost_bps=0):
    """Buy once; weights drift with relative performance. Cost charged once."""
    r, w = np.asarray(asset_returns, dtype=float), np.asarray(weights, dtype=float)
    if r.ndim != 2 or r.shape[1] != len(w) or (r <= -1).any() or not np.isclose(w.sum(), 1):
        raise ValueError("Invalid fixed-holdings inputs.")
    wealth = (np.cumprod(1 + r, axis=0) @ w) / (1 + cost_bps / 10000.0)
    return wealth / np.r_[1.0, wealth[:-1]] - 1


def run_portfolio(config, run_dir=None, horizon=1):
    output = resolve_run(config, run_dir)
    _, prices, data_manifest = load_prepared(config)
    verify_analysis_inputs(config, output, data_manifest)
    manifest = json.loads((output / "manifest.json").read_text())
    selected = pd.read_csv(output / "selected_models.csv")
    selected = selected[selected.horizon == horizon]
    if set(selected.ticker) != set(config["universe"]):
        raise ValueError(
            "Portfolio analysis needs a completed run for every configured equity and the selected horizon."
        )
    predictions = pd.read_csv(
        output / "predictions.csv", parse_dates=["origin_date", "target_date"]
    )
    chosen = predictions.merge(
        selected[["ticker", "horizon", "selected_model"]], on=["ticker", "horizon"]
    )
    chosen = chosen[
        (chosen.model == chosen.selected_model)
        & (chosen.split == "test")
        & (chosen.horizon == horizon)
    ]
    forward = chosen.pivot(
        index="origin_date", columns="ticker", values="predicted_return"
    ).dropna()
    if forward.empty:
        raise ValueError("No shared forecast date for the full portfolio universe.")
    decision = forward.index[0]
    historical = prices.loc[:decision].pct_change(fill_method=None).dropna()
    universe = config["universe"]
    mu_history = historical[universe].mean().to_numpy() * 252
    mu_forecast = np.expm1(forward.loc[decision, universe].to_numpy() / horizon) * 252
    r = config["regression"]
    blend = float(r["forecast_blend"])
    if not 0 <= blend <= 1:
        raise ValueError("forecast_blend must be in [0,1]")
    mu = (1 - blend) * mu_history + blend * mu_forecast
    sigma = LedoitWolf().fit(historical[universe]).covariance_ * 252
    weights = {
        "Equal weight": np.full(len(universe), 1 / len(universe)),
        "Minimum variance": optimise(
            mu, sigma, r["annual_risk_free_rate"], r["weight_cap"], objective="variance"
        ),
        "Maximum Sharpe": optimise(mu, sigma, r["annual_risk_free_rate"], r["weight_cap"]),
    }
    clean = pd.read_parquet(prepared_dir(config) / "clean_prices.parquet")
    adjusted_open = clean["Open"] * clean["Adj Close"] / clean["Close"]
    adjusted_open = adjusted_open.loc[prices.index]
    entry_position = int(adjusted_open.index.get_loc(decision)) + 1
    if entry_position + 1 >= len(adjusted_open):
        raise ValueError("Not enough future opens to evaluate the allocation.")
    entry_date = adjusted_open.index[entry_position]
    future = adjusted_open.iloc[entry_position:].pct_change(fill_method=None).iloc[1:]
    returns = pd.DataFrame(index=future.index)
    for name, w in weights.items():
        returns[name] = fixed_holdings_returns(future[universe], w, r["transaction_cost_bps"])
    returns["Benchmark"] = fixed_holdings_returns(
        future[[config["benchmark"]]], [1.0], r["transaction_cost_bps"]
    )
    dest = output / "portfolio"
    dest.mkdir(exist_ok=True)
    from .recommend import build_recommendations

    strategy = build_recommendations(
        config, output, chosen, prices, weights["Maximum Sharpe"], adjusted_open
    )
    if not strategy.index.equals(returns.index):
        raise ValueError("Recommendation and portfolio evaluation dates differ.")
    returns["Recommendations"] = strategy
    records = []
    for name in returns:
        stats = risk_metrics(returns[name], returns["Benchmark"], r["annual_risk_free_rate"])
        interval = sharpe_bootstrap(
            returns[name],
            r["annual_risk_free_rate"],
            r["bootstrap_samples"],
            r["bootstrap_block"],
            config["random_seed"],
        )
        records.append(
            {
                "strategy": name,
                **stats,
                "sharpe_ci_lower": interval["lower"],
                "sharpe_ci_upper": interval["upper"],
            }
        )
    returns.to_csv(dest / "returns.csv", index_label="Date")
    pd.DataFrame(records).to_csv(dest / "risk_metrics.csv", index=False)
    pd.DataFrame(weights, index=universe).to_csv(dest / "weights.csv", index_label="ticker")
    pd.DataFrame(
        {"historical_mu": mu_history, "forecast_mu": mu_forecast, "blended_mu": mu}, index=universe
    ).to_csv(dest / "expected_returns.csv", index_label="ticker")
    pd.DataFrame(sigma, index=universe, columns=universe).to_csv(
        dest / "covariance.csv", index_label="ticker"
    )
    frontier = efficient_frontier(mu, sigma, r["annual_risk_free_rate"], r["weight_cap"])
    cloud = random_portfolios(mu, sigma, r["weight_cap"], r["mc_portfolios"], config["random_seed"])
    frontier.to_csv(dest / "frontier.csv", index=False)
    cloud.to_csv(dest / "random_portfolios.csv", index=False)
    min_vol = float(np.sqrt(weights["Minimum variance"] @ sigma @ weights["Minimum variance"]))
    write_json(
        dest / "metadata.json",
        {
            "decision_close": decision,
            "entry_open": entry_date,
            "first_return_date": returns.index[0],
            "last_return_date": returns.index[-1],
            "forecast_horizon": horizon,
            "expected_return_source": f"{1 - blend:.0%} historical arithmetic mean + {blend:.0%} model-implied daily arithmetic return, annualised by 252",
            "weights_policy": "One initial allocation, then fixed holdings; weights drift. No daily free rebalancing.",
            "execution": "Signal after close t; fill at adjusted open t+1; value at subsequent opens.",
            "cost_bps_one_way": r["transaction_cost_bps"],
            "test_exposure": manifest["test_exposure"],
            "monte_carlo_count": len(cloud),
            "mc_minimum_variance_check_passed": bool(cloud.volatility.min() >= min_vol - 1e-6),
            "limitations": "Expected returns are noisy; no bid/ask spread, market impact, tax or realistic fill simulation. The 1-day close forecast is a view at t, not an exact forecast of the following open-to-open trading return.",
        },
    )
    return dest
