"""Financial mathematics and a checked gradient-descent example."""

from __future__ import annotations

import numpy as np


def covariance(returns):
    x = np.asarray(returns, dtype=float)
    if x.ndim != 2 or len(x) < 2 or not np.isfinite(x).all():
        raise ValueError("Covariance requires at least two finite observations.")
    centred = x - x.mean(axis=0)
    return centred.T @ centred / (len(x) - 1)


def gradient_descent_linear(X, y, tolerance=1e-10, max_iter=100000):
    X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
    design = np.column_stack([np.ones(len(X)), X])
    if y.shape != (len(X),) or not np.isfinite(design).all() or not np.isfinite(y).all():
        raise ValueError("Invalid gradient descent inputs.")
    theta = np.zeros(design.shape[1])
    hessian = 2 * design.T @ design / len(X)
    forcing = 2 * design.T @ y / len(X)
    rate = 0.9 / np.linalg.eigvalsh(hessian).max()
    history = []
    for epoch in range(max_iter):
        gradient = hessian @ theta - forcing
        theta -= rate * gradient
        if epoch % 100 == 0:
            history.append({"iteration": epoch, "mse": float(np.mean((design @ theta - y) ** 2))})
        if np.linalg.norm(gradient, ord=np.inf) < tolerance:
            break
    return {
        "intercept": theta[0],
        "coef": theta[1:],
        "iterations": epoch + 1,
        "converged": bool(np.linalg.norm(gradient, ord=np.inf) < tolerance),
        "learning_rate": float(rate),
        "history": history,
    }


def risk_metrics(returns, benchmark=None, annual_rf=0.04):
    r = np.asarray(returns, dtype=float)
    if r.ndim != 1 or len(r) < 2 or not np.isfinite(r).all() or (r <= -1).any():
        raise ValueError("Risk metrics need finite simple returns greater than -1.")
    rf_daily = (1 + annual_rf) ** (1 / 252) - 1
    excess = r - rf_daily
    volatility = np.std(r, ddof=1) * np.sqrt(252)
    downside = np.sqrt(np.mean(np.minimum(excess, 0) ** 2)) * np.sqrt(252)
    wealth = np.r_[1.0, np.cumprod(1 + r)]
    drawdown = wealth / np.maximum.accumulate(wealth) - 1
    max_dd = float(drawdown.min())
    cagr = float(wealth[-1] ** (252 / len(r)) - 1)
    q = float(np.quantile(r, 0.05))
    beta = np.nan
    if benchmark is not None:
        b = np.asarray(benchmark, dtype=float)
        if b.shape != r.shape or not np.isfinite(b).all():
            raise ValueError("Benchmark dates/length must align with portfolio returns.")
        cov = covariance(np.column_stack([r, b]))
        if cov[1, 1] > 0:
            beta = float(cov[0, 1] / cov[1, 1])
    return {
        "annual_return_arithmetic": float(r.mean() * 252),
        "cagr": cagr,
        "volatility": float(volatility),
        "sharpe": float(excess.mean() * 252 / volatility) if volatility > 0 else np.nan,
        "sortino": float(excess.mean() * 252 / downside) if downside > 0 else np.nan,
        "max_drawdown": max_dd,
        "calmar": cagr / abs(max_dd) if max_dd < 0 else np.nan,
        "var95_loss": -q,
        "cvar95_loss": float(-r[r <= q].mean()),
        "beta": beta,
        "worst_day": float(r.min()),
        "best_day": float(r.max()),
    }


def sharpe_bootstrap(returns, annual_rf=0.04, samples=1000, block=20, seed=42):
    r = np.asarray(returns, dtype=float)
    rng = np.random.default_rng(seed)
    block = min(int(block), len(r))
    if block < 1 or samples < 2:
        raise ValueError("Invalid bootstrap configuration.")
    nblocks = int(np.ceil(len(r) / block))
    estimates = []
    for _ in range(samples):
        starts = rng.integers(0, len(r) - block + 1, size=nblocks)
        positions = (starts[:, None] + np.arange(block)).ravel()[: len(r)]
        estimates.append(risk_metrics(r[positions], annual_rf=annual_rf)["sharpe"])
    estimates = np.asarray(estimates)
    estimates = estimates[np.isfinite(estimates)]
    return (
        {
            "lower": float(np.quantile(estimates, 0.025)),
            "upper": float(np.quantile(estimates, 0.975)),
            "samples": samples,
            "block_sessions": block,
        }
        if len(estimates)
        else {"lower": None, "upper": None, "samples": samples, "block_sessions": block}
    )
