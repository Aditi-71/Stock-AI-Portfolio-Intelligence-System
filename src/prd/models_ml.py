"""The four classical regressor families, tuned on purged training folds."""

from __future__ import annotations

import json

import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from .data import walk_forward_blocks
from .metrics import reconstruct_prices, regression_metrics


def candidates(name):
    if name == "linear":
        return [{"alpha": a} for a in (0.0, 0.1, 1.0, 10.0)]
    if name == "random_forest":
        return [
            {"n_estimators": n, "max_depth": d, "min_samples_leaf": leaf}
            for n, d, leaf in ((200, 5, 5), (400, 8, 10), (400, None, 20))
        ]
    if name == "xgboost":
        return [
            {"n_estimators": n, "max_depth": d, "learning_rate": lr, "subsample": s}
            for n, d, lr, s in ((200, 2, 0.03, 0.8), (400, 3, 0.03, 0.8), (200, 3, 0.05, 1.0))
        ]
    if name == "svr":
        return [
            {"C": c, "epsilon": e, "gamma": g, "kernel": k}
            for c, e, g, k in (
                (0.1, 0.1, "scale", "rbf"),
                (1.0, 0.1, "scale", "rbf"),
                (10.0, 0.2, 0.01, "rbf"),
                (1.0, 0.1, "scale", "linear"),
            )
        ]
    raise ValueError(f"Unknown classical model: {name}")


def build_classical(name, params, config):
    seed, jobs = config["random_seed"], config["regression"]["n_jobs"]
    if name == "linear":
        reg = LinearRegression() if params["alpha"] == 0 else Ridge(alpha=params["alpha"])
    elif name == "random_forest":
        reg = RandomForestRegressor(**params, random_state=seed, n_jobs=jobs)
    elif name == "xgboost":
        from xgboost import XGBRegressor

        reg = XGBRegressor(
            **params,
            objective="reg:squarederror",
            tree_method="hist",
            random_state=seed,
            n_jobs=jobs,
            colsample_bytree=0.8,
            verbosity=0,
        )
    elif name == "svr":
        reg = SVR(**params)
    else:
        raise ValueError(name)
    return TransformedTargetRegressor(
        regressor=make_pipeline(StandardScaler(), reg), transformer=StandardScaler()
    )


def tune_classical(name, features, targets, horizon, config, progress=print):
    records, scores = [], []
    for candidate, params in enumerate(candidates(name)):
        fold_scores = []
        for fold, train, validation in walk_forward_blocks(features, targets, horizon, config):
            estimator = build_classical(name, params, config).fit(train.X, train.y)
            prediction = reconstruct_prices(
                validation.rows.current_price.to_numpy(), estimator.predict(validation.X)
            )
            metrics = regression_metrics(
                validation.rows.actual_price, prediction, validation.rows.current_price
            )
            records.append(
                {
                    "model": name,
                    "candidate": candidate,
                    "params": json.dumps(params),
                    "fold": fold,
                    "train_end": str(train.rows.index[-1].date()),
                    "last_train_target": str(train.rows.target_date.max().date()),
                    "validation_start": str(validation.rows.index[0].date()),
                    **metrics,
                }
            )
            fold_scores.append(metrics["mae"])
        mean = float(np.mean(fold_scores))
        scores.append(mean)
        progress(f"  {name}: candidate {candidate + 1}, mean CV price MAE={mean:.6f}")
    winner = int(np.argmin(scores))
    return candidates(name)[winner], records
