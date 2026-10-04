"""Reproducible model comparison without selecting winners on the test set."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from .config import DEEP_MODELS, REQUIRED_MODELS, project_path, sha256, write_json
from .data import (
    blocks_for,
    load_prepared,
    make_block,
    regression_targets,
    split_bounds,
    walk_forward_blocks,
)
from .maths import covariance, gradient_descent_linear
from .metrics import calibration_radius, predictions_frame, score_frame
from .models_ml import build_classical, tune_classical


def predict(estimator, block):
    return (
        estimator.predict_block(block)
        if hasattr(estimator, "predict_block")
        else estimator.predict(block.X)
    )


def preflight(models):
    missing = []
    if "xgboost" in models and importlib.util.find_spec("xgboost") is None:
        missing.append("xgboost")
    if set(models).intersection(DEEP_MODELS) and importlib.util.find_spec("tensorflow") is None:
        missing.append("tensorflow")
    if missing:
        raise ModuleNotFoundError(
            f"Install the required packages before this run: {', '.join(missing)}. See requirements-prd.txt and RUN_PRD.md."
        )


def environment_versions():
    versions = {"python": sys.version.split()[0]}
    for name in (
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "xgboost",
        "tensorflow",
        "pyarrow",
        "joblib",
    ):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not_installed"
    return versions


def math_checks(features, prices, config):
    train_end = split_bounds(len(features), config)["train"][1]
    cols = [f"{t}_ret_1d" for t in config["universe"][:3]]
    cols = [c for c in cols if c in features]
    x = StandardScaler().fit_transform(features.iloc[: train_end - 1][cols])
    y = np.log(prices[config["universe"][0]]).diff().shift(-1).iloc[: train_end - 1].to_numpy()
    gd = gradient_descent_linear(x, y)
    ols = LinearRegression().fit(x, y)
    gd_prediction = gd["intercept"] + x @ gd["coef"]
    error = float(np.max(np.abs(gd_prediction - ols.predict(x))))
    returns = prices.iloc[:train_end].pct_change(fill_method=None).dropna().to_numpy()
    return {
        "gradient_descent": {
            **gd,
            "features": cols,
            "rank": int(np.linalg.matrix_rank(x)),
            "max_prediction_difference_from_sklearn": error,
            "max_coefficient_difference": float(np.max(np.abs(gd["coef"] - ols.coef_))),
            "passed": bool(gd["converged"] and error < 1e-6),
        },
        "covariance_max_difference_from_numpy": float(
            np.max(np.abs(covariance(returns) - np.cov(returns, rowvar=False, ddof=1)))
        ),
    }


def run_experiment(
    config, models=None, tickers=None, horizons=None, run_name=None, inputs=None, progress=print
):
    models = list(dict.fromkeys(models or ["naive", *REQUIRED_MODELS]))
    if "naive" not in models:
        models.insert(0, "naive")
    unknown = set(models) - {"naive", *REQUIRED_MODELS}
    if unknown:
        raise ValueError(f"Unknown models: {sorted(unknown)}")
    preflight(models)
    features, prices, data_manifest = load_prepared(config) if inputs is None else inputs
    tickers = tickers or config["universe"]
    horizons = horizons or config["regression"]["horizons"]
    if set(tickers) - set(config["universe"]) or set(horizons) - set(
        config["regression"]["horizons"]
    ):
        raise ValueError("Requested ticker/horizon is not in config.yaml.")
    name = run_name or datetime.now(UTC).strftime("run-%Y%m%d-%H%M%S")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
        raise ValueError("Use a simple run name with letters, digits, hyphens or underscores.")
    output = project_path(config, "reports", "prd", name)
    if output.exists():
        raise FileExistsError(
            f"Run already exists: {output}. Choose another --run-name to preserve its results."
        )
    output.mkdir(parents=True)
    source = Path(__file__).parent
    manifest = {
        "run_name": name,
        "started_utc": datetime.now(UTC).isoformat(),
        "status": "running",
        "models": models,
        "tickers": tickers,
        "horizons": horizons,
        "config": config,
        "data": data_manifest,
        "environment": environment_versions(),
        "source_hashes": {p.name: sha256(p) for p in sorted(source.glob("*.py"))},
        "selection_rule": "Minimum price MAE on first half of validation; ties follow configured model order. Test never selects a model.",
        "interval_method": "Absolute log-return residual quantile on reserved calibration block; time-series dependence means nominal coverage is not guaranteed.",
        "fit_policy": "Final models train on the training block only; validation selects models/early-stopping, calibration sets bands, test evaluates.",
        "test_exposure": data_manifest.get("test_status", "unknown"),
        "full_model_coverage": set(REQUIRED_MODELS).issubset(models)
        and set(tickers) == set(config["universe"])
        and set(horizons) == set(config["regression"]["horizons"]),
    }
    write_json(output / "manifest.json", manifest)
    prices.to_csv(output / "adjusted_prices.csv", index_label="Date")
    frozen = sorted(
        {
            f"{d.metadata['Name']}=={d.version}"
            for d in importlib.metadata.distributions()
            if d.metadata.get("Name")
        }
    )
    (output / "environment.lock.txt").write_text("\n".join(frozen) + "\n")
    all_predictions, metrics, cv_records, selected, settings = [], [], [], [], []
    try:
        checks = math_checks(features, prices, config)
        write_json(output / "math_checks.json", checks)
        if not checks["gradient_descent"]["passed"]:
            progress(
                "Manual gradient-descent comparison did not converge to tolerance; see math_checks.json. This acceptance check remains open."
            )
        for ticker in tickers:
            for horizon in horizons:
                progress(f"\n{ticker} — horizon {horizon} trading session(s)")
                blocks = blocks_for(features, prices, ticker, horizon, config)
                targets = regression_targets(prices, ticker, horizon)
                validation_scores = {}
                for model_name in models:
                    progress(f"Training/evaluating {model_name}...")
                    artifact = output / "models" / f"{ticker}_h{horizon}" / model_name
                    artifact.mkdir(parents=True)
                    params = {}
                    if model_name == "naive":
                        estimator = None
                        for fold, _tr, va in walk_forward_blocks(
                            features, targets, horizon, config
                        ):
                            pf = predictions_frame(
                                va,
                                np.zeros(len(va.origins)),
                                ticker,
                                horizon,
                                model_name,
                                "walk_forward",
                            )
                            cv_records.append(
                                {
                                    "ticker": ticker,
                                    "horizon": horizon,
                                    "model": model_name,
                                    "candidate": 0,
                                    "fold": fold,
                                    **score_frame(pf),
                                }
                            )
                    elif model_name in DEEP_MODELS:
                        from .models_dl import fit_deep

                        for fold, outer_train, outer_val in walk_forward_blocks(
                            features, targets, horizon, config
                        ):
                            stop_size = max(
                                int(outer_train.hi * 0.2), config["sequence_length"] + horizon + 2
                            )
                            inner_stop = outer_train.hi - stop_size
                            inner_train = make_block(
                                features, targets, 0, inner_stop, horizon, config["sequence_length"]
                            )
                            inner_val = make_block(
                                features,
                                targets,
                                inner_stop,
                                outer_train.hi,
                                horizon,
                                config["sequence_length"],
                            )
                            fold_model = fit_deep(model_name, inner_train, inner_val, config)
                            pf = predictions_frame(
                                outer_val,
                                predict(fold_model, outer_val),
                                ticker,
                                horizon,
                                model_name,
                                "walk_forward",
                            )
                            cv_records.append(
                                {
                                    "ticker": ticker,
                                    "horizon": horizon,
                                    "model": model_name,
                                    "candidate": 0,
                                    "fold": fold,
                                    "early_stop_end": str(inner_val.rows.target_date.max().date()),
                                    **score_frame(pf),
                                }
                            )
                            pd.DataFrame(fold_model.history).to_csv(
                                artifact / f"fold_{fold}_loss.csv", index_label="epoch"
                            )
                            del fold_model
                        estimator = fit_deep(
                            model_name, blocks["train"], blocks["validation"], config
                        )
                        estimator.save(artifact)
                        pd.DataFrame(estimator.history).to_csv(
                            artifact / "training_loss.csv", index_label="epoch"
                        )
                        params = {
                            "epochs_run": len(estimator.history["loss"]),
                            "loss": "Huber on scaled forward log returns",
                            "learning_rate": config["regression"]["learning_rate"],
                            "dropout": config["regression"]["dropout"],
                        }
                    else:
                        params, records = tune_classical(
                            model_name, features, targets, horizon, config, progress
                        )
                        cv_records.extend(
                            {"ticker": ticker, "horizon": horizon, **row} for row in records
                        )
                        estimator = build_classical(model_name, params, config).fit(
                            blocks["train"].X, blocks["train"].y
                        )
                        joblib.dump(estimator, artifact / "model.joblib")
                    cal_prediction = (
                        np.zeros(len(blocks["calibration"].origins))
                        if estimator is None
                        else predict(estimator, blocks["calibration"])
                    )
                    radius = calibration_radius(
                        blocks["calibration"].y,
                        cal_prediction,
                        config["regression"]["interval_alpha"],
                    )
                    write_json(
                        artifact / "metadata.json",
                        {
                            "ticker": ticker,
                            "horizon": horizon,
                            "model": model_name,
                            "params": params,
                            "interval_radius_log_return": radius,
                            "feature_columns": list(features.columns),
                        },
                    )
                    for split in ("validation", "calibration", "test"):
                        block = blocks[split]
                        predictions = (
                            np.zeros(len(block.origins))
                            if estimator is None
                            else predict(estimator, block)
                        )
                        frame = predictions_frame(
                            block,
                            predictions,
                            ticker,
                            horizon,
                            model_name,
                            split,
                            radius if split == "test" else None,
                        )
                        all_predictions.append(frame)
                        row = {
                            "ticker": ticker,
                            "horizon": horizon,
                            "model": model_name,
                            "split": split,
                            **score_frame(frame),
                        }
                        metrics.append(row)
                        if split == "validation":
                            validation_scores[model_name] = row["mae"]
                    settings.append(
                        {
                            "ticker": ticker,
                            "horizon": horizon,
                            "model": model_name,
                            "params": json.dumps(params),
                        }
                    )
                    pd.DataFrame(metrics).to_csv(output / "metrics.csv", index=False)
                    pd.DataFrame(cv_records).to_csv(output / "cv_metrics.csv", index=False)
                learned = [m for m in models if m != "naive"]
                winner = min(learned, key=lambda m: validation_scores[m]) if learned else "naive"
                selected.append(
                    {
                        "ticker": ticker,
                        "horizon": horizon,
                        "selected_model": winner,
                        "validation_mae": validation_scores[winner],
                        "naive_validation_mae": validation_scores["naive"],
                    }
                )
        predictions = pd.concat(all_predictions, ignore_index=True)
        keys = ["ticker", "horizon", "split"]
        for _, group in predictions.groupby(keys):
            origin_sets = group.groupby("model").origin_date.apply(lambda x: tuple(x))
            if len(set(origin_sets)) != 1:
                raise RuntimeError("Models were evaluated on different forecast origins.")
        predictions.to_csv(output / "predictions.csv", index=False)
        pd.DataFrame(selected).to_csv(output / "selected_models.csv", index=False)
        pd.DataFrame(settings).to_csv(output / "selected_parameters.csv", index=False)
        frame = pd.DataFrame(metrics)
        cols = ["rmse", "mae", "mape_pct", "r2", "directional_accuracy_pct"]
        frame.groupby(["horizon", "model", "split"])[cols].mean().reset_index().to_csv(
            output / "macro_average_metrics.csv", index=False
        )
        manifest["status"] = "completed"
        manifest["completed_utc"] = datetime.now(UTC).isoformat()
        manifest["acceptance_note"] = (
            "A completed code run does not certify PRD completion or a genuinely untouched test set. See data/test exposure and PRD_STATUS.md."
        )
        write_json(output / "manifest.json", manifest)
        write_json(
            project_path(config, "reports", "prd", "latest.json"),
            {"run_dir": str(output.relative_to(project_path(config)))},
        )
        return output
    except Exception as error:
        manifest["status"], manifest["error"] = "failed", f"{type(error).__name__}: {error}"
        write_json(output / "manifest.json", manifest)
        raise


def resolve_run(config, run_dir=None):
    if run_dir is not None:
        path = Path(run_dir)
        path = path if path.is_absolute() else project_path(config, str(path))
    else:
        latest = project_path(config, "reports", "prd", "latest.json")
        if not latest.exists():
            raise FileNotFoundError("No completed PRD run. Run python -m src.prd train first.")
        path = project_path(config, json.loads(latest.read_text())["run_dir"])
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest["status"] != "completed":
        raise ValueError("Only completed model runs can feed analysis.")
    return path


def verify_analysis_inputs(config, output, data_manifest):
    """Avoid combining an old model run with subsequently rebuilt input data."""
    manifest = json.loads((output / "manifest.json").read_text())
    for key in (
        "universe",
        "benchmark",
        "sequence_length",
        "train_ratio",
        "validation_ratio",
        "test_ratio",
    ):
        if manifest["config"][key] != config[key]:
            raise ValueError(
                f"{key} differs from the selected experiment. Use its original configuration."
            )
    recorded = manifest["data"].get("prepared_hashes")
    if recorded is not None and recorded != data_manifest.get("prepared_hashes"):
        raise ValueError(
            "The current prepared dataset differs from this model run. Restore its data snapshot or train a new run."
        )
