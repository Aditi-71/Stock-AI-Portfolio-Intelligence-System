"""Resume PRD work in fresh processes, verifying and preserving saved results.

Run from the project root: python recover_prd.py
See README_RECOVERY.md. Only load your own trusted model artifacts.
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.prd.config import DEEP_MODELS, REQUIRED_MODELS, load_config, sha256, write_json
from src.prd.data import blocks_for, load_prepared
from src.prd.experiment import environment_versions, run_experiment
from src.prd.metrics import predictions_frame, score_frame

SPLITS = ("validation", "calibration", "test")


def read_json(path):
    return json.loads(Path(path).read_text())


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    write_json(temp, value)
    temp.replace(path)


def atomic_csv(frame, path):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    frame.to_csv(temp, index=False)
    temp.replace(path)


def normal_config(config):
    return {k: v for k, v in config.items() if k != "project_root"}


def identity(config, data):
    source = Path(__file__).parent / "src" / "prd"
    return {
        "config": normal_config(config),
        "data": data,
        "source_hashes": {p.name: sha256(p) for p in sorted(source.glob("*.py"))},
        "environment": environment_versions(),
        "runner_sha256": sha256(Path(__file__)),
    }


def verify_source(source, expected):
    manifest = read_json(source / "manifest.json")
    for key in ("data", "source_hashes", "environment"):
        if manifest.get(key) != expected[key]:
            raise ValueError(
                f"{source.name}: {key} differs. Restore the original inputs/environment before recovery."
            )
    if normal_config(manifest["config"]) != expected["config"]:
        raise ValueError(
            f"{source.name}: configuration differs. Recovery must not mix experiments."
        )
    return manifest


def subset(frame, ticker, horizon, model):
    return frame[
        (frame.ticker == ticker) & (frame.horizon == horizon) & (frame.model == model)
    ].copy()


def saved_job(source, ticker, horizon, model, config):
    """A model counts as reusable only after metrics, metadata and CV exist."""
    metrics_path, cv_path = source / "metrics.csv", source / "cv_metrics.csv"
    artifact = source / "models" / f"{ticker}_h{horizon}" / model
    if not all(p.exists() for p in (metrics_path, cv_path, artifact / "metadata.json")):
        return False
    metrics = subset(pd.read_csv(metrics_path), ticker, horizon, model)
    if len(metrics) != 3 or set(metrics.split) != set(SPLITS):
        return False
    cv = subset(pd.read_csv(cv_path), ticker, horizon, model)
    if cv.empty:
        return False
    expected = set(range(1, config["regression"]["cv_splits"] + 1))
    if any(set(g.fold) != expected or len(g) != len(expected) for _, g in cv.groupby("candidate")):
        return False
    if model in DEEP_MODELS:
        required = ["model.keras", "scalers.joblib", "training_loss.csv"]
        required += [f"fold_{fold}_loss.csv" for fold in sorted(expected)]
    elif model == "naive":
        required = []
    else:
        required = ["model.joblib"]
    return all((artifact / name).is_file() for name in required)


def verify_checkpoint(directory, expected=None):
    marker = directory / "complete.json"
    if not marker.exists():
        return False
    record = read_json(marker)
    if expected is not None and record["identity"] != expected:
        raise ValueError(f"Checkpoint inputs differ: {directory}")
    for name, digest in record["files"].items():
        path = directory / name
        if not path.is_file() or sha256(path) != digest:
            raise ValueError(f"Checkpoint changed or damaged: {path}")
    return True


def checkpoint(source, dest, ticker, horizon, model, config, inputs, expected, restore=False):
    """Rebuild missing predictions from saved weights and verify old metrics."""
    verify_source(source, expected)
    if not saved_job(source, ticker, horizon, model, config):
        raise ValueError(f"Incomplete saved artifact: {ticker} h{horizon} {model}")
    features, prices, _ = inputs
    blocks = blocks_for(features, prices, ticker, horizon, config)
    artifact = source / "models" / f"{ticker}_h{horizon}" / model
    meta = read_json(artifact / "metadata.json")
    if meta["feature_columns"] != list(features.columns):
        raise ValueError("Saved model feature order differs from prepared data.")
    if (meta["ticker"], meta["horizon"], meta["model"]) != (ticker, horizon, model):
        raise ValueError("Saved model identity differs from requested job.")
    if restore:
        if model in DEEP_MODELS:
            from src.prd.models_dl import DeepRegressor, backend

            tf, _ = backend()
            scalers = joblib.load(artifact / "scalers.joblib")
            estimator = DeepRegressor(
                tf.keras.models.load_model(artifact / "model.keras", compile=False),
                scalers["x_scaler"],
                scalers["y_scaler"],
                {},
            )
        elif model != "naive":
            estimator = joblib.load(artifact / "model.joblib")
        frames = []
        for split, block in blocks.items():
            if split not in SPLITS:
                continue
            if model == "naive":
                prediction = np.zeros(len(block.origins))
            elif model in DEEP_MODELS:
                prediction = estimator.predict_block(block)
            else:
                prediction = estimator.predict(block.X)
            radius = meta["interval_radius_log_return"] if split == "test" else None
            frames.append(
                predictions_frame(block, prediction, ticker, horizon, model, split, radius)
            )
        predictions = pd.concat(frames, ignore_index=True)
    else:
        predictions = subset(pd.read_csv(source / "predictions.csv"), ticker, horizon, model)
    for column in ("origin_date", "target_date"):
        predictions[column] = pd.to_datetime(predictions[column])
    old = subset(pd.read_csv(source / "metrics.csv"), ticker, horizon, model).set_index("split")
    records = []
    for split in SPLITS:
        frame = predictions[predictions.split == split].copy()
        block = blocks[split]
        if len(frame) != len(block.origins):
            raise ValueError("Saved prediction count differs from prepared scoring dates.")
        if not pd.DatetimeIndex(pd.to_datetime(frame.origin_date)).equals(block.rows.index):
            raise ValueError("Saved forecast origins differ from prepared data.")
        if not pd.DatetimeIndex(pd.to_datetime(frame.target_date)).equals(
            pd.DatetimeIndex(block.rows.target_date)
        ):
            raise ValueError("Saved target dates differ from prepared data.")
        if not np.allclose(
            frame[["current_price", "actual_price"]],
            block.rows[["current_price", "actual_price"]],
            rtol=1e-10,
            atol=1e-10,
        ):
            raise ValueError("Saved actual/current prices differ from prepared data.")
        values = score_frame(
            frame.drop(columns=["lower_price", "upper_price"], errors="ignore")
            if split != "test"
            else frame
        )
        for key, value in values.items():
            tolerance = (
                1e-8
                if key
                in (
                    "n",
                    "directional_accuracy_pct",
                    "unchanged_forecast_pct",
                    "interval_coverage_pct",
                )
                else 1e-5
            )
            if not np.isclose(
                value, old.loc[split, key], rtol=tolerance, atol=1e-6, equal_nan=True
            ):
                raise ValueError(
                    f"Recovered {ticker} h{horizon} {model} {split} {key} does not match saved metric."
                )
        records.append(
            {"ticker": ticker, "horizon": horizon, "model": model, "split": split, **values}
        )
    if dest.exists():
        raise FileExistsError(f"Checkpoint destination already exists: {dest}")
    dest.mkdir(parents=True)
    shutil.copytree(artifact, dest / "model")
    atomic_csv(predictions, dest / "predictions.csv")
    atomic_csv(pd.DataFrame(records), dest / "metrics.csv")
    atomic_csv(
        subset(pd.read_csv(source / "cv_metrics.csv"), ticker, horizon, model),
        dest / "cv_metrics.csv",
    )
    atomic_json(
        dest / "origin.json",
        {
            "source": str(source.resolve()),
            "ticker": ticker,
            "horizon": horizon,
            "model": model,
            "restored_predictions": restore,
            "source_manifest_sha256": sha256(source / "manifest.json"),
        },
    )
    files = {p.relative_to(dest).as_posix(): sha256(p) for p in dest.rglob("*") if p.is_file()}
    atomic_json(dest / "complete.json", {"identity": expected, "files": files})


def aggregate(checkpoints, output, config, inputs, expected, source_manifest):
    """Re-select only on validation; write a normal complete PRD run."""
    models = ["naive", *REQUIRED_MODELS]
    frames, cvs, settings, origins = [], [], [], []
    for path in checkpoints:
        if not verify_checkpoint(path, expected):
            raise ValueError(f"Incomplete checkpoint: {path}")
        origin = read_json(path / "origin.json")
        origins.append(origin)
        frames.append(pd.read_csv(path / "predictions.csv"))
        cvs.append(pd.read_csv(path / "cv_metrics.csv"))
        meta = read_json(path / "model/metadata.json")
        ticker, horizon, model = meta["ticker"], meta["horizon"], meta["model"]
        settings.append(
            {
                "ticker": ticker,
                "horizon": horizon,
                "model": model,
                "params": json.dumps(meta["params"]),
            }
        )
        target = output / "models" / f"{ticker}_h{horizon}" / model
        shutil.copytree(path / "model", target, dirs_exist_ok=True)
    predictions = pd.concat(frames, ignore_index=True)
    key = ["ticker", "horizon", "model", "split", "origin_date"]
    if predictions.duplicated(key).any():
        raise ValueError("Duplicate predictions in recovery.")
    expected_pairs = {(t, h) for t in config["universe"] for h in config["regression"]["horizons"]}
    if set(zip(predictions.ticker, predictions.horizon)) != expected_pairs:
        raise ValueError("Recovery does not cover the full configured universe/horizons.")
    metrics = []
    for (ticker, horizon), pair in predictions.groupby(["ticker", "horizon"], sort=False):
        for split in SPLITS:
            group = pair[pair.split == split]
            if set(group.model) != set(models):
                raise ValueError(f"Missing models for {ticker} h{horizon} {split}")
            reference = None
            for model in models:
                frame = group[group.model == model]
                grid = frame[
                    ["origin_date", "target_date", "current_price", "actual_price"]
                ].reset_index(drop=True)
                if reference is None:
                    reference = grid
                else:
                    pd.testing.assert_frame_equal(
                        reference, grid, check_exact=False, rtol=1e-10, atol=1e-10
                    )
                scoring = (
                    frame
                    if split == "test"
                    else frame.drop(columns=["lower_price", "upper_price"], errors="ignore")
                )
                metrics.append(
                    {
                        "ticker": ticker,
                        "horizon": horizon,
                        "model": model,
                        "split": split,
                        **score_frame(scoring),
                    }
                )
    metrics = pd.DataFrame(metrics)
    selections = []
    for ticker in config["universe"]:
        for horizon in config["regression"]["horizons"]:
            val = metrics[
                (metrics.ticker == ticker)
                & (metrics.horizon == horizon)
                & (metrics.split == "validation")
            ].set_index("model")
            winner = min(REQUIRED_MODELS, key=lambda m: val.loc[m, "mae"])
            selections.append(
                {
                    "ticker": ticker,
                    "horizon": horizon,
                    "selected_model": winner,
                    "validation_mae": val.loc[winner, "mae"],
                    "naive_validation_mae": val.loc["naive", "mae"],
                }
            )
    for name, frame in [
        ("predictions", predictions),
        ("metrics", metrics),
        ("cv_metrics", pd.concat(cvs, ignore_index=True)),
        ("selected_models", pd.DataFrame(selections)),
        ("selected_parameters", pd.DataFrame(settings)),
    ]:
        atomic_csv(frame, output / f"{name}.csv")
    columns = ["rmse", "mae", "mape_pct", "r2", "directional_accuracy_pct"]
    atomic_csv(
        metrics.groupby(["horizon", "model", "split"])[columns].mean().reset_index(),
        output / "macro_average_metrics.csv",
    )
    inputs[1].to_csv(output / "adjusted_prices.csv", index_label="Date")
    manifest = copy.deepcopy(source_manifest)
    manifest.update(
        run_name=output.name,
        status="completed",
        models=models,
        tickers=config["universe"],
        horizons=config["regression"]["horizons"],
        config=config,
        full_model_coverage=True,
        started_utc=read_json(output / "_recovery_state.json")["started_utc"],
        completed_utc=datetime.now(UTC).isoformat(),
        recovery_sources=origins,
        recovery_runner_sha256=expected["runner_sha256"],
        acceptance_note="Recovered compatible runs with matching scoring dates; historical test remains exposed. PRD empirical review remains required.",
    )
    manifest.pop("error", None)
    atomic_json(output / "manifest.json", manifest)
    atomic_json(
        output.parent / "latest.json",
        {"run_dir": str(output.relative_to(Path(config["project_root"])))},
    )


def fresh_run_name(root, prefix):
    attempt = 1
    while (root / f"{prefix}-{attempt:03d}").exists():
        attempt += 1
    return f"{prefix}-{attempt:03d}"


def run_recovery(args):
    import fcntl

    config = load_config(args.config)
    inputs = load_prepared(config)
    expected = identity(config, inputs[2])
    root = Path(config["project_root"])
    reports = root / "reports/prd"
    sources = [(root / p).resolve() for p in args.sources]
    manifests = [verify_source(p, expected) for p in sources]
    output = reports / args.run_name
    if not args.run_name.replace("-", "").replace("_", "").isalnum():
        raise ValueError("Use letters, digits, hyphens and underscores for the recovery run name.")
    state_path = output / "_recovery_state.json"
    if output.exists() and not state_path.exists():
        raise ValueError(
            f"{output} already exists and is not a recovery folder. Choose a new --run-name."
        )
    output.mkdir(parents=True, exist_ok=True)
    with (output / "_recovery.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("This recovery is already active in another terminal.") from None
        if state_path.exists():
            state = read_json(state_path)
            if state["identity"] != expected or state["sources"] != [str(p) for p in sources]:
                raise ValueError(
                    "Recovery settings, inputs, code or source list changed. Restore them before resuming."
                )
        else:
            state = {
                "identity": expected,
                "sources": [str(p) for p in sources],
                "started_utc": datetime.now(UTC).isoformat(),
            }
            atomic_json(state_path, state)
        if (output / "manifest.json").exists() and read_json(output / "manifest.json")[
            "status"
        ] == "completed":
            print(f"Already completed: {output}", flush=True)
            return
        progress_manifest = copy.deepcopy(manifests[0])
        progress_manifest.update(
            run_name=args.run_name,
            status="running",
            full_model_coverage=False,
            recovery_runner_sha256=expected["runner_sha256"],
        )
        progress_manifest.pop("completed_utc", None)
        atomic_json(output / "manifest.json", progress_manifest)
        for name in ("math_checks.json", "environment.lock.txt"):
            if (sources[0] / name).exists():
                shutil.copy2(sources[0] / name, output / name)
        jobs = [
            (t, h, m)
            for t in config["universe"]
            for h in config["regression"]["horizons"]
            for m in ["naive", *REQUIRED_MODELS]
        ]
        completed = []
        try:
            for number, (ticker, horizon, model) in enumerate(jobs, 1):
                key = f"{ticker}-h{horizon}-{model}"
                parent = output / "_checkpoints" / key
                ready = [
                    p for p in sorted(parent.glob("attempt-*")) if verify_checkpoint(p, expected)
                ]
                if ready:
                    completed.append(ready[-1])
                    print(f"[{number}/{len(jobs)}] Already saved: {key}", flush=True)
                    continue
                parent.mkdir(parents=True, exist_ok=True)
                dest = parent / fresh_run_name(parent, "attempt")
                source = next(
                    (p for p in sources if saved_job(p, ticker, horizon, model, config)), None
                )
                if source is None:
                    candidates = sorted(reports.glob(f"{args.run_name}-work-{key}-*"))
                    for candidate in reversed(candidates):
                        if not (candidate / "manifest.json").exists():
                            continue
                        verify_source(candidate, expected)
                        if saved_job(candidate, ticker, horizon, model, config):
                            source = candidate
                            break
                base = [
                    sys.executable,
                    "-u",
                    str(Path(__file__).resolve()),
                    "--config",
                    str(Path(args.config).resolve()),
                ]
                if source is None:
                    name = fresh_run_name(reports, f"{args.run_name}-work-{key}")
                    print(f"[{number}/{len(jobs)}] Training in a fresh process: {key}", flush=True)
                    command = base + [
                        "--worker",
                        "train",
                        "--ticker",
                        ticker,
                        "--horizon",
                        str(horizon),
                        "--model",
                        model,
                        "--child-name",
                        name,
                    ]
                    result = subprocess.run(command, cwd=root, pass_fds=(lock.fileno(),))
                    if result.returncode:
                        raise RuntimeError(
                            f"{key}: worker stopped (exit {result.returncode}). Saved checkpoints remain; rerun the same command to resume."
                        )
                    source = reports / name
                if (source / "predictions.csv").exists():
                    print(f"[{number}/{len(jobs)}] Checking saved predictions: {key}", flush=True)
                    checkpoint(source, dest, ticker, horizon, model, config, inputs, expected)
                else:
                    print(
                        f"[{number}/{len(jobs)}] Recovering saved model without fitting: {key}",
                        flush=True,
                    )
                    command = base + [
                        "--worker",
                        "restore",
                        "--ticker",
                        ticker,
                        "--horizon",
                        str(horizon),
                        "--model",
                        model,
                        "--source",
                        str(source),
                        "--destination",
                        str(dest),
                    ]
                    result = subprocess.run(command, cwd=root, pass_fds=(lock.fileno(),))
                    if result.returncode or not verify_checkpoint(dest, expected):
                        raise RuntimeError(
                            f"{key}: recovery verification stopped (exit {result.returncode}). Inspect the message above; no automatic retraining was performed."
                        )
                completed.append(dest)
            aggregate(completed, output, config, inputs, expected, manifests[0])
            print(
                f"\nCompleted comparison: {output}\nAll configured model families plus baseline, all stocks and horizons.",
                flush=True,
            )
        except BaseException as error:
            progress_manifest.update(status="interrupted", error=str(error))
            atomic_json(output / "manifest.json", progress_manifest)
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--sources", nargs="+", default=["reports/prd/prd-classical-2", "reports/prd/prd-deep-full"]
    )
    parser.add_argument("--run-name", default="prd-recovered-full")
    parser.add_argument("--worker", choices=["train", "restore"], help=argparse.SUPPRESS)
    parser.add_argument("--ticker", help=argparse.SUPPRESS)
    parser.add_argument("--horizon", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--model", help=argparse.SUPPRESS)
    parser.add_argument("--child-name", help=argparse.SUPPRESS)
    parser.add_argument("--source", help=argparse.SUPPRESS)
    parser.add_argument("--destination", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        if args.worker:
            config = load_config(args.config)
            if args.worker == "train":
                run_experiment(config, [args.model], [args.ticker], [args.horizon], args.child_name)
            else:
                inputs = load_prepared(config)
                checkpoint(
                    Path(args.source),
                    Path(args.destination),
                    args.ticker,
                    args.horizon,
                    args.model,
                    config,
                    inputs,
                    identity(config, inputs[2]),
                    restore=True,
                )
        else:
            run_recovery(args)
    except KeyboardInterrupt:
        print(
            "\nInterrupted. Rerun the same recovery command to continue from verified checkpoints.",
            file=sys.stderr,
        )
        return 130
    except Exception as error:
        print(f"Recovery stopped: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
