"""Paired regression A/B experiments using existing session-aligned news."""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pandas as pd

from .config import DEEP_MODELS, project_path, write_json
from .data import blocks_for, load_prepared
from .experiment import predict, resolve_run, verify_analysis_inputs
from .metrics import predictions_frame, score_frame
from .models_ml import build_classical


def sentiment_features(daily, coverage, dates, ticker, score_column="sentiment_mean"):
    day = daily[daily.ticker == ticker].copy()
    day["session_date"] = pd.to_datetime(day.session_date).dt.normalize()
    day = (
        day.groupby("session_date")
        .agg(sentiment=(score_column, "mean"), article_count=("article_count", "sum"))
        .reindex(dates)
    )
    known = day.article_count.notna()
    if coverage is not None and "complete" in coverage:
        cov = coverage[coverage.ticker == ticker].copy()
        cov["session_date"] = pd.to_datetime(cov.session_date).dt.normalize()
        complete = cov.groupby("session_date").complete.all().reindex(dates, fill_value=False)
        known |= complete
    day.loc[known, ["sentiment", "article_count"]] = day.loc[
        known, ["sentiment", "article_count"]
    ].fillna(0)
    result = pd.DataFrame(index=dates)
    result["news_sentiment"] = day.sentiment
    result["news_momentum_3d"] = day.sentiment.diff(3)
    result["news_log_count"] = np.log1p(day.article_count)
    average = day.article_count.shift(1).rolling(20, min_periods=5).mean()
    result["news_volume_spike"] = (
        (day.article_count > 2 * average).astype(float).where(known & average.notna())
    )
    result["news_has_articles"] = (day.article_count > 0).astype(float).where(known)
    return result


def filter_paired_blocks(base_blocks, augmented_blocks):
    result_base, result_augmented = {}, {}
    for split, aug in augmented_blocks.items():
        valid_rows = aug.features.notna().all(axis=1).astype(int)
        valid_window = valid_rows.rolling(aug.length).sum().eq(aug.length).to_numpy()
        origins = aug.origins[valid_window[aug.origins]]
        if not len(origins):
            raise ValueError(f"No complete {split} windows with observed/verified news coverage.")
        result_augmented[split] = replace(aug, origins=origins)
        result_base[split] = replace(base_blocks[split], origins=origins)
    return result_base, result_augmented


def score_vader(config):
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    path = project_path(config, "data", "processed", "clean_news.parquet")
    if not path.exists():
        raise FileNotFoundError("Need the existing clean_news.parquet for the VADER comparison.")
    news = pd.read_parquet(path)
    if not {"session_date", "ticker", "text"}.issubset(news.columns):
        raise ValueError("clean_news.parquet must contain session_date, ticker and text.")
    text = (
        news.text.fillna("")
        .astype(str)
        .str.lower()
        .str.replace(r"https?://\S+|\$[a-z]+\b", "", regex=True)
    )
    text = text.str.replace(r"\beps\b", "earnings per share", regex=True).str.replace(
        r"\bqoq\b", "quarter over quarter", regex=True
    )
    scorer = SentimentIntensityAnalyzer()
    news["vader"] = text.map(lambda value: scorer.polarity_scores(value)["compound"])
    result = (
        news.groupby(["session_date", "ticker"])
        .agg(sentiment_mean=("vader", "mean"), article_count=("vader", "size"))
        .reset_index()
    )
    dest = project_path(config, "data", "processed", "daily_sentiment_vader.parquet")
    result.to_parquet(dest, index=False)
    return dest


def run_sentiment(config, run_dir=None, progress=print):
    output = resolve_run(config, run_dir)
    features, prices, data_manifest = load_prepared(config)
    verify_analysis_inputs(config, output, data_manifest)
    selections = pd.read_csv(output / "selected_models.csv")
    settings = pd.read_csv(output / "selected_parameters.csv")
    coverage_path = project_path(config, "data", "processed", "news_coverage.parquet")
    coverage = pd.read_parquet(coverage_path) if coverage_path.exists() else None
    metrics, comparisons, statuses = [], [], []
    dest = output / "sentiment"
    dest.mkdir(exist_ok=True)
    for scorer, filename in [
        ("finbert", "daily_sentiment.parquet"),
        ("vader", "daily_sentiment_vader.parquet"),
    ]:
        path = project_path(config, "data", "processed", filename)
        if not path.exists():
            statuses.append(
                {"scorer": scorer, "status": "pending", "reason": f"Missing {filename}"}
            )
            continue
        daily = pd.read_parquet(path)
        for selection in selections.itertuples(index=False):
            ticker, horizon, model = (
                selection.ticker,
                int(selection.horizon),
                selection.selected_model,
            )
            if model == "naive":
                statuses.append(
                    {
                        "ticker": ticker,
                        "horizon": horizon,
                        "scorer": scorer,
                        "status": "pending",
                        "reason": "Train the required regressors first.",
                    }
                )
                continue
            news_features = sentiment_features(daily, coverage, features.index, ticker)
            base = blocks_for(features, prices, ticker, horizon, config)
            augmented = blocks_for(features.join(news_features), prices, ticker, horizon, config)
            try:
                paired = filter_paired_blocks(base, augmented)
            except ValueError as error:
                statuses.append(
                    {
                        "ticker": ticker,
                        "horizon": horizon,
                        "scorer": scorer,
                        "status": "pending",
                        "reason": str(error),
                    }
                )
                continue
            params = json.loads(
                settings[
                    (settings.ticker == ticker)
                    & (settings.horizon == horizon)
                    & (settings.model == model)
                ]
                .iloc[0]
                .params
            )
            pair_results = {}
            for label, blocks in zip(("without_sentiment", f"with_{scorer}"), paired):
                progress(f"Sentiment A/B: {ticker} h={horizon} {model} {label}")
                if model in DEEP_MODELS:
                    from .models_dl import fit_deep

                    estimator = fit_deep(model, blocks["train"], blocks["validation"], config)
                else:
                    estimator = build_classical(model, params, config).fit(
                        blocks["train"].X, blocks["train"].y
                    )
                for split in ("validation", "test"):
                    frame = predictions_frame(
                        blocks[split],
                        predict(estimator, blocks[split]),
                        ticker,
                        horizon,
                        label,
                        split,
                    )
                    row = {
                        "ticker": ticker,
                        "horizon": horizon,
                        "selected_architecture": model,
                        "scorer": scorer,
                        "variant": label,
                        "split": split,
                        **score_frame(frame),
                    }
                    metrics.append(row)
                    if split == "test":
                        pair_results[label] = row
                    frame.to_csv(
                        dest / f"{ticker}_h{horizon}_{scorer}_{label}_{split}.csv", index=False
                    )
            a, b = pair_results["without_sentiment"], pair_results[f"with_{scorer}"]
            comparisons.append(
                {
                    "ticker": ticker,
                    "horizon": horizon,
                    "scorer": scorer,
                    "model": model,
                    "n_test": a["n"],
                    "delta_mae_with_minus_without": b["mae"] - a["mae"],
                    "delta_directional_accuracy_points": b["directional_accuracy_pct"]
                    - a["directional_accuracy_pct"],
                }
            )
            statuses.append(
                {"ticker": ticker, "horizon": horizon, "scorer": scorer, "status": "completed"}
            )
    if metrics:
        pd.DataFrame(metrics).to_csv(dest / "metrics.csv", index=False)
        pd.DataFrame(comparisons).to_csv(dest / "comparison.csv", index=False)
    write_json(
        dest / "status.json",
        {
            "experiments": statuses,
            "pairing": "Base and augmented models use the same eligible training, validation and test origins; gaps remain on the original trading-session grid.",
            "hyperparameters": "Architecture and classical parameters fixed from the base development experiment; no test-driven retuning.",
            "coverage": "Observed-news rows are retained. Zero-news fills require an explicit complete coverage flag. Legacy downloaded flags do not certify API completeness.",
            "limitation": "An experiment restricted to observed news windows is conditional on that coverage, not representative of uncollected history. FinBERT/VADER experiments can have different eligible samples; compare each against its own paired control.",
        },
    )
    return dest
