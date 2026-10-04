"""Generate an evidence-backed results draft, with explicit pending work."""

from __future__ import annotations

import json

import pandas as pd

from .experiment import resolve_run


def markdown_table(frame):
    def cell(value):
        if isinstance(value, float):
            return f"{value:.4f}" if pd.notna(value) else "unavailable"
        return str(value).replace("|", "\\|").replace("\n", " ")

    return "\n".join(
        [
            "| " + " | ".join(map(str, frame.columns)) + " |",
            "| " + " | ".join(["---"] * len(frame.columns)) + " |",
        ]
        + [
            "| " + " | ".join(cell(v) for v in row) + " |"
            for row in frame.itertuples(index=False, name=None)
        ]
    )


def write_report(config, run_dir=None):
    output = resolve_run(config, run_dir)
    manifest = json.loads((output / "manifest.json").read_text())
    metrics = pd.read_csv(output / "metrics.csv")
    selected = pd.read_csv(output / "selected_models.csv")
    test = metrics[metrics.split == "test"]
    checks = json.loads((output / "math_checks.json").read_text())
    parts = [
        "# Stock forecasting and portfolio research results",
        "",
        f"Run: {manifest['run_name']}. This draft records outputs produced by this experiment. The final 10–15 page report still requires review, model-specific interpretation, and presentation editing.",
        "## Data and evaluation scope",
        f"Assets: {', '.join(manifest['tickers'])}. Horizons: {manifest['horizons']} trading sessions. Evaluation status: **{manifest['test_exposure']}**.",
        manifest["data"].get(
            "macro_vintage_status", "Macro data provenance was not supplied for this run."
        ),
        "## Methodology",
        "The model predicts forward adjusted log returns. Price forecasts are reconstructed from the adjusted price at the forecast origin. RMSE, MAE, MAPE and R² use reconstructed prices; direction compares the forecast and actual change from that origin. Splits are chronological. Labels reaching a later block are purged, and all models use the same scoring origins and historical window eligibility.",
        manifest["fit_policy"],
        manifest["selection_rule"],
        "## Financial mathematics checks",
        f"Gradient descent matches the library prediction tolerance: {checks['gradient_descent']['passed']}. Maximum covariance difference from NumPy: {checks['covariance_max_difference_from_numpy']:.3g}.",
        "## Development model selection",
        markdown_table(selected),
        "## Held-out or historical test metrics",
        markdown_table(
            test[
                [
                    "ticker",
                    "horizon",
                    "model",
                    "n",
                    "rmse",
                    "mae",
                    "mape_pct",
                    "r2",
                    "directional_accuracy_pct",
                ]
            ]
        ),
        "## Baseline comparison",
    ]
    baseline = test[test.model == "naive"][
        ["ticker", "horizon", "mae", "directional_accuracy_pct"]
    ].rename(
        columns={"mae": "naive_mae", "directional_accuracy_pct": "naive_directional_accuracy_pct"}
    )
    chosen = test.merge(selected[["ticker", "horizon", "selected_model"]], on=["ticker", "horizon"])
    chosen = chosen[chosen.model == chosen.selected_model].merge(baseline, on=["ticker", "horizon"])
    chosen["beats_naive_mae"] = chosen.mae < chosen.naive_mae
    chosen["beats_naive_direction"] = (
        chosen.directional_accuracy_pct > chosen.naive_directional_accuracy_pct
    )
    parts.append(
        markdown_table(
            chosen[["ticker", "horizon", "model", "beats_naive_mae", "beats_naive_direction"]]
        )
    )
    parts += [
        "The random walk predicts no price change. Its directional accuracy is the unchanged-price fraction under strict sign matching; it is not assigned a theoretical 50%. Beating that directional score alone is weak evidence of useful direction prediction.",
        "## Prediction uncertainty",
        manifest["interval_method"],
        "## EDA and model diagnosis",
        "Inspect eda/statistics.csv and the captioned figures. Inspect every deep model's training_loss.csv for overfitting. The lowest validation MAE defines the selected model, but does not by itself establish why an architecture won. Discuss sample size, representation, regularisation and regime sensitivity using the recorded curves and fold metrics.",
    ]
    for title, file in [
        ("Sentiment integration", "sentiment/comparison.csv"),
        ("Portfolio performance", "portfolio/risk_metrics.csv"),
        ("Recommendations and rebalancing", "portfolio/recommendations.csv"),
    ]:
        path = output / file
        parts += [
            f"## {title}",
            markdown_table(pd.read_csv(path))
            if path.exists()
            else "Pending: the corresponding analysis has not produced a result for this run.",
        ]
    parts += [
        "## Limitations and remaining acceptance checks",
        "Current-vintage macro data, a fixed surviving large-cap universe, limited or incomplete news coverage, previously inspected test periods and simplified execution can all affect the interpretation. Retain negative results. An implemented model is not evidence of predictive value, and a code run is not evidence that every PRD acceptance criterion passed.",
        "The recommendation module's current evaluation status is recorded separately. Confirm the full eight-model universe/horizon coverage, news experiments, frontier cross-check, report review and genuinely unseen evaluation before declaring the PRD finished.",
        "## Reproducibility",
        "The run manifest contains configuration, source/data hashes and package versions. Saved models, preprocessing, fold metrics, prediction rows and loss curves support reproduction. See RUN_PRD.md for the commands.",
        "Educational research only; not financial advice.",
    ]
    path = output / "RESULTS_DRAFT.md"
    path.write_text("\n\n".join(parts) + "\n")
    return path
