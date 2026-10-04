"""Training-only exploratory statistics and captioned HTML figures."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .config import prepared_dir, write_json
from .data import load_prepared, split_bounds
from .experiment import resolve_run, verify_analysis_inputs


def run_eda(config, run_dir=None):
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from statsmodels.tsa.seasonal import STL
    from statsmodels.tsa.stattools import adfuller

    output = resolve_run(config, run_dir)
    features, adjusted, data_manifest = load_prepared(config)
    verify_analysis_inputs(config, output, data_manifest)
    cutoff = features.index[split_bounds(len(features), config)["train"][1] - 1]
    prices = adjusted.loc[:cutoff]
    returns = prices.pct_change(fill_method=None).dropna()
    clean = pd.read_parquet(prepared_dir(config) / "clean_prices.parquet").loc[prices.index]
    dest = output / "eda"
    dest.mkdir(exist_ok=True)
    rows, captions = [], []

    def save(figure, filename, caption):
        figure.update_layout(template="plotly_white", title=caption)
        figure.write_html(dest / filename, include_plotlyjs="directory")
        captions.append({"file": filename, "finding_and_consequence": caption})

    for ticker in config["universe"]:
        p = prices[ticker]
        r = returns[ticker]
        log_r = np.log(p).diff().dropna()
        price_adf, return_adf = (
            float(adfuller(p, autolag="AIC")[1]),
            float(adfuller(log_r, autolag="AIC")[1]),
        )
        jb = stats.jarque_bera(r)
        df, loc, scale = stats.t.fit(r)
        rows.append(
            {
                "ticker": ticker,
                "price_adf_p": price_adf,
                "return_adf_p": return_adf,
                "skew": float(stats.skew(r)),
                "excess_kurtosis": float(stats.kurtosis(r)),
                "jarque_bera_p": float(jb.pvalue),
                "t_df": float(df),
                "parametric_t_var95_loss": float(-stats.t.ppf(0.05, df, loc=loc, scale=scale)),
            }
        )
        chart = make_subplots(
            rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25], vertical_spacing=0.04
        )
        for name, series in [
            ("Adjusted close", p),
            ("SMA50", p.rolling(50).mean()),
            ("SMA200", p.rolling(200).mean()),
        ]:
            chart.add_trace(go.Scatter(x=p.index, y=series, name=name), row=1, col=1)
        chart.add_trace(go.Bar(x=p.index, y=clean[("Volume", ticker)], name="Volume"), row=2, col=1)
        chart.update_yaxes(type="log", row=1, col=1)
        save(
            chart,
            f"{ticker}_trend.html",
            f"{ticker}: ADF p(price)={price_adf:.3g}, p(log return)={return_adf:.3g}; assess the fixed return target against this evidence.",
        )
        histogram = go.Figure(go.Histogram(x=r, histnorm="probability density", name="Returns"))
        axis = np.linspace(float(r.min()), float(r.max()), 300)
        histogram.add_trace(
            go.Scatter(x=axis, y=stats.norm.pdf(axis, r.mean(), r.std()), name="Normal reference")
        )
        save(
            histogram,
            f"{ticker}_distribution.html",
            f"{ticker}: excess kurtosis={stats.kurtosis(r):.2f}; compare empirical tail risk with normality assumptions.",
        )
        theoretical, ordered = stats.probplot(r, fit=False)
        save(
            px.scatter(
                x=theoretical,
                y=ordered,
                labels={"x": "Normal theoretical quantile", "y": "Observed return"},
            ),
            f"{ticker}_qq.html",
            f"{ticker}: Jarque–Bera p={jb.pvalue:.3g}; tail deviations motivate empirical VaR and block-bootstrap intervals.",
        )
        seasonal = r.groupby([r.index.year, r.index.month]).mean().unstack()
        save(
            px.imshow(seasonal * 100, aspect="auto", labels={"color": "Mean daily return (%)"}),
            f"{ticker}_seasonality.html",
            f"{ticker}: year-by-month means show whether apparent seasonality persists across years; do not select features from test-season patterns.",
        )
        pd.DataFrame(
            {"month": r.index.month, "day_of_week": r.index.dayofweek, "return": r.to_numpy()}
        ).groupby("day_of_week")["return"].mean().to_csv(dest / f"{ticker}_weekday.csv")
        if len(p) >= 504:
            decomposition = STL(np.log(p), period=252, robust=True).fit()
            pd.DataFrame(
                {
                    "log_price": np.log(p),
                    "trend": decomposition.trend,
                    "seasonal": decomposition.seasonal,
                    "residual": decomposition.resid,
                },
                index=p.index,
            ).to_csv(dest / f"{ticker}_stl.csv", index_label="Date")
    save(
        px.imshow(
            returns[config["universe"]].corr(), zmin=-1, zmax=1, color_continuous_scale="RdBu_r"
        ),
        "asset_correlation.html",
        "Training return correlations guide diversification; sector-related holdings may move together.",
    )
    for method in ("pearson", "spearman"):
        returns.corr(method=method).to_csv(dest / f"asset_{method}.csv")
        features.loc[:cutoff].corr(method=method).to_csv(dest / f"feature_{method}.csv")
    chart = make_subplots(specs=[[{"secondary_y": True}]])
    chart.add_trace(
        go.Scatter(
            x=returns.index,
            y=returns[config["benchmark"]].rolling(20).std() * np.sqrt(252),
            name="20-day annualised market volatility",
        ),
        secondary_y=False,
    )
    chart.add_trace(
        go.Scatter(x=prices.index, y=clean[("Close", "^VIX")], name="VIX"), secondary_y=True
    )
    save(
        chart,
        "volatility_and_vix.html",
        "Volatility clustering and VIX co-movement motivate regime features and time-aware validation.",
    )
    pd.DataFrame(rows).to_csv(dest / "statistics.csv", index=False)
    write_json(dest / "figures.json", captions)
    write_json(
        dest / "scope.json",
        {
            "analysis_end": cutoff,
            "data_scope": "training only",
            "stl_period_sessions": 252,
            "parametric_var": "Student-t fit to training simple returns; empirical VaR remains reported separately.",
            "adf_caveat": "ADF rejection concerns a unit-root null under the chosen specification; it is not proof that a financial return process is stationary in every sense.",
        },
    )
    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}
        },
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Training data exploration\n",
                    "This notebook displays the generated training-only EDA findings and figures. Run the EDA command first.\n",
                ],
            },
            {
                "cell_type": "code",
                "metadata": {},
                "execution_count": None,
                "outputs": [],
                "source": [
                    "from pathlib import Path\n",
                    "import json, pandas as pd\n",
                    "from IPython.display import display, Markdown, IFrame\n",
                    "display(pd.read_csv('statistics.csv'))\n",
                    "for item in json.loads(Path('figures.json').read_text()):\n",
                    "    display(Markdown(item['finding_and_consequence']))\n",
                    "    display(IFrame(item['file'], width='100%', height=520))\n",
                ],
            },
        ],
    }
    write_json(dest / "EDA.ipynb", notebook)
    return dest
