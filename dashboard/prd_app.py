"""Seven PRD panels, reading precomputed experiments without training."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.prd.maths import risk_metrics

st.set_page_config(
    page_title="Stock forecasts and portfolio research", page_icon="📈", layout="wide"
)
st.title("Stock forecasts and portfolio research")
st.caption(
    "Educational research only. These forecasts and allocation examples are not financial advice."
)


@st.cache_data
def csv_file(path, modified):
    return pd.read_csv(path)


@st.cache_data
def parquet_file(path, modified):
    return pd.read_parquet(path)


def read_csv(path):
    return csv_file(str(path), path.stat().st_mtime_ns) if path.exists() else None


def read_parquet(path):
    return parquet_file(str(path), path.stat().st_mtime_ns) if path.exists() else None


def absent(label):
    st.info(f"{label} results are not available for this experiment yet.")


run_paths = sorted((ROOT / "reports" / "prd").glob("*/manifest.json"), reverse=True)
runs = [p.parent for p in run_paths if json.loads(p.read_text()).get("status") == "completed"]
if not runs:
    st.info("No completed regression experiment is available yet.")
    st.code(
        "python -m src.prd prepare\npython -m src.prd train --models naive --tickers AAPL --horizons 1",
        language="bash",
    )
    st.stop()

run = st.sidebar.selectbox("Experiment", runs, format_func=lambda p: p.name)
manifest = json.loads((run / "manifest.json").read_text())
ticker = st.sidebar.selectbox("Stock", manifest["tickers"])
horizon = st.sidebar.selectbox("Forecast horizon in trading sessions", manifest["horizons"])
rf = (
    st.sidebar.number_input(
        "Annual risk-free rate (%)",
        min_value=0.0,
        max_value=25.0,
        value=manifest["config"]["regression"]["annual_risk_free_rate"] * 100,
        step=0.25,
    )
    / 100
)
st.sidebar.caption(
    "The risk-free rate updates displayed risk ratios. Allocation charts use the saved experiment's weights."
)
if manifest["test_exposure"] == "historical_exposed":
    st.warning(
        "Historical evaluation: this test period was inspected during earlier project development. It is not an untouched final holdout."
    )
if not manifest["full_model_coverage"]:
    st.info("This experiment covers a subset of the required models, stocks or horizons.")
metrics = read_csv(run / "metrics.csv")
predictions = read_csv(run / "predictions.csv")
selected = read_csv(run / "selected_models.csv")
predictions["origin_date"] = pd.to_datetime(predictions.origin_date)
predictions["target_date"] = pd.to_datetime(predictions.target_date)
choice = selected[(selected.ticker == ticker) & (selected.horizon == horizon)].iloc[0]
tests = metrics[
    (metrics.ticker == ticker) & (metrics.horizon == horizon) & (metrics.split == "test")
]
overview, forecasting, comparison, portfolio, risk, sentiment, recommendations = st.tabs(
    [
        "Overview",
        "Price and prediction",
        "Model comparison",
        "Portfolio analytics",
        "Risk",
        "Sentiment",
        "Recommendations",
    ]
)

with overview:
    a, b, c, d = st.columns(4)
    a.metric("Stocks in this run", len(manifest["tickers"]))
    b.metric("Forecast horizons", ", ".join(map(str, manifest["horizons"])))
    c.metric("Evaluated models including baseline", len(manifest["models"]))
    d.metric("Selected model for this stock", choice.selected_model)
    st.write(
        "Models forecast adjusted returns, which are converted into adjusted-price forecasts. Development data selects each model; later observations measure error. The portfolio uses model views alongside historical returns and risk estimates."
    )
    st.dataframe(selected, hide_index=True, use_container_width=True)
    st.caption(
        "Selection uses validation MAE. A selected model may still fail to beat the random walk."
    )
    daily = read_parquet(ROOT / "data" / "processed" / "daily_sentiment.parquet")
    if daily is not None and not daily.empty:
        daily["session_date"] = pd.to_datetime(daily.session_date)
        mood_date = daily.session_date.max()
        recent = daily[daily.session_date == mood_date]
        mood = np.average(recent.sentiment_mean, weights=recent.article_count)
        label = "Bullish" if mood > 0.1 else "Bearish" if mood < -0.1 else "Neutral"
        st.metric(f"Observed market mood on {mood_date.date()}", label, f"{mood:.3f}")
        st.caption(
            f"Article-count-weighted sentiment across {recent.ticker.nunique()} stocks with observed news that day."
        )

with forecasting:
    model_names = list(tests.model)
    model = st.selectbox(
        "Forecasting model", model_names, index=model_names.index(choice.selected_model)
    )
    p = predictions[
        (predictions.ticker == ticker)
        & (predictions.horizon == horizon)
        & (predictions.model == model)
        & (predictions.split == "test")
    ].sort_values("target_date")
    fig = go.Figure()
    if "lower_price" in p and p.lower_price.notna().all():
        fig.add_trace(
            go.Scatter(x=p.target_date, y=p.lower_price, line={"width": 0}, showlegend=False)
        )
        fig.add_trace(
            go.Scatter(
                x=p.target_date,
                y=p.upper_price,
                fill="tonexty",
                line={"width": 0},
                name="Calibration-based prediction band",
                fillcolor="rgba(37,99,235,.14)",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=p.target_date,
            y=p.actual_price,
            name="Actual adjusted close",
            line={"color": "#0f172a"},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=p.target_date,
            y=p.predicted_price,
            name="Forecast adjusted close",
            line={"color": "#2563eb"},
        )
    )
    fig.update_layout(
        xaxis_title="Target trading date",
        yaxis_title="Adjusted price (USD)",
        template="plotly_white",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Historical test forecasts. Bands use reserved calibration residuals; nominal coverage is not guaranteed under changing market conditions."
    )
    st.dataframe(tests[tests.model == model], hide_index=True)
    st.dataframe(
        p[["origin_date", "target_date", "current_price", "predicted_price", "actual_price"]].tail(
            20
        ),
        hide_index=True,
    )

with comparison:
    cols = [
        "model",
        "n",
        "rmse",
        "mae",
        "mape_pct",
        "r2",
        "directional_accuracy_pct",
        "interval_coverage_pct",
    ]
    st.dataframe(tests[[c for c in cols if c in tests]], hide_index=True, use_container_width=True)
    st.plotly_chart(
        px.bar(
            tests.melt(
                id_vars="model",
                value_vars=["rmse", "mae"],
                var_name="Metric",
                value_name="Price error",
            ),
            x="model",
            y="Price error",
            color="Metric",
            barmode="group",
        ),
        use_container_width=True,
    )
    st.caption(
        "All models share scoring dates. Direction is measured from each origin's current price; the random walk predicts unchanged, not a randomly chosen up/down sign."
    )
    curves = list((run / "models" / f"{ticker}_h{horizon}").glob("*/training_loss.csv"))
    if curves:
        curve = st.selectbox("Training loss curve", curves, format_func=lambda p: p.parent.name)
        loss = read_csv(curve)
        st.plotly_chart(
            px.line(
                loss,
                x="epoch",
                y=["loss", "val_loss"],
                labels={"value": "Huber loss on scaled returns"},
            ),
            use_container_width=True,
        )

port = run / "portfolio"
weights = read_csv(port / "weights.csv")
portfolio_returns = read_csv(port / "returns.csv")
with portfolio:
    if weights is None:
        absent("Portfolio")
    else:
        meta = json.loads((port / "metadata.json").read_text())
        st.caption(
            f"Decision after close {meta['decision_close']}; entry at next open {meta['entry_open']}. {meta['weights_policy']}"
        )
        frontier, cloud = read_csv(port / "frontier.csv"), read_csv(port / "random_portfolios.csv")
        fig = px.scatter(
            cloud,
            x="volatility",
            y="expected_return",
            opacity=0.12,
            labels={
                "volatility": "Expected annual volatility",
                "expected_return": "Expected annual return",
            },
        )
        fig.add_trace(
            go.Scatter(
                x=frontier.volatility,
                y=frontier.expected_return,
                mode="lines",
                name="Efficient frontier",
                line={"color": "#f97316", "width": 3},
            )
        )
        st.plotly_chart(fig, use_container_width=True)
        strategy = st.selectbox("Allocation", list(weights.columns[1:]))
        st.plotly_chart(
            px.pie(weights, names="ticker", values=strategy, hole=0.45), use_container_width=True
        )
        st.dataframe(weights, hide_index=True, use_container_width=True)
        r = portfolio_returns.set_index("Date")
        st.plotly_chart(
            px.line((1 + r).cumprod(), labels={"value": "Growth of $1", "index": "Date"}),
            use_container_width=True,
        )
        st.caption(meta["expected_return_source"])

with risk:
    if portfolio_returns is None:
        absent("Risk")
    else:
        r = portfolio_returns.set_index("Date")
        rows = [{"strategy": c, **risk_metrics(r[c], r["Benchmark"], rf)} for c in r]
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        wealth = (1 + r).cumprod()
        peak = wealth.cummax().clip(lower=1)
        st.plotly_chart(
            px.line(wealth / peak - 1, labels={"value": "Drawdown"}), use_container_width=True
        )
        st.plotly_chart(
            px.line(
                r.rolling(20).std() * np.sqrt(252), labels={"value": "20-day annualised volatility"}
            ),
            use_container_width=True,
        )
        st.caption(
            "VaR and CVaR are historical one-day loss measures. Sharpe confidence intervals in the saved risk report use moving-block bootstrap."
        )
        adjusted = read_csv(run / "adjusted_prices.csv")
        if adjusted is not None:
            adjusted["Date"] = pd.to_datetime(adjusted.Date)
            adjusted = adjusted.set_index("Date")
            asset_returns = (
                adjusted.loc[pd.to_datetime(r.index)].pct_change(fill_method=None).dropna()
            )
            st.plotly_chart(
                px.imshow(
                    asset_returns.corr(),
                    zmin=-1,
                    zmax=1,
                    color_continuous_scale="RdBu_r",
                    title="Asset adjusted-close return correlations over this period",
                ),
                use_container_width=True,
            )

with sentiment:
    daily = read_parquet(ROOT / "data" / "processed" / "daily_sentiment.parquet")
    if daily is None:
        absent("News sentiment")
    else:
        rows = daily[daily.ticker == ticker].sort_values("session_date")
        st.plotly_chart(
            px.line(
                rows,
                x="session_date",
                y="sentiment_mean",
                title=f"Observed FinBERT sentiment for {ticker}",
            ),
            use_container_width=True,
        )
        st.plotly_chart(
            px.bar(rows, x="session_date", y="article_count", title="Observed article volume"),
            use_container_width=True,
        )
    ab = read_csv(run / "sentiment" / "comparison.csv")
    if ab is not None:
        st.dataframe(ab[(ab.ticker == ticker) & (ab.horizon == horizon)], hide_index=True)
        st.caption(
            "Negative MAE change favours sentiment; positive directional-accuracy change favours sentiment. Each result uses a matched control on the same news-covered dates."
        )
    else:
        absent("Regression sentiment A/B")
    status_path = run / "sentiment" / "status.json"
    if status_path.exists():
        with st.expander("News coverage and experiment status"):
            status = json.loads(status_path.read_text())
            st.dataframe(pd.DataFrame(status["experiments"]), hide_index=True)
            st.write(status["coverage"])

with recommendations:
    table = read_csv(port / "recommendations.csv")
    if table is None:
        absent("Recommendation")
    else:
        meta = json.loads((port / "recommendation_metadata.json").read_text())
        st.caption(f"Research signals at the initial allocation date. {meta['current_holdings']}")
        c1, c2, c3 = st.columns(3)
        a = c1.slider("Forecast contribution", 0.0, 1.0, float(meta["weights"][0]), 0.05)
        b = c2.slider("Sentiment contribution", 0.0, 1.0, float(meta["weights"][1]), 0.05)
        c = c3.slider("Risk discount", 0.0, 1.0, float(meta["weights"][2]), 0.05)
        total = a + b + c
        if total <= 0:
            st.info("Choose at least one nonzero signal weight.")
        else:
            table["composite_score"] = (
                a * table.forecast_subscore
                + b * table.sentiment_subscore.fillna(0)
                - c * table.risk_subscore
            ) / total
            low, high = meta["thresholds"]
            table["recommendation"] = np.where(
                table.composite_score >= high,
                "BUY",
                np.where(table.composite_score <= low, "SELL", "HOLD"),
            )
            st.dataframe(table, hide_index=True, use_container_width=True)
            st.caption(
                "Slider changes are illustrative and do not rerun the saved backtest. Missing sentiment is shown explicitly and contributes zero."
            )
        hits = read_csv(port / "recommendation_hit_rates.csv")
        if hits is not None:
            st.dataframe(hits, hide_index=True)
            st.caption(meta["hit_rate_definition"])

st.caption("Educational research only · Historical results do not establish future performance.")
