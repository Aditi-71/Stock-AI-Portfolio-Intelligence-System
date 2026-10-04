from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="Stock AI System",
    page_icon="📈",
    layout="wide",
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "processed"


@st.cache_data
def load_portfolio_returns():

    path = DATA_DIR / "portfolio_returns.parquet"

    if path.exists():
        return pd.read_parquet(path)

    return None


@st.cache_data
def load_risk_report():

    path = DATA_DIR / "portfolio_risk_report.csv"

    if path.exists():
        return pd.read_csv(path)

    return None


@st.cache_data
def load_sentiment_comparison():

    path = DATA_DIR / "sentiment_rf_comparison.csv"

    if path.exists():
        return pd.read_csv(path)

    return None


portfolio_returns = load_portfolio_returns()
risk_report = load_risk_report()
sentiment_results = load_sentiment_comparison()


st.title("📈 Stock AI & Portfolio Intelligence System")

st.markdown(
    """
    An end-to-end quantitative finance project combining:

    - Machine learning for next-day stock direction prediction
    - Technical and macroeconomic feature engineering
    - FinBERT financial-news sentiment analysis
    - Walk-forward validation
    - Trading-strategy backtesting
    - Portfolio optimization
    - Risk analytics
    """
)

st.divider()


st.subheader("Project Overview")


if portfolio_returns is not None:
    start_date = portfolio_returns.index.min().date()
    end_date = portfolio_returns.index.max().date()

else:
    start_date = "2015"
    end_date = "2026"


col1, col2, col3, col4 = st.columns(4)


with col1:
    st.metric(label="Stock Universe", value="10 Stocks")


with col2:
    st.metric(label="Data Period", value=f"{start_date} → {end_date}")


with col3:
    st.metric(label="Final ML Model", value="Random Forest")


with col4:
    st.metric(label="Walk-Forward ROC-AUC", value="0.5266")


st.caption(
    "The predictive signal is modest; results should not be interpreted "
    "as evidence of a strong or guaranteed forecasting edge."
)


st.subheader("Stock Universe")


stocks = [
    "AAPL",
    "MSFT",
    "NVDA",
    "JPM",
    "XOM",
    "JNJ",
    "PG",
    "KO",
    "CAT",
    "HD",
]


st.write(" • ".join(stocks))

st.write("**Benchmark:** S&P 500 (^GSPC)")


st.divider()

st.subheader("Machine Learning Summary")


ml_results = pd.DataFrame(
    {
        "Model": [
            "Logistic Regression",
            "Random Forest",
            "Gradient Boosting",
            "XGBoost",
            "GRU",
            "LSTM",
        ],
        "Walk-Forward ROC-AUC": [
            0.5123,
            0.5266,
            0.5159,
            0.5220,
            0.5157,
            0.4985,
        ],
    }
)


fig_ml = px.bar(
    ml_results,
    x="Model",
    y="Walk-Forward ROC-AUC",
    title="Walk-Forward ROC-AUC by Model",
    text_auto=".4f",
)


fig_ml.add_hline(y=0.5, line_dash="dash", annotation_text="Random classifier")


fig_ml.update_layout(
    yaxis_title="ROC-AUC",
    xaxis_title="Model",
)


st.plotly_chart(fig_ml, use_container_width=True)


st.info(
    """
    **Random Forest was selected as the final forecasting model.**

    Its mean walk-forward ROC-AUC was approximately **0.5266**, which was
    the strongest and most stable result among the tested models.

    The result indicates only a weak predictive signal, so the project
    emphasizes realistic validation rather than claiming high forecasting
    accuracy.
    """
)


st.divider()

st.subheader("FinBERT Sentiment Experiment")


if sentiment_results is not None:
    sentiment_summary = (
        sentiment_results.groupby("model")
        .agg(
            Accuracy=("accuracy", "mean"),
            F1=("f1", "mean"),
            ROC_AUC=("roc_auc", "mean"),
        )
        .reset_index()
    )

    sentiment_summary["Model"] = sentiment_summary["model"].replace(
        {
            "RF WITHOUT SENTIMENT": "RF without sentiment",
            "RF WITH SENTIMENT": "RF with FinBERT sentiment",
        }
    )

    fig_sentiment = px.bar(
        sentiment_summary,
        x="Model",
        y="ROC_AUC",
        title="Effect of FinBERT Sentiment",
        text_auto=".4f",
    )

    fig_sentiment.update_layout(yaxis_title="Mean Walk-Forward ROC-AUC", xaxis_title="")

    st.plotly_chart(fig_sentiment, use_container_width=True)


st.write(
    """
    Adding AAPL FinBERT sentiment features **did not improve forecasting
    performance**.

    - Without sentiment ROC-AUC: **0.5261**
    - With sentiment ROC-AUC: **0.5181**
    - Difference: **−0.0080**

    Therefore sentiment was retained as an NLP research component but
    excluded from the final Random Forest forecasting model.
    """
)


st.divider()

st.subheader("Portfolio Optimization Summary")


if risk_report is not None:
    display_columns = [
        "Strategy",
        "Cumulative Return",
        "Annualized Volatility",
        "Sharpe Ratio",
        "Sortino Ratio",
        "Max Drawdown",
    ]

    portfolio_table = risk_report[display_columns].copy()

    percentage_columns = [
        "Cumulative Return",
        "Annualized Volatility",
        "Max Drawdown",
    ]

    for column in percentage_columns:
        portfolio_table[column] = portfolio_table[column] * 100

    st.dataframe(
        portfolio_table.style.format(
            {
                "Cumulative Return": "{:.2f}%",
                "Annualized Volatility": "{:.2f}%",
                "Sharpe Ratio": "{:.2f}",
                "Sortino Ratio": "{:.2f}",
                "Max Drawdown": "{:.2f}%",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    fig_portfolio = px.bar(
        risk_report,
        x="Strategy",
        y="Sharpe Ratio",
        title="Out-of-Sample Sharpe Ratio",
        text_auto=".2f",
    )

    st.plotly_chart(fig_portfolio, use_container_width=True)


st.success(
    """
    **Best downside-risk result: Minimum-Variance Portfolio**

    Test-period results:

    - Cumulative return: **42.26%**
    - Annualized return: **22.37%**
    - Sharpe ratio: **1.54**
    - Sortino ratio: **2.35**
    - Maximum drawdown: **−9.14%**
    """
)


st.divider()

st.subheader("Methodology")


st.markdown(
    """
    **1. Data ingestion**
    - Daily OHLCV equity data
    - S&P 500 benchmark
    - VIX
    - Treasury yields, CPI and unemployment

    **2. Feature engineering**
    - Returns and momentum
    - Moving averages
    - RSI, MACD and Bollinger Bands
    - Volatility and ATR
    - Volume features
    - Market and macroeconomic variables

    **3. Machine learning**
    - Logistic Regression
    - Random Forest
    - Gradient Boosting
    - XGBoost
    - LSTM
    - GRU

    **4. NLP**
    - Historical financial news
    - Trading-session alignment
    - FinBERT sentiment classification
    - Daily sentiment aggregation
    - Walk-forward A/B testing

    **5. Portfolio optimization**
    - Equal-weight portfolio
    - Minimum-variance portfolio
    - Maximum-Sharpe portfolio
    - Out-of-sample evaluation

    **6. Risk analytics**
    - Sharpe ratio
    - Sortino ratio
    - Maximum drawdown
    - VaR
    - CVaR / Expected Shortfall
    - Market beta
    """
)


st.divider()

st.caption(
    """
    This project is for research and educational purposes only.
    Historical performance does not guarantee future results and this
    dashboard does not constitute financial advice.
    """
)
