from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="News Sentiment",
    page_icon="📰",
    layout="wide",
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed"


@st.cache_data
def load_daily_sentiment():

    path = DATA_DIR / "daily_sentiment.parquet"

    if path.exists():
        return pd.read_parquet(path)

    return None


@st.cache_data
def load_news_sentiment():

    path = DATA_DIR / "news_sentiment.parquet"

    if path.exists():
        return pd.read_parquet(path)

    return None


@st.cache_data
def load_sentiment_comparison():

    path = DATA_DIR / "sentiment_rf_comparison.csv"

    if path.exists():
        return pd.read_csv(path)

    return None


daily = load_daily_sentiment()
articles = load_news_sentiment()
comparison = load_sentiment_comparison()


st.title("📰 Financial News Sentiment")

st.write(
    """
    Historical financial news was processed using **FinBERT**, a
    transformer model designed for financial-language sentiment analysis.

    Article-level predictions were aligned to trading sessions and
    aggregated into daily sentiment features for the forecasting model.
    """
)


if articles is not None:
    aapl_articles = articles[articles["ticker"] == "AAPL"].copy()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("AAPL Articles", f"{len(aapl_articles):,}")

    with col2:
        st.metric("Start Date", str(aapl_articles["time_published"].min().date()))

    with col3:
        st.metric("End Date", str(aapl_articles["time_published"].max().date()))

    with col4:
        st.metric("Sentiment Model", "FinBERT")


st.divider()

st.subheader("Article Sentiment Distribution")


if articles is not None:
    aapl_articles = articles[articles["ticker"] == "AAPL"].copy()

    label_counts = aapl_articles["sentiment_label"].value_counts().reset_index()

    label_counts.columns = ["Sentiment", "Articles"]

    fig_labels = px.bar(
        label_counts,
        x="Sentiment",
        y="Articles",
        text_auto=True,
        title="FinBERT Sentiment Labels",
    )

    st.plotly_chart(fig_labels, use_container_width=True)


st.subheader("Sentiment Score Distribution")


if articles is not None:
    fig_scores = px.histogram(
        aapl_articles,
        x="sentiment_score",
        nbins=50,
        title=("Distribution of FinBERT Sentiment Scores"),
    )

    fig_scores.add_vline(
        x=0,
        line_dash="dash",
        annotation_text="Neutral boundary",
    )

    fig_scores.update_layout(
        xaxis_title=("Sentiment Score (positive probability − negative probability)"),
        yaxis_title="Article count",
    )

    st.plotly_chart(fig_scores, use_container_width=True)


st.divider()

st.subheader("Daily AAPL Sentiment")


if daily is not None:
    aapl_daily = daily[daily["ticker"] == "AAPL"].copy().sort_values("session_date")

    aapl_daily["sentiment_20d"] = (
        aapl_daily["sentiment_mean"].rolling(window=20, min_periods=1).mean()
    )

    fig_daily = px.line(
        aapl_daily,
        x="session_date",
        y="sentiment_20d",
        title=("20-Day Rolling Average of AAPL News Sentiment"),
    )

    fig_daily.add_hline(y=0, line_dash="dash")

    fig_daily.update_layout(
        xaxis_title="Date",
        yaxis_title="Rolling sentiment score",
    )

    st.plotly_chart(fig_daily, use_container_width=True)


st.subheader("Financial News Volume")


if daily is not None:
    monthly_news = (
        aapl_daily.set_index("session_date")["article_count"].resample("ME").sum().reset_index()
    )

    fig_volume = px.line(
        monthly_news,
        x="session_date",
        y="article_count",
        title=("Monthly AAPL News Article Count"),
    )

    fig_volume.update_layout(
        xaxis_title="Date",
        yaxis_title="Articles",
    )

    st.plotly_chart(fig_volume, use_container_width=True)


st.divider()

st.subheader("News Coverage")


if daily is not None:
    news_days = len(aapl_daily)

    total_articles = int(aapl_daily["article_count"].sum())

    average_articles = aapl_daily["article_count"].mean()

    max_articles = aapl_daily["article_count"].max()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Trading Days With News", f"{news_days:,}")

    with col2:
        st.metric("Total Aggregated Articles", f"{total_articles:,}")

    with col3:
        st.metric("Avg Articles / News Day", f"{average_articles:.2f}")

    with col4:
        st.metric("Max Articles in One Day", f"{int(max_articles):,}")


st.divider()

st.subheader("Does Sentiment Improve Forecasting?")


if comparison is not None:
    summary = (
        comparison.groupby("model")
        .agg(
            Accuracy=("accuracy", "mean"),
            F1=("f1", "mean"),
            ROC_AUC=("roc_auc", "mean"),
        )
        .reset_index()
    )

    summary["Model"] = summary["model"].replace(
        {
            "RF WITHOUT SENTIMENT": "Without sentiment",
            "RF WITH SENTIMENT": "With FinBERT sentiment",
        }
    )

    fig_compare = px.bar(
        summary,
        x="Model",
        y="ROC_AUC",
        text_auto=".4f",
        title=("Random Forest Walk-Forward ROC-AUC With vs Without Sentiment"),
    )

    fig_compare.update_layout(
        xaxis_title="",
        yaxis_title="Mean ROC-AUC",
        yaxis_range=[0.49, 0.54],
    )

    st.plotly_chart(fig_compare, use_container_width=True)


st.warning(
    """
    Adding FinBERT sentiment did **not** improve the forecasting model.

    - RF without sentiment: **0.5261 ROC-AUC**
    - RF with sentiment: **0.5181 ROC-AUC**
    - Change: **−0.0080**

    The NLP pipeline was therefore retained as an experimental research
    component, but sentiment features were excluded from the final
    Random Forest model.
    """
)


st.divider()

st.subheader("Sentiment Pipeline")


st.markdown(
    """
    **1. Historical news ingestion**

    News was downloaded for the stock universe using ticker-filtered
    financial-news data.

    **2. Trading-session alignment**

    Articles were mapped to appropriate NYSE trading sessions.
    News after market close was assigned to the next trading session
    to avoid look-ahead leakage.

    **3. FinBERT inference**

    Each article's title and summary were classified as:

    - Positive
    - Negative
    - Neutral

    A continuous sentiment score was calculated as:

    **positive probability − negative probability**

    **4. Daily aggregation**

    Article-level sentiment was aggregated into features including:

    - Mean sentiment
    - Sentiment standard deviation
    - Positive ratio
    - Negative ratio
    - Neutral ratio
    - Article count
    - Sentiment balance

    **5. Walk-forward A/B test**

    The Random Forest was evaluated on identical time periods with
    and without sentiment features.
    """
)


st.divider()

st.subheader("Interpretation")


st.write(
    """
    The sentiment experiment demonstrates an important machine-learning
    result: additional data does not automatically improve a model.

    FinBERT successfully transformed unstructured financial news into
    numerical features, but those features did not provide a consistent
    incremental signal for next-day AAPL direction prediction.

    This negative result was retained rather than discarded because it
    provides evidence that the modeling process was evaluated
    empirically rather than selectively reporting only successful
    experiments.
    """
)
