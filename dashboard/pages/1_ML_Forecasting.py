from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="ML Forecasting",
    page_icon="🤖",
    layout="wide",
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "processed"


@st.cache_data
def load_predictions():

    path = DATA_DIR / "final_rf_test_predictions.parquet"

    if path.exists():
        df = pd.read_parquet(path)

        return df.sort_index()

    return None


@st.cache_data
def load_sentiment_comparison():

    path = DATA_DIR / "sentiment_rf_comparison.csv"

    if path.exists():
        return pd.read_csv(path)

    return None


predictions = load_predictions()

sentiment_results = load_sentiment_comparison()


st.title("🤖 Machine Learning Forecasting")

st.write(
    """
    This section evaluates machine-learning models for predicting
    whether AAPL's next adjusted-close return will be positive or
    negative.
    """
)


st.warning(
    """
    The final models show only a modest predictive signal.
    Results should therefore be interpreted as an ML research
    experiment rather than reliable market forecasts.
    """
)


st.subheader("Final Random Forest")

col1, col2, col3, col4 = st.columns(4)


with col1:
    st.metric("Walk-Forward ROC-AUC", "0.5266")


with col2:
    st.metric("Test ROC-AUC", "0.4900")


with col3:
    st.metric("Test Accuracy", "51.14%")


with col4:
    st.metric("Test F1", "0.5996")


st.caption(
    """
    Model selection was based primarily on walk-forward performance,
    not the single fixed test split.
    """
)


st.divider()

st.subheader("Walk-Forward Model Comparison")


model_results = pd.DataFrame(
    {
        "Model": [
            "Logistic Regression",
            "Random Forest",
            "Gradient Boosting",
            "XGBoost",
            "GRU",
            "LSTM",
        ],
        "ROC-AUC": [
            0.5123,
            0.5266,
            0.5159,
            0.5220,
            0.5157,
            0.4985,
        ],
    }
)


fig_models = px.bar(
    model_results,
    x="Model",
    y="ROC-AUC",
    text_auto=".4f",
    title=("Mean Walk-Forward ROC-AUC"),
)


fig_models.add_hline(
    y=0.50,
    line_dash="dash",
    annotation_text=("Random classifier"),
)


fig_models.update_layout(
    yaxis_title="ROC-AUC",
    xaxis_title="",
    yaxis_range=[0.47, 0.54],
)


st.plotly_chart(fig_models, use_container_width=True)


st.subheader("Random Forest Walk-Forward Performance")


rf_walk_forward = pd.DataFrame(
    {
        "Metric": [
            "Accuracy",
            "F1",
            "ROC-AUC",
        ],
        "Mean": [
            0.5299,
            0.6022,
            0.5266,
        ],
        "Std Dev": [
            0.0251,
            0.0445,
            0.0120,
        ],
    }
)


st.dataframe(
    rf_walk_forward.style.format(
        {
            "Mean": "{:.4f}",
            "Std Dev": "{:.4f}",
        }
    ),
    hide_index=True,
    use_container_width=True,
)


st.info(
    """
    The Random Forest had the highest average walk-forward ROC-AUC
    and one of the lowest fold-to-fold ROC-AUC variations.

    However, mean accuracy of 52.99% was close to the approximately
    53.31% always-up baseline.
    """
)


if predictions is not None:
    st.divider()

    st.subheader("Final Test Predictions")

    st.write(
        f"""
        Test observations: **{len(predictions):,}**

        Period: **{predictions.index.min().date()}**
        to **{predictions.index.max().date()}**
        """
    )

    if "probability_up" in predictions.columns:
        fig_probability = px.histogram(
            predictions,
            x="probability_up",
            nbins=30,
            title=("Distribution of Predicted Probability of an Up Day"),
        )

        fig_probability.add_vline(
            x=0.5,
            line_dash="dash",
            annotation_text=("Decision threshold"),
        )

        fig_probability.update_layout(
            xaxis_title=("Probability of next-day positive return"),
            yaxis_title="Count",
        )

        st.plotly_chart(fig_probability, use_container_width=True)

    if {"actual", "prediction"}.issubset(predictions.columns):
        actual = predictions["actual"].astype(int)

        predicted = predictions["prediction"].astype(int)

        tn = ((actual == 0) & (predicted == 0)).sum()

        fp = ((actual == 0) & (predicted == 1)).sum()

        fn = ((actual == 1) & (predicted == 0)).sum()

        tp = ((actual == 1) & (predicted == 1)).sum()

        confusion = [
            [tn, fp],
            [fn, tp],
        ]

        fig_confusion = go.Figure(
            data=go.Heatmap(
                z=confusion,
                x=[
                    "Predicted Down",
                    "Predicted Up",
                ],
                y=[
                    "Actual Down",
                    "Actual Up",
                ],
                text=confusion,
                texttemplate="%{text}",
                hovertemplate=("%{y}<br>%{x}<br>Count: %{z}<extra></extra>"),
            )
        )

        fig_confusion.update_layout(
            title=("Random Forest Test Confusion Matrix"),
            xaxis_title="Prediction",
            yaxis_title="Actual",
        )

        st.plotly_chart(fig_confusion, use_container_width=True)


st.divider()

st.subheader("Does News Sentiment Improve the Model?")


if sentiment_results is not None:
    summary = (
        sentiment_results.groupby("model")
        .agg(
            {
                "accuracy": "mean",
                "f1": "mean",
                "roc_auc": "mean",
            }
        )
        .reset_index()
    )

    summary["Model"] = summary["model"].replace(
        {
            "RF WITHOUT SENTIMENT": "Without sentiment",
            "RF WITH SENTIMENT": "With FinBERT sentiment",
        }
    )

    fig_sentiment = px.bar(
        summary,
        x="Model",
        y="roc_auc",
        text_auto=".4f",
        title=("FinBERT Sentiment A/B Test"),
    )

    fig_sentiment.update_layout(
        xaxis_title="",
        yaxis_title=("Mean Walk-Forward ROC-AUC"),
        yaxis_range=[0.49, 0.54],
    )

    st.plotly_chart(fig_sentiment, use_container_width=True)


st.write(
    """
    **Result**

    - Without sentiment: **0.5261 ROC-AUC**
    - With FinBERT sentiment: **0.5181 ROC-AUC**
    - Change: **−0.0080**

    The sentiment features therefore were not included in the
    final forecasting model.
    """
)


st.divider()

st.subheader("Interpretation")


st.markdown(
    """
    **What worked**

    - Random Forest was the strongest classical/deep-learning model
      in walk-forward validation.
    - Tree-based models generally performed better than the recurrent
      neural networks.
    - Walk-forward validation prevented conclusions from relying on a
      single favorable train/test split.

    **What did not work**

    - LSTM performance was approximately random.
    - FinBERT sentiment did not improve the Random Forest.
    - Direction accuracy remained close to a naïve always-up strategy.

    **Main conclusion**

    The project found a small but unstable predictive signal. The
    forecasting component is therefore treated as an experimental
    signal generator rather than a reliable prediction engine.
    """
)
