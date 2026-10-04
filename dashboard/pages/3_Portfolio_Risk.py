from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="Portfolio & Risk",
    page_icon="💼",
    layout="wide",
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed"


@st.cache_data
def load_min_variance():

    path = DATA_DIR / "min_variance_portfolio_test.parquet"

    return pd.read_parquet(path)


@st.cache_data
def load_max_sharpe():

    path = DATA_DIR / "max_sharpe_portfolio_test.parquet"

    return pd.read_parquet(path)


@st.cache_data
def load_risk_report():

    path = DATA_DIR / "portfolio_risk_report.csv"

    return pd.read_csv(path)


@st.cache_data
def load_weights(filename):

    path = DATA_DIR / filename

    df = pd.read_csv(path, index_col=0)

    df.columns = ["Weight"]

    return df


min_var = load_min_variance()
max_sharpe = load_max_sharpe()
risk = load_risk_report()

min_weights = load_weights("min_variance_weights.csv")

max_weights = load_weights("max_sharpe_weights.csv")


st.title("💼 Portfolio Optimization & Risk")

st.write(
    """
    Three portfolio strategies were evaluated out-of-sample
    from **2024-11-22 to 2026-08-26**:

    - Equal Weight
    - Minimum Variance
    - Maximum Sharpe

    All optimized weights were estimated using historical data
    available before the test period.
    """
)


min_row = risk[risk["Strategy"] == "Minimum Variance"].iloc[0]


col1, col2, col3, col4 = st.columns(4)


with col1:
    st.metric("Best Test Portfolio", "Minimum Variance")


with col2:
    st.metric("Cumulative Return", f"{min_row['Cumulative Return']:.2%}")


with col3:
    st.metric("Sharpe Ratio", f"{min_row['Sharpe Ratio']:.2f}")


with col4:
    st.metric("Max Drawdown", f"{min_row['Max Drawdown']:.2%}")


st.divider()

st.subheader("Cumulative Growth of $1")


comparison = pd.DataFrame(index=min_var.index)


comparison["Minimum Variance"] = (1 + min_var["min_variance_return"]).cumprod()


comparison["Equal Weight"] = (1 + min_var["equal_weight_return"]).cumprod()


comparison["Maximum Sharpe"] = (1 + max_sharpe["max_sharpe_return"]).cumprod()


comparison["S&P 500"] = (1 + min_var["benchmark_return"]).cumprod()


wealth_long = comparison.reset_index().melt(
    id_vars=comparison.index.name or "index", var_name="Strategy", value_name="Portfolio Value"
)


date_column = comparison.index.name if comparison.index.name else "index"


fig_wealth = px.line(
    wealth_long,
    x=date_column,
    y="Portfolio Value",
    color="Strategy",
    title=("Out-of-Sample Portfolio Growth"),
)


fig_wealth.update_layout(
    xaxis_title="Date",
    yaxis_title="Value of $1 Investment",
)


st.plotly_chart(fig_wealth, use_container_width=True)


st.subheader("Performance Comparison")


display = risk.copy()


percentage_columns = [
    "Cumulative Return",
    "Annualized Return",
    "Annualized Volatility",
    "Max Drawdown",
    "VaR 95%",
    "CVaR 95%",
    "Worst Day",
    "Best Day",
]


for column in percentage_columns:
    display[column] = display[column] * 100


st.dataframe(
    display.style.format(
        {
            "Cumulative Return": "{:.2f}%",
            "Annualized Return": "{:.2f}%",
            "Annualized Volatility": "{:.2f}%",
            "Sharpe Ratio": "{:.2f}",
            "Sortino Ratio": "{:.2f}",
            "Max Drawdown": "{:.2f}%",
            "VaR 95%": "{:.2f}%",
            "CVaR 95%": "{:.2f}%",
            "Beta": "{:.2f}",
            "Worst Day": "{:.2f}%",
            "Best Day": "{:.2f}%",
        }
    ),
    hide_index=True,
    use_container_width=True,
)


st.divider()

st.subheader("Risk-Adjusted Performance")


risk_long = risk[
    [
        "Strategy",
        "Sharpe Ratio",
        "Sortino Ratio",
    ]
].melt(
    id_vars="Strategy",
    var_name="Metric",
    value_name="Ratio",
)


fig_ratios = px.bar(
    risk_long,
    x="Strategy",
    y="Ratio",
    color="Metric",
    barmode="group",
    text_auto=".2f",
    title="Sharpe and Sortino Ratios",
)


st.plotly_chart(fig_ratios, use_container_width=True)


st.subheader("Drawdown Comparison")


drawdowns = pd.DataFrame(index=comparison.index)


for strategy in comparison.columns:
    wealth = comparison[strategy]

    drawdowns[strategy] = wealth / wealth.cummax() - 1


drawdown_long = drawdowns.reset_index().melt(
    id_vars=drawdowns.index.name or "index",
    var_name="Strategy",
    value_name="Drawdown",
)


drawdown_date_column = drawdowns.index.name if drawdowns.index.name else "index"


fig_drawdown = px.line(
    drawdown_long,
    x=drawdown_date_column,
    y="Drawdown",
    color="Strategy",
    title="Portfolio Drawdowns",
)


fig_drawdown.update_layout(
    xaxis_title="Date",
    yaxis_title="Drawdown",
    yaxis_tickformat=".0%",
)


st.plotly_chart(fig_drawdown, use_container_width=True)


st.divider()

st.subheader("Optimized Portfolio Weights")


weight_option = st.radio(
    "Select optimized portfolio",
    [
        "Minimum Variance",
        "Maximum Sharpe",
    ],
    horizontal=True,
)


if weight_option == "Minimum Variance":
    selected_weights = min_weights.copy()

else:
    selected_weights = max_weights.copy()


selected_weights = selected_weights.reset_index()


selected_weights.columns = [
    "Ticker",
    "Weight",
]


selected_weights = selected_weights.sort_values("Weight", ascending=False)


fig_weights = px.bar(
    selected_weights,
    x="Ticker",
    y="Weight",
    text_auto=".1%",
    title=f"{weight_option} Initial Allocation",
)


fig_weights.update_layout(
    yaxis_tickformat=".0%",
    yaxis_title="Portfolio Weight",
)


st.plotly_chart(fig_weights, use_container_width=True)


st.divider()

st.subheader("Tail Risk")


tail = risk[
    [
        "Strategy",
        "VaR 95%",
        "CVaR 95%",
    ]
].copy()


tail_long = tail.melt(
    id_vars="Strategy",
    var_name="Metric",
    value_name="Daily Loss",
)


tail_long["Daily Loss"] = tail_long["Daily Loss"] * 100


fig_tail = px.bar(
    tail_long,
    x="Strategy",
    y="Daily Loss",
    color="Metric",
    barmode="group",
    text_auto=".2f",
    title=("95% Value at Risk and Expected Shortfall"),
)


fig_tail.update_layout(yaxis_title="Daily Loss (%)")


st.plotly_chart(fig_tail, use_container_width=True)


st.subheader("Market Beta")


fig_beta = px.bar(
    risk,
    x="Strategy",
    y="Beta",
    text_auto=".2f",
    title="Beta Relative to S&P 500",
)


fig_beta.add_hline(
    y=1,
    line_dash="dash",
    annotation_text="Market beta = 1",
)


st.plotly_chart(fig_beta, use_container_width=True)


st.divider()

st.subheader("Interpretation")


st.success(
    """
    **Minimum Variance produced the strongest downside-risk result.**

    It achieved:

    - 42.26% cumulative test-period return
    - 1.54 Sharpe ratio
    - 2.35 Sortino ratio
    - −9.14% maximum drawdown
    - 1.69% 95% CVaR
    """
)


st.info(
    """
    **Equal Weight was remarkably competitive.**

    Despite requiring no covariance or return estimation, it achieved
    a 41.18% cumulative return and 1.53 Sharpe ratio.

    This illustrates the robustness of simple diversification.
    """
)


st.warning(
    """
    **Maximum-Sharpe optimization did not generalize well.**

    The optimizer heavily favored stocks with high historical expected
    returns—particularly NVDA.

    Although its estimated training Sharpe ratio was superior, its
    out-of-sample Sharpe fell to approximately 0.99.

    This demonstrates the instability of historical expected-return
    estimates and the risk of optimization overfitting.
    """
)


st.divider()

st.subheader("Portfolio Methodology")


st.markdown(
    """
    ### Equal Weight

    Each of the 10 stocks received an initial allocation of 10%.
    Positions were then allowed to drift naturally without daily
    rebalancing.

    ### Minimum Variance

    Portfolio weights were selected by minimizing estimated portfolio
    variance using a **Ledoit-Wolf shrinkage covariance matrix**.

    Constraints:

    - Long-only
    - Fully invested
    - No leverage

    ### Maximum Sharpe

    Portfolio weights were selected by maximizing historical
    return relative to estimated volatility.

    Constraints:

    - Long-only
    - Fully invested
    - Maximum 30% allocation per stock

    ### Out-of-Sample Evaluation

    Optimization used data only through **2024-11-21**.

    The resulting weights were locked before evaluating performance
    from **2024-11-22 through 2026-08-26**.
    """
)


st.caption(
    """
    Portfolio results are historical research results and are not
    investment recommendations. Past performance does not guarantee
    future returns.
    """
)
