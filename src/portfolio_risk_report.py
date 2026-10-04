from pathlib import Path

import numpy as np
import pandas as pd

TRADING_DAYS = 252

data_dir = Path("data/processed")


min_var = pd.read_parquet(data_dir / "min_variance_portfolio_test.parquet")

max_sharpe = pd.read_parquet(data_dir / "max_sharpe_portfolio_test.parquet")


returns = pd.DataFrame(index=min_var.index)

returns["Minimum Variance"] = min_var["min_variance_return"]

returns["Equal Weight"] = min_var["equal_weight_return"]

returns["Maximum Sharpe"] = max_sharpe["max_sharpe_return"]

returns["S&P 500"] = min_var["benchmark_return"]


returns = returns.dropna()


def sortino_ratio(series):

    downside_returns = np.minimum(series, 0)

    downside_deviation = np.sqrt(np.mean(downside_returns**2)) * np.sqrt(TRADING_DAYS)

    annual_return = series.mean() * TRADING_DAYS

    if downside_deviation == 0:
        return np.nan

    return annual_return / downside_deviation


def historical_var(series, confidence=0.95):

    return -np.quantile(series, 1 - confidence)


def historical_cvar(series, confidence=0.95):

    cutoff = np.quantile(series, 1 - confidence)

    tail = series[series <= cutoff]

    return -tail.mean()


def beta(portfolio, market):

    covariance = np.cov(portfolio, market, ddof=1)[0, 1]

    market_variance = np.var(market, ddof=1)

    return covariance / market_variance


rows = []

market = returns["S&P 500"]


for strategy in returns.columns:
    series = returns[strategy]

    cumulative_return = (1 + series).prod() - 1

    annualized_return = (1 + cumulative_return) ** (TRADING_DAYS / len(series)) - 1

    annualized_volatility = series.std() * np.sqrt(TRADING_DAYS)

    sharpe = series.mean() / series.std() * np.sqrt(TRADING_DAYS)

    wealth = (1 + series).cumprod()

    wealth_with_start = pd.concat(
        [pd.Series([1.0], index=[wealth.index[0] - pd.Timedelta(days=1)]), wealth]
    )

    drawdown = wealth_with_start / wealth_with_start.cummax() - 1

    rows.append(
        {
            "Strategy": strategy,
            "Cumulative Return": cumulative_return,
            "Annualized Return": annualized_return,
            "Annualized Volatility": annualized_volatility,
            "Sharpe Ratio": sharpe,
            "Sortino Ratio": sortino_ratio(series),
            "Max Drawdown": drawdown.min(),
            "VaR 95%": historical_var(series),
            "CVaR 95%": historical_cvar(series),
            "Beta": (1.0 if strategy == "S&P 500" else beta(series, market)),
            "Worst Day": series.min(),
            "Best Day": series.max(),
        }
    )


report = pd.DataFrame(rows)


pd.set_option("display.max_columns", None)

pd.set_option("display.width", 200)


print()
print("=" * 100)
print("PORTFOLIO RISK REPORT")
print("=" * 100)

print(report.set_index("Strategy").round(4))


output_file = data_dir / "portfolio_risk_report.csv"

report.to_csv(output_file, index=False)


print("\nSaved to:", output_file)
