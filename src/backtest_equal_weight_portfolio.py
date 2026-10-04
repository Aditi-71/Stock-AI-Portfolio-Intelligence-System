from pathlib import Path

import numpy as np
import pandas as pd

STOCKS = [
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

BENCHMARK = "^GSPC"

TEST_START = "2024-11-22"
TEST_END = "2026-08-26"

TRADING_DAYS = 252

data_dir = Path("data/processed")

returns_file = data_dir / "portfolio_returns.parquet"

output_file = data_dir / "equal_weight_portfolio_test.parquet"


returns = pd.read_parquet(returns_file)

returns = returns.sort_index()


test = returns.loc[TEST_START:TEST_END].copy()


print("\nTest period:")
print(test.index.min(), "to", test.index.max())

print("\nTrading days:")
print(len(test))


number_of_stocks = len(STOCKS)

initial_weight = 1 / number_of_stocks


print("\nInitial weight per stock:")
print(f"{initial_weight:.2%}")


growth = (1 + test[STOCKS]).cumprod()


position_values = growth * initial_weight


portfolio_value = position_values.sum(axis=1)


portfolio_return = portfolio_value.pct_change()


portfolio_return.iloc[0] = test[STOCKS].iloc[0].mean()


benchmark_return = test[BENCHMARK]


benchmark_value = (1 + benchmark_return).cumprod()


def calculate_metrics(daily_returns):

    daily_returns = daily_returns.dropna()

    cumulative_return = (1 + daily_returns).prod() - 1

    annualized_return = (1 + cumulative_return) ** (TRADING_DAYS / len(daily_returns)) - 1

    annualized_volatility = daily_returns.std() * np.sqrt(TRADING_DAYS)

    if daily_returns.std() > 0:
        sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(TRADING_DAYS)

    else:
        sharpe = np.nan

    wealth = (1 + daily_returns).cumprod()

    wealth_with_start = pd.concat(
        [pd.Series([1.0], index=[wealth.index[0] - pd.Timedelta(days=1)]), wealth]
    )

    running_max = wealth_with_start.cummax()

    drawdown = wealth_with_start / running_max - 1

    max_drawdown = drawdown.min()

    return {
        "Cumulative Return": cumulative_return,
        "Annualized Return": annualized_return,
        "Annualized Volatility": annualized_volatility,
        "Sharpe Ratio": sharpe,
        "Max Drawdown": max_drawdown,
    }


portfolio_metrics = calculate_metrics(portfolio_return)

benchmark_metrics = calculate_metrics(benchmark_return)


print()
print("=" * 60)
print("EQUAL-WEIGHT PORTFOLIO")
print("=" * 60)

for metric, value in portfolio_metrics.items():
    print(f"{metric:25s}: {value:.4f}")


print()
print("=" * 60)
print("S&P 500 BENCHMARK")
print("=" * 60)

for metric, value in benchmark_metrics.items():
    print(f"{metric:25s}: {value:.4f}")


final_position_values = position_values.iloc[-1]

final_weights = final_position_values / final_position_values.sum()


print()
print("=" * 60)
print("FINAL PORTFOLIO WEIGHTS")
print("=" * 60)

print(final_weights.sort_values(ascending=False).round(4))


results = pd.DataFrame(index=test.index)

results["equal_weight_return"] = portfolio_return

results["equal_weight_value"] = portfolio_value

results["benchmark_return"] = benchmark_return

results["benchmark_value"] = benchmark_value


results.to_parquet(output_file)


print("\nSaved to:", output_file)
