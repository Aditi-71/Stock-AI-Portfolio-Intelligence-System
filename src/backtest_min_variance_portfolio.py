from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

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

output_file = data_dir / "min_variance_portfolio_test.parquet"


returns = pd.read_parquet(returns_file)

returns = returns.sort_index()


train = returns.loc[returns.index < TEST_START].copy()


test = returns.loc[TEST_START:TEST_END].copy()


print("\nTraining period:")

print(train.index.min(), "to", train.index.max())


print("Training days:", len(train))


print("\nTest period:")

print(test.index.min(), "to", test.index.max())


print("Test days:", len(test))


lw = LedoitWolf()

lw.fit(train[STOCKS])


covariance_matrix = lw.covariance_ * TRADING_DAYS


def portfolio_variance(weights, covariance):

    return weights.T @ covariance @ weights


number_of_assets = len(STOCKS)


initial_weights = np.repeat(1 / number_of_assets, number_of_assets)


constraints = {"type": "eq", "fun": lambda weights: np.sum(weights) - 1}


bounds = tuple((0.0, 1.0) for _ in range(number_of_assets))


result = minimize(
    portfolio_variance,
    initial_weights,
    args=(covariance_matrix,),
    method="SLSQP",
    bounds=bounds,
    constraints=constraints,
    options={
        "ftol": 1e-12,
        "maxiter": 1000,
        "disp": False,
    },
)

if not result.success:
    raise RuntimeError("Optimization failed: " + result.message)


weights = result.x


equal_variance = initial_weights.T @ covariance_matrix @ initial_weights

optimized_variance = weights.T @ covariance_matrix @ weights


print("\nOptimization iterations:")
print(result.nit)

print("\nEqual-weight expected volatility:")
print(np.sqrt(equal_variance))

print("\nOptimized expected volatility:")
print(np.sqrt(optimized_variance))

print("\nVariance reduction:")
print(1 - optimized_variance / equal_variance)

weights_series = pd.Series(weights, index=STOCKS, name="weight")


print()
print("=" * 60)

print("MINIMUM-VARIANCE WEIGHTS")

print("=" * 60)

print(weights_series.sort_values(ascending=False).round(4))


print("\nWeight sum:", weights.sum())


growth = (1 + test[STOCKS]).cumprod()


position_values = growth * weights_series


portfolio_value = position_values.sum(axis=1)


portfolio_return = portfolio_value.pct_change()


portfolio_return.iloc[0] = test[STOCKS].iloc[0] @ weights_series


equal_weights = pd.Series(1 / number_of_assets, index=STOCKS)


equal_growth = (1 + test[STOCKS]).cumprod()


equal_position_values = equal_growth * equal_weights


equal_value = equal_position_values.sum(axis=1)


equal_return = equal_value.pct_change()


equal_return.iloc[0] = test[STOCKS].iloc[0].mean()


benchmark_return = test[BENCHMARK]


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

    running_max = pd.concat(
        [pd.Series([1.0], index=[wealth.index[0] - pd.Timedelta(days=1)]), wealth]
    ).cummax()

    wealth_with_start = pd.concat(
        [pd.Series([1.0], index=[wealth.index[0] - pd.Timedelta(days=1)]), wealth]
    )

    drawdown = wealth_with_start / running_max - 1

    max_drawdown = drawdown.min()

    return {
        "Cumulative Return": cumulative_return,
        "Annualized Return": annualized_return,
        "Annualized Volatility": annualized_volatility,
        "Sharpe Ratio": sharpe,
        "Max Drawdown": max_drawdown,
    }


min_var_metrics = calculate_metrics(portfolio_return)

equal_metrics = calculate_metrics(equal_return)

benchmark_metrics = calculate_metrics(benchmark_return)


for name, metrics in [
    ("MINIMUM VARIANCE", min_var_metrics),
    ("EQUAL WEIGHT", equal_metrics),
    ("S&P 500", benchmark_metrics),
]:
    print()
    print("=" * 60)
    print(name)
    print("=" * 60)

    for metric, value in metrics.items():
        print(f"{metric:25s}: {value:.4f}")


final_positions = position_values.iloc[-1]


final_weights = final_positions / final_positions.sum()


print()
print("=" * 60)

print("FINAL DRIFTED MIN-VARIANCE WEIGHTS")

print("=" * 60)


print(final_weights.sort_values(ascending=False).round(4))


output = pd.DataFrame(index=test.index)


output["min_variance_return"] = portfolio_return


output["min_variance_value"] = portfolio_value


output["equal_weight_return"] = equal_return


output["equal_weight_value"] = equal_value


output["benchmark_return"] = benchmark_return


output.to_parquet(output_file)

weights_series.to_csv(data_dir / "min_variance_weights.csv")

print("\nSaved to:", output_file)
