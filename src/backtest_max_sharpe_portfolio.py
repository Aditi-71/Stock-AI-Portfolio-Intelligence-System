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

RISK_FREE_RATE = 0.0

MAX_WEIGHT = 0.30


data_dir = Path("data/processed")

returns_file = data_dir / "portfolio_returns.parquet"

output_file = data_dir / "max_sharpe_portfolio_test.parquet"


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


expected_returns = train[STOCKS].mean() * TRADING_DAYS


print("\nAnnualized training returns:")

print(expected_returns.sort_values(ascending=False).round(4))


lw = LedoitWolf()

lw.fit(train[STOCKS])

covariance_matrix = lw.covariance_ * TRADING_DAYS


def portfolio_return(weights):

    return weights @ expected_returns.values


def portfolio_volatility(weights):

    variance = weights.T @ covariance_matrix @ weights

    return np.sqrt(variance)


def negative_sharpe(weights):

    expected_return = portfolio_return(weights)

    volatility = portfolio_volatility(weights)

    if volatility == 0:
        return 1e6

    sharpe = (expected_return - RISK_FREE_RATE) / volatility

    return -sharpe


number_of_assets = len(STOCKS)

initial_weights = np.repeat(1 / number_of_assets, number_of_assets)


constraints = {
    "type": "eq",
    "fun": lambda weights: np.sum(weights) - 1,
}


bounds = tuple((0.0, MAX_WEIGHT) for _ in range(number_of_assets))


result = minimize(
    negative_sharpe,
    initial_weights,
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


weights_series = pd.Series(weights, index=STOCKS, name="weight")


optimized_return = portfolio_return(weights)

optimized_volatility = portfolio_volatility(weights)

optimized_sharpe = (optimized_return - RISK_FREE_RATE) / optimized_volatility


equal_return_train = portfolio_return(initial_weights)

equal_vol_train = portfolio_volatility(initial_weights)

equal_sharpe_train = (equal_return_train - RISK_FREE_RATE) / equal_vol_train


print("\nOptimization iterations:")
print(result.nit)


print()
print("=" * 60)
print("MAXIMUM-SHARPE INITIAL WEIGHTS")
print("=" * 60)

print(weights_series.sort_values(ascending=False).round(4))


print("\nWeight sum:", weights.sum())


print()
print("TRAINING EXPECTED PERFORMANCE")

print(f"Max-Sharpe return:     {optimized_return:.4f}")

print(f"Max-Sharpe volatility: {optimized_volatility:.4f}")

print(f"Max-Sharpe ratio:      {optimized_sharpe:.4f}")


print()

print(f"Equal-weight return:     {equal_return_train:.4f}")

print(f"Equal-weight volatility: {equal_vol_train:.4f}")

print(f"Equal-weight Sharpe:      {equal_sharpe_train:.4f}")


growth = (1 + test[STOCKS]).cumprod()


position_values = growth * weights_series


max_sharpe_value = position_values.sum(axis=1)


max_sharpe_return = max_sharpe_value.pct_change()


max_sharpe_return.iloc[0] = test[STOCKS].iloc[0] @ weights_series


equal_weights = pd.Series(1 / number_of_assets, index=STOCKS)


equal_positions = growth * equal_weights


equal_value = equal_positions.sum(axis=1)


equal_return = equal_value.pct_change()


equal_return.iloc[0] = test[STOCKS].iloc[0].mean()


benchmark_return = test[BENCHMARK]


def calculate_metrics(daily_returns):

    daily_returns = daily_returns.dropna()

    cumulative_return = (1 + daily_returns).prod() - 1

    annualized_return = (1 + cumulative_return) ** (TRADING_DAYS / len(daily_returns)) - 1

    volatility = daily_returns.std() * np.sqrt(TRADING_DAYS)

    if daily_returns.std() > 0:
        sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(TRADING_DAYS)

    else:
        sharpe = np.nan

    wealth = (1 + daily_returns).cumprod()

    start = pd.Series([1.0], index=[wealth.index[0] - pd.Timedelta(days=1)])

    wealth_with_start = pd.concat([start, wealth])

    running_max = wealth_with_start.cummax()

    drawdown = wealth_with_start / running_max - 1

    max_drawdown = drawdown.min()

    return {
        "Cumulative Return": cumulative_return,
        "Annualized Return": annualized_return,
        "Annualized Volatility": volatility,
        "Sharpe Ratio": sharpe,
        "Max Drawdown": max_drawdown,
    }


max_sharpe_metrics = calculate_metrics(max_sharpe_return)

equal_metrics = calculate_metrics(equal_return)

benchmark_metrics = calculate_metrics(benchmark_return)


for name, metrics in [
    ("MAXIMUM SHARPE", max_sharpe_metrics),
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

print("FINAL DRIFTED MAX-SHARPE WEIGHTS")

print("=" * 60)


print(final_weights.sort_values(ascending=False).round(4))


output = pd.DataFrame(index=test.index)


output["max_sharpe_return"] = max_sharpe_return


output["max_sharpe_value"] = max_sharpe_value


output["equal_weight_return"] = equal_return


output["equal_weight_value"] = equal_value


output["benchmark_return"] = benchmark_return


output.to_parquet(output_file)


weights_series.to_csv(data_dir / "max_sharpe_weights.csv")


print("\nSaved to:", output_file)
