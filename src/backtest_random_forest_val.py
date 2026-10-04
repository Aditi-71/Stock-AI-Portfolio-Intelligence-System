from pathlib import Path

import numpy as np
import pandas as pd

data_dir = Path("data/processed")

TRADING_DAYS = 252

TRANSACTION_COST = 0.001


predictions = pd.read_parquet(data_dir / "random_forest_val_predictions.parquet")


prices = pd.read_parquet(data_dir / "clean_prices.parquet")

aapl = prices[("Adj Close", "AAPL")].copy()


next_day_return = aapl.shift(-1) / aapl - 1

next_day_return.name = "next_day_return"


backtest = predictions.join(next_day_return, how="left")

backtest = backtest.dropna(subset=["next_day_return"])


print("\nBacktest rows:")
print(len(backtest))

print("\nDate range:")
print(backtest.index.min(), "to", backtest.index.max())


def calculate_metrics(returns, positions, transaction_count):

    equity_curve = (1 + returns).cumprod()

    cumulative_return = equity_curve.iloc[-1] - 1

    years = len(returns) / TRADING_DAYS

    annualized_return = equity_curve.iloc[-1] ** (1 / years) - 1

    annualized_volatility = returns.std() * np.sqrt(TRADING_DAYS)

    if returns.std() == 0:
        sharpe = np.nan

    else:
        sharpe = returns.mean() / returns.std() * np.sqrt(TRADING_DAYS)

    running_max = equity_curve.cummax()

    drawdown = equity_curve / running_max - 1

    max_drawdown = drawdown.min()

    exposure = positions.mean()

    return {
        "cumulative_return": cumulative_return,
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_volatility,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "transactions": transaction_count,
        "exposure": exposure,
    }


results = []


thresholds = np.arange(0.50, 0.66, 0.01)


for threshold in thresholds:
    position = (backtest["probability_up"] >= threshold).astype(int)

    gross_return = position * backtest["next_day_return"]

    position_change = position.diff().abs().fillna(position.iloc[0])

    trading_cost = position_change * TRANSACTION_COST

    net_return = gross_return - trading_cost

    transaction_count = int(position_change.sum())

    metrics = calculate_metrics(net_return, position, transaction_count)

    metrics["threshold"] = threshold

    results.append(metrics)


results_df = pd.DataFrame(results)


best_row = results_df.sort_values("sharpe", ascending=False).iloc[0]


print("\n===== BEST VALIDATION THRESHOLD =====")

print(f"Threshold: {best_row['threshold']:.2f}")

print(f"Cumulative Return: {best_row['cumulative_return']:.4f}")

print(f"Annualized Return: {best_row['annualized_return']:.4f}")

print(f"Annualized Volatility: {best_row['annualized_volatility']:.4f}")

print(f"Sharpe Ratio: {best_row['sharpe']:.4f}")

print(f"Maximum Drawdown: {best_row['max_drawdown']:.4f}")

print(f"Transactions: {int(best_row['transactions'])}")

print(f"Market Exposure: {best_row['exposure']:.4f}")


buy_hold_return = backtest["next_day_return"]

buy_hold_position = pd.Series(1, index=backtest.index)

buy_hold_metrics = calculate_metrics(buy_hold_return, buy_hold_position, 1)


print("\n===== AAPL BUY-AND-HOLD =====")

print(f"Cumulative Return: {buy_hold_metrics['cumulative_return']:.4f}")

print(f"Annualized Return: {buy_hold_metrics['annualized_return']:.4f}")

print(f"Annualized Volatility: {buy_hold_metrics['annualized_volatility']:.4f}")

print(f"Sharpe Ratio: {buy_hold_metrics['sharpe']:.4f}")

print(f"Maximum Drawdown: {buy_hold_metrics['max_drawdown']:.4f}")


print("\n===== TOP 10 THRESHOLDS BY SHARPE =====")

top_thresholds = results_df.sort_values("sharpe", ascending=False).head(10)


print(
    top_thresholds[
        [
            "threshold",
            "cumulative_return",
            "annualized_return",
            "sharpe",
            "max_drawdown",
            "transactions",
            "exposure",
        ]
    ].to_string(index=False)
)


results_df.to_csv(data_dir / "rf_validation_backtest_thresholds.csv", index=False)


best_threshold = float(best_row["threshold"])

with open(data_dir / "rf_best_threshold.txt", "w") as file:
    file.write(str(best_threshold))


print("\nBest threshold saved to:")

print(data_dir / "rf_best_threshold.txt")

print("\nValidation backtest results saved to:")

print(data_dir / "rf_validation_backtest_thresholds.csv")
