from pathlib import Path

import numpy as np
import pandas as pd

data_dir = Path("data/processed")

THRESHOLD = 0.50
TRANSACTION_COST = 0.001
TRADING_DAYS = 252


predictions = pd.read_parquet(data_dir / "final_rf_test_predictions.parquet")


prices = pd.read_parquet(data_dir / "clean_prices.parquet")

aapl = prices[("Adj Close", "AAPL")].copy()


next_day_return = aapl.shift(-1) / aapl - 1

next_day_return.name = "next_day_return"


backtest = predictions.join(next_day_return, how="left")

backtest = backtest.dropna(subset=["next_day_return"])


position = (backtest["probability_up"] >= THRESHOLD).astype(int)


gross_return = position * backtest["next_day_return"]


position_change = position.diff().abs().fillna(position.iloc[0])

trading_cost = position_change * TRANSACTION_COST

strategy_return = gross_return - trading_cost


def calculate_metrics(returns, positions):

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
        "exposure": exposure,
    }


strategy_metrics = calculate_metrics(strategy_return, position)


buy_hold_return = backtest["next_day_return"]

buy_hold_position = pd.Series(1, index=backtest.index)

buy_hold_metrics = calculate_metrics(buy_hold_return, buy_hold_position)


transactions = int(position_change.sum())

days_in_market = int(position.sum())


print("\n===== FINAL TEST BACKTEST =====")

print(f"Locked Threshold: {THRESHOLD:.2f}")

print(f"Cumulative Return: {strategy_metrics['cumulative_return']:.4f}")

print(f"Annualized Return: {strategy_metrics['annualized_return']:.4f}")

print(f"Annualized Volatility: {strategy_metrics['annualized_volatility']:.4f}")

print(f"Sharpe Ratio: {strategy_metrics['sharpe']:.4f}")

print(f"Maximum Drawdown: {strategy_metrics['max_drawdown']:.4f}")

print(f"Market Exposure: {strategy_metrics['exposure']:.4f}")

print(f"Transactions: {transactions}")

print(f"Days in Market: {days_in_market}")


print("\n===== AAPL BUY-AND-HOLD TEST =====")

print(f"Cumulative Return: {buy_hold_metrics['cumulative_return']:.4f}")

print(f"Annualized Return: {buy_hold_metrics['annualized_return']:.4f}")

print(f"Annualized Volatility: {buy_hold_metrics['annualized_volatility']:.4f}")

print(f"Sharpe Ratio: {buy_hold_metrics['sharpe']:.4f}")

print(f"Maximum Drawdown: {buy_hold_metrics['max_drawdown']:.4f}")


backtest["position"] = position

backtest["strategy_return"] = strategy_return

backtest["buy_hold_return"] = buy_hold_return

backtest["strategy_equity"] = (1 + strategy_return).cumprod()

backtest["buy_hold_equity"] = (1 + buy_hold_return).cumprod()


output_file = data_dir / "final_rf_backtest.parquet"

backtest.to_parquet(output_file)


print("\nSaved final backtest to:")

print(output_file)
