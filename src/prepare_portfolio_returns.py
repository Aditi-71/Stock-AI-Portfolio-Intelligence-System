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

data_dir = Path("data/processed")

prices_file = data_dir / "clean_prices.parquet"

output_file = data_dir / "portfolio_returns.parquet"


prices = pd.read_parquet(prices_file)

prices = prices.sort_index()

print("\nPrice data shape:")
print(prices.shape)

print("\nDate range:")
print(prices.index.min(), "to", prices.index.max())


tickers = STOCKS + [BENCHMARK]

adjusted_close = pd.DataFrame(index=prices.index)


for ticker in tickers:
    adjusted_close[ticker] = prices[("Adj Close", ticker)]


print("\nAdjusted close shape:")
print(adjusted_close.shape)


returns = adjusted_close.pct_change(fill_method=None)


returns = returns.replace([np.inf, -np.inf], np.nan)


returns = returns.dropna()


print("\nPortfolio return data shape:")
print(returns.shape)

print("\nReturn date range:")
print(returns.index.min(), "to", returns.index.max())

print("\nMissing values:")
print(returns.isna().sum())

print("\nMean daily returns:")
print(returns.mean().round(6))

print("\nDaily volatility:")
print(returns.std().round(6))


returns.to_parquet(output_file)


print("\nSaved to:", output_file)
