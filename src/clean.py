from pathlib import Path

import pandas as pd

prices = pd.read_parquet("data/raw/prices.parquet")
macro = pd.read_parquet("data/raw/macro.parquet")

print("Raw price data shape:", prices.shape)
print("Raw macro data shape:", macro.shape)


prices = prices[~prices.index.duplicated(keep="first")]
macro = macro[~macro.index.duplicated(keep="first")]


prices = prices.sort_index()
macro = macro.sort_index()


stocks = ["AAPL", "MSFT", "NVDA", "JPM", "XOM", "JNJ", "PG", "KO", "CAT", "HD"]

benchmark = "^GSPC"
vix = "^VIX"


print("\nMissing values in price data:")
print(prices.isna().sum().sum())

print("\nMissing values in macro data:")
print(macro.isna().sum())


prices = prices.ffill()


macro = macro.reindex(prices.index)

macro = macro.ffill()


for ticker in stocks:
    high = prices[("High", ticker)]
    low = prices[("Low", ticker)]
    open_price = prices[("Open", ticker)]
    close = prices[("Close", ticker)]

    invalid = (
        (low > high) | (open_price > high) | (open_price < low) | (close > high) | (close < low)
    )

    print(f"{ticker}: {invalid.sum()} invalid OHLC rows")


output_dir = Path("data/processed")
output_dir.mkdir(parents=True, exist_ok=True)


prices.to_parquet(output_dir / "clean_prices.parquet")
macro.to_parquet(output_dir / "clean_macro.parquet")


print("\nCleaning completed!")
print("Saved:")
print("  data/processed/clean_prices.parquet")
print("  data/processed/clean_macro.parquet")
