from datetime import date
from pathlib import Path

import pandas_datareader.data as web
import yaml
import yfinance as yf

with open("config.yaml") as file:
    config = yaml.safe_load(file)


universe = config["universe"]
benchmark = config["benchmark"]
start_date = config["start_date"]


output_dir = Path("data/raw")
output_dir.mkdir(parents=True, exist_ok=True)


tickers = universe + [benchmark, "^VIX"]

print("Downloading market data...")

prices = yf.download(tickers, start=start_date, auto_adjust=False)


output_file = output_dir / "prices.parquet"

prices.to_parquet(output_file)

print(f"Data saved to: {output_file}")
print(f"Downloaded {len(prices)} rows.")


print("Downloading macroeconomic data...")

macro = web.DataReader(["DGS10", "DGS3MO", "CPIAUCSL", "UNRATE"], "fred", start_date, date.today())

macro_file = output_dir / "macro.parquet"
macro.to_parquet(macro_file)

print(f"Macro data saved to: {macro_file}")
print(f"Downloaded {len(macro)} macro rows.")
