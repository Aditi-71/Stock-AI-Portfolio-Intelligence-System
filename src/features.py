from pathlib import Path

import numpy as np
import pandas as pd

prices = pd.read_parquet("data/processed/clean_prices.parquet")

macro = pd.read_parquet("data/processed/clean_macro.parquet")


stocks = ["AAPL", "MSFT", "NVDA", "JPM", "XOM", "JNJ", "PG", "KO", "CAT", "HD"]

benchmark = "^GSPC"
vix_ticker = "^VIX"


feature_dict = {}


for ticker in stocks:
    adj_close = prices[("Adj Close", ticker)]
    raw_close = prices[("Close", ticker)]

    raw_high = prices[("High", ticker)]
    raw_low = prices[("Low", ticker)]

    volume = prices[("Volume", ticker)]

    adjustment_factor = adj_close / raw_close

    high = raw_high * adjustment_factor
    low = raw_low * adjustment_factor

    log_price = np.log(adj_close)

    feature_dict[f"{ticker}_ret_1d"] = log_price.diff(1)

    feature_dict[f"{ticker}_ret_5d"] = log_price.diff(5)

    feature_dict[f"{ticker}_ret_10d"] = log_price.diff(10)

    feature_dict[f"{ticker}_ret_21d"] = log_price.diff(21)

    sma10 = adj_close.rolling(10).mean()
    sma20 = adj_close.rolling(20).mean()
    sma50 = adj_close.rolling(50).mean()
    sma200 = adj_close.rolling(200).mean()

    ema10 = adj_close.ewm(span=10, adjust=False).mean()

    ema20 = adj_close.ewm(span=20, adjust=False).mean()

    feature_dict[f"{ticker}_p_sma10"] = adj_close / sma10

    feature_dict[f"{ticker}_p_sma20"] = adj_close / sma20

    feature_dict[f"{ticker}_p_sma50"] = adj_close / sma50

    feature_dict[f"{ticker}_p_sma200"] = adj_close / sma200

    feature_dict[f"{ticker}_p_ema20"] = adj_close / ema20

    ema12 = adj_close.ewm(span=12, adjust=False).mean()

    ema26 = adj_close.ewm(span=26, adjust=False).mean()

    macd = ema12 - ema26

    signal = macd.ewm(span=9, adjust=False).mean()

    feature_dict[f"{ticker}_MACD_pct"] = macd / adj_close

    feature_dict[f"{ticker}_MACD_signal_pct"] = signal / adj_close

    delta = adj_close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss

    feature_dict[f"{ticker}_RSI14"] = 100 - (100 / (1 + rs))

    feature_dict[f"{ticker}_ROC10"] = adj_close.pct_change(10)

    daily_return = log_price.diff()

    feature_dict[f"{ticker}_volatility_20d"] = daily_return.rolling(20).std()

    feature_dict[f"{ticker}_volatility_60d"] = daily_return.rolling(60).std()

    previous_close = adj_close.shift(1)

    true_range_1 = high - low

    true_range_2 = (high - previous_close).abs()

    true_range_3 = (low - previous_close).abs()

    true_range = pd.concat([true_range_1, true_range_2, true_range_3], axis=1).max(axis=1)

    atr14 = true_range.rolling(14).mean()

    feature_dict[f"{ticker}_ATR14_pct"] = atr14 / adj_close

    middle_band = adj_close.rolling(20).mean()

    rolling_std = adj_close.rolling(20).std()

    upper_band = middle_band + 2 * rolling_std

    lower_band = middle_band - 2 * rolling_std

    feature_dict[f"{ticker}_BB_width"] = (upper_band - lower_band) / middle_band

    feature_dict[f"{ticker}_BB_position"] = (adj_close - lower_band) / (upper_band - lower_band)

    volume_mean = volume.rolling(20).mean()

    volume_std = volume.rolling(20).std()

    feature_dict[f"{ticker}_Volume_z"] = (volume - volume_mean) / volume_std

    feature_dict[f"{ticker}_Volume_change"] = volume.pct_change()

    feature_dict[f"{ticker}_ret_lag1"] = daily_return.shift(1)

    feature_dict[f"{ticker}_ret_lag2"] = daily_return.shift(2)

    feature_dict[f"{ticker}_ret_lag3"] = daily_return.shift(3)


features = pd.DataFrame(feature_dict, index=prices.index)


benchmark_adj = prices[("Adj Close", benchmark)]

vix_close = prices[("Close", vix_ticker)]


features["market_return"] = np.log(benchmark_adj).diff()


features["VIX"] = vix_close


features["VIX_change"] = vix_close.pct_change()


features["day_of_week"] = features.index.dayofweek

features["month"] = features.index.month

features["turn_of_month"] = ((features.index.day <= 3) | (features.index.day >= 28)).astype(int)


macro_features = macro[["DGS10", "DGS3MO", "CPIAUCSL", "UNRATE"]].copy()


macro_features = macro_features.ffill()


macro_features["CPIAUCSL"] = macro_features["CPIAUCSL"].shift(21)

macro_features["UNRATE"] = macro_features["UNRATE"].shift(21)


features = pd.concat([features, macro_features], axis=1)


features["yield_spread"] = features["DGS10"] - features["DGS3MO"]


features = features.replace([np.inf, -np.inf], np.nan)


output_dir = Path("data/processed")

output_dir.mkdir(parents=True, exist_ok=True)

output_file = output_dir / "features.parquet"

features.to_parquet(output_file)


print()
print("Feature engineering completed!")

print()

print("Feature dataset shape:")

print(features.shape)

print()

print("Total missing values:")

print(features.isna().sum().sum())

print()

print("Number of features:")

print(features.shape[1])

print()

print(f"Saved to: {output_file}")
