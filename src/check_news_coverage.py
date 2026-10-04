import re
from pathlib import Path

import pandas as pd

news_dir = Path("data/raw/news")

sentiment_file = Path("data/processed/daily_sentiment.parquet")

prices_file = Path("data/processed/clean_prices.parquet")

output_file = Path("data/processed/news_coverage.parquet")


prices = pd.read_parquet(prices_file)

trading_dates = pd.DatetimeIndex(prices.index)

trading_dates = pd.to_datetime(trading_dates).normalize()


daily_sentiment = pd.read_parquet(sentiment_file)

daily_sentiment["session_date"] = pd.to_datetime(daily_sentiment["session_date"]).dt.normalize()


coverage_windows = []


pattern = re.compile(
    r"([A-Z]+)_"
    r"(\d{8})_"
    r"(\d{8})\.parquet"
)


for file in news_dir.glob("*/*.parquet"):
    match = pattern.match(file.name)

    if match is None:
        continue

    ticker = match.group(1)

    start_date = pd.to_datetime(match.group(2), format="%Y%m%d")

    end_date = pd.to_datetime(match.group(3), format="%Y%m%d")

    coverage_windows.append({"ticker": ticker, "start_date": start_date, "end_date": end_date})


coverage_windows = pd.DataFrame(coverage_windows)


print("\nDownloaded windows:")

print(len(coverage_windows))


print("\nWindows by ticker:")

if not coverage_windows.empty:
    print(coverage_windows["ticker"].value_counts().sort_index())


coverage_rows = []


for ticker in sorted(coverage_windows["ticker"].unique()):
    ticker_windows = coverage_windows[coverage_windows["ticker"] == ticker]

    ticker_sentiment = daily_sentiment[daily_sentiment["ticker"] == ticker]

    news_dates = set(ticker_sentiment["session_date"])

    for date in trading_dates:
        downloaded = (
            (ticker_windows["start_date"] <= date) & (ticker_windows["end_date"] >= date)
        ).any()

        has_news = date in news_dates

        if downloaded and has_news:
            status = "covered_with_news"

        elif downloaded:
            status = "covered_no_news"

        else:
            status = "not_downloaded"

        coverage_rows.append(
            {
                "session_date": date,
                "ticker": ticker,
                "downloaded": downloaded,
                "has_news": has_news,
                "coverage_status": status,
            }
        )


coverage = pd.DataFrame(coverage_rows)


coverage.to_parquet(output_file, index=False)


print("\n===== NEWS COVERAGE SUMMARY =====")


for ticker in sorted(coverage["ticker"].unique()):
    ticker_data = coverage[coverage["ticker"] == ticker]

    print()
    print(f"Ticker: {ticker}")

    print(ticker_data["coverage_status"].value_counts())

    downloaded_data = ticker_data[ticker_data["downloaded"]]

    if len(downloaded_data) > 0:
        coverage_rate = downloaded_data["has_news"].mean()

        print(f"News-day coverage among downloaded trading days: {coverage_rate:.2%}")

        print("Downloaded trading-date range:")

        print(downloaded_data["session_date"].min(), "to", downloaded_data["session_date"].max())


print(f"\nSaved to: {output_file}")
