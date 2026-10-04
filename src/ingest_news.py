import os
import time
from pathlib import Path

import pandas as pd
import requests

API_KEY = os.getenv("ALPHAVANTAGE_API_KEY")

if API_KEY is None:
    raise ValueError("ALPHAVANTAGE_API_KEY is not set.")


stocks = ["AAPL", "MSFT", "NVDA", "JPM", "XOM", "JNJ", "PG", "KO", "CAT", "HD"]


START_DATE = "2015-01-01"
END_DATE = "2026-08-26"


raw_news_dir = Path("data/raw/news")

raw_news_dir.mkdir(parents=True, exist_ok=True)


API_URL = "https://www.alphavantage.co/query"


def create_date_windows(start_date, end_date):

    start = pd.Timestamp(start_date)

    end = pd.Timestamp(end_date)

    windows = []

    current = start

    while current <= end:
        window_end = current + pd.DateOffset(months=3) - pd.Timedelta(days=1)

        if window_end > end:
            window_end = end

        windows.append((current, window_end))

        current = window_end + pd.Timedelta(days=1)

    return windows


date_windows = create_date_windows(START_DATE, END_DATE)


print("\nNumber of date windows:", len(date_windows))


def download_news(ticker, start, end):

    time_from = start.strftime("%Y%m%dT0000")

    time_to = end.strftime("%Y%m%dT2359")

    params = {
        "function": "NEWS_SENTIMENT",
        "tickers": ticker,
        "time_from": time_from,
        "time_to": time_to,
        "sort": "EARLIEST",
        "limit": 1000,
        "apikey": API_KEY,
    }

    response = requests.get(API_URL, params=params, timeout=30)

    response.raise_for_status()

    return response.json()


def process_articles(ticker, data):

    articles = data.get("feed", [])

    rows = []

    for article in articles:
        rows.append(
            {
                "ticker": ticker,
                "time_published": article.get("time_published"),
                "title": article.get("title"),
                "summary": article.get("summary"),
                "source": article.get("source"),
                "url": article.get("url"),
            }
        )

    news = pd.DataFrame(rows)

    if news.empty:
        return news

    news["time_published"] = pd.to_datetime(
        news["time_published"], format="%Y%m%dT%H%M%S", errors="coerce", utc=True
    )

    news = news.dropna(subset=["time_published", "title"])

    news = news.drop_duplicates(subset=["ticker", "title", "time_published"])

    news = news.sort_values("time_published")

    return news


request_count = 0

rate_limit_hit = False


for ticker in stocks:
    print()
    print("=" * 60)

    print(f"Ticker: {ticker}")

    print("=" * 60)

    ticker_dir = raw_news_dir / ticker

    ticker_dir.mkdir(parents=True, exist_ok=True)

    for start, end in date_windows:
        filename = f"{ticker}_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}.parquet"

        output_file = ticker_dir / filename

        if output_file.exists():
            print(f"Skipping existing: {filename}")

            continue

        print(f"\nDownloading {ticker}: {start.date()} to {end.date()}")

        try:
            data = download_news(ticker, start, end)

        except Exception as error:
            print("Request failed:")

            print(error)

            rate_limit_hit = True

            break

        request_count += 1

        if "feed" not in data:
            print("\nAPI returned:")

            print(data)

            if "Information" in data or "Note" in data:
                print("\nAPI limit probably reached.")

                rate_limit_hit = True

                break

            empty_df = pd.DataFrame(
                columns=["ticker", "time_published", "title", "summary", "source", "url"]
            )

            empty_df.to_parquet(output_file, index=False)

            continue

        news = process_articles(ticker, data)

        print("Articles:", len(news))

        if len(data.get("feed", [])) >= 1000:
            print("WARNING: API returned 1000 articles.")

            print("This period may need to be split into smaller windows.")

        news.to_parquet(output_file, index=False)

        print("Saved:", output_file)

        time.sleep(1)

    if rate_limit_hit:
        break


print()
print("=" * 60)

print("NEWS INGESTION RUN FINISHED")

print("=" * 60)

print("API requests made:", request_count)


if rate_limit_hit:
    print("\nThe script stopped because the API limit was reached.")

    print("Run the same script again later.")

    print("Existing files will automatically be skipped.")

else:
    print("\nAll requested news windows were processed.")
