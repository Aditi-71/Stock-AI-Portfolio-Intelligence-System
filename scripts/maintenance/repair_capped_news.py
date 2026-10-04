import os
import time
from pathlib import Path

import pandas as pd
import requests

API_KEY = os.getenv("ALPHAVANTAGE_API_KEY")

if API_KEY is None:
    raise ValueError("ALPHAVANTAGE_API_KEY is not set.")


API_URL = "https://www.alphavantage.co/query"

ticker = "AAPL"

output_dir = Path("data/raw/news/AAPL")

output_dir.mkdir(parents=True, exist_ok=True)


windows = [
    ("2025-10-01", "2025-10-31"),
    ("2025-11-01", "2025-11-30"),
    ("2025-12-01", "2025-12-31"),
    ("2026-01-01", "2026-01-31"),
    ("2026-02-01", "2026-02-28"),
    ("2026-03-01", "2026-03-31"),
    ("2026-04-01", "2026-04-30"),
    ("2026-05-01", "2026-05-31"),
    ("2026-06-01", "2026-06-30"),
    ("2026-07-01", "2026-07-31"),
    ("2026-08-01", "2026-08-26"),
]


for start_string, end_string in windows:
    start = pd.Timestamp(start_string)

    end = pd.Timestamp(end_string)

    filename = f"{ticker}_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}.parquet"

    output_file = output_dir / filename

    if output_file.exists():
        print("Skipping existing:", filename)

        continue

    print()
    print(f"Downloading {ticker}: {start.date()} to {end.date()}")

    params = {
        "function": "NEWS_SENTIMENT",
        "tickers": ticker,
        "time_from": start.strftime("%Y%m%dT0000"),
        "time_to": end.strftime("%Y%m%dT2359"),
        "sort": "EARLIEST",
        "limit": 1000,
        "apikey": API_KEY,
    }

    response = requests.get(API_URL, params=params, timeout=30)

    response.raise_for_status()

    data = response.json()

    if "feed" not in data:
        print("\nAPI returned:")

        print(data)

        print("\nStopping. Run again after quota resets.")

        break

    articles = data["feed"]

    print("Raw articles:", len(articles))

    if len(articles) >= 1000:
        print("WARNING: Even this monthly window hit 1000.")

        print("This month should be split into smaller windows.")

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

    if not news.empty:
        news["time_published"] = pd.to_datetime(
            news["time_published"], format="%Y%m%dT%H%M%S", errors="coerce", utc=True
        )

        news = news.dropna(subset=["time_published", "title"])

        news = news.drop_duplicates(subset=["ticker", "title", "time_published"])

        news = news.sort_values("time_published")

    news.to_parquet(output_file, index=False)

    print("Saved articles:", len(news))

    print("Saved:", output_file)

    time.sleep(1)


print("\nRepair run finished.")
