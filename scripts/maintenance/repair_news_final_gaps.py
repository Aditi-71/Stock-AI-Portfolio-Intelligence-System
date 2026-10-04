import os
import time
from pathlib import Path

import pandas as pd
import requests

API_KEY = os.getenv("ALPHAVANTAGE_API_KEY")

if not API_KEY:
    raise ValueError("ALPHAVANTAGE_API_KEY is not set.")


API_URL = "https://www.alphavantage.co/query"

ticker = "AAPL"

output_dir = Path("data/raw/news/AAPL")

output_dir.mkdir(parents=True, exist_ok=True)


windows = [
    ("2025-10-16", "2025-10-22"),
    ("2025-10-23", "2025-10-29"),
    ("2025-10-30", "2025-10-31"),
    ("2026-08-01", "2026-08-15"),
    ("2026-08-16", "2026-08-26"),
]


requests_made = 0


for start_string, end_string in windows:
    start = pd.Timestamp(start_string)
    end = pd.Timestamp(end_string)

    filename = f"{ticker}_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}_finalrepair.parquet"

    output_file = output_dir / filename

    if output_file.exists():
        print("Skipping existing:", filename)

        continue

    print()
    print("=" * 60)

    print(f"Downloading {ticker}: {start.date()} to {end.date()}")

    print("=" * 60)

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

    requests_made += 1

    data = response.json()

    if "feed" not in data:
        print()
        print("API returned:")
        print(data)

        print("\nStopping. Run this same script later.")

        break

    articles = data["feed"]

    print("Raw articles:", len(articles))

    if len(articles) >= 1000:
        print("WARNING: THIS WINDOW STILL HIT 1000.")

        print("It must be split again.")

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


print()
print("=" * 60)

print("FINAL GAP REPAIR FINISHED")

print("API requests made:", requests_made)

print("=" * 60)
