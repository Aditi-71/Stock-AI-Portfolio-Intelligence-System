from pathlib import Path

import pandas as pd

news_dir = Path("data/raw/news")

output_file = Path("data/raw/news_raw.parquet")


files = list(news_dir.glob("*/*.parquet"))


print("\nNews files found:", len(files))


if len(files) == 0:
    raise FileNotFoundError("No news parquet files were found.")


dataframes = []


for file in files:
    df = pd.read_parquet(file)

    if df.empty:
        continue

    dataframes.append(df)


if len(dataframes) == 0:
    raise ValueError("All downloaded news files are empty.")


news = pd.concat(dataframes, ignore_index=True)


print("Rows before cleaning:", len(news))


news["time_published"] = pd.to_datetime(news["time_published"], utc=True, errors="coerce")


news = news.dropna(subset=["ticker", "time_published", "title"])


before = len(news)


news = news.drop_duplicates(subset=["ticker", "title", "time_published"])


duplicates_removed = before - len(news)


news = news.sort_values(["time_published", "ticker"]).reset_index(drop=True)


news.to_parquet(output_file, index=False)


print("\nCombined news dataset created!")

print("\nShape:")

print(news.shape)

print("\nDuplicates removed:")

print(duplicates_removed)

print("\nDate range:")

print(news["time_published"].min(), "to", news["time_published"].max())

print("\nArticles by ticker:")

print(news["ticker"].value_counts().sort_index())

print("\nMissing values:")

print(news.isna().sum())

print(f"\nSaved to: {output_file}")
