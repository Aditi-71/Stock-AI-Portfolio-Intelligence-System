from pathlib import Path

import numpy as np
import pandas as pd

input_file = Path("data/processed/news_sentiment.parquet")

output_file = Path("data/processed/daily_sentiment.parquet")


news = pd.read_parquet(input_file)

news["session_date"] = pd.to_datetime(news["session_date"])


print("\nArticles loaded:")
print(len(news))


news["is_positive"] = (news["sentiment_label"] == "positive").astype(int)

news["is_negative"] = (news["sentiment_label"] == "negative").astype(int)

news["is_neutral"] = (news["sentiment_label"] == "neutral").astype(int)


daily = (
    news.groupby(["session_date", "ticker"])
    .agg(
        sentiment_mean=("sentiment_score", "mean"),
        sentiment_median=("sentiment_score", "median"),
        sentiment_std=("sentiment_score", "std"),
        sentiment_min=("sentiment_score", "min"),
        sentiment_max=("sentiment_score", "max"),
        positive_prob_mean=("sentiment_positive", "mean"),
        negative_prob_mean=("sentiment_negative", "mean"),
        neutral_prob_mean=("sentiment_neutral", "mean"),
        article_count=("title", "count"),
        positive_count=("is_positive", "sum"),
        negative_count=("is_negative", "sum"),
        neutral_count=("is_neutral", "sum"),
    )
    .reset_index()
)


daily["sentiment_std"] = daily["sentiment_std"].fillna(0.0)


daily["positive_ratio"] = daily["positive_count"] / daily["article_count"]

daily["negative_ratio"] = daily["negative_count"] / daily["article_count"]

daily["neutral_ratio"] = daily["neutral_count"] / daily["article_count"]


daily["sentiment_balance"] = (daily["positive_count"] - daily["negative_count"]) / daily[
    "article_count"
]


daily["log_article_count"] = np.log1p(daily["article_count"])


daily = daily.sort_values(["session_date", "ticker"]).reset_index(drop=True)


daily.to_parquet(output_file, index=False)


print("\nDaily sentiment aggregation completed!")

print("\nShape:")

print(daily.shape)

print("\nDate range:")

print(daily["session_date"].min(), "to", daily["session_date"].max())

print("\nTrading days with news:")

print(daily.groupby("ticker")["session_date"].nunique())

print("\nArticles per news day:")

print(daily["article_count"].describe())

print("\nDaily sentiment statistics:")

print(daily["sentiment_mean"].describe())

print("\nFirst 10 daily rows:")

print(
    daily[
        [
            "session_date",
            "ticker",
            "article_count",
            "sentiment_mean",
            "sentiment_std",
            "positive_ratio",
            "negative_ratio",
            "neutral_ratio",
            "sentiment_balance",
        ]
    ]
    .head(10)
    .to_string(index=False)
)

print(f"\nSaved to: {output_file}")
