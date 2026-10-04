from pathlib import Path

import pandas as pd

data_dir = Path("data/processed")

model_file = data_dir / "model_data.parquet"

sentiment_file = data_dir / "daily_sentiment.parquet"

coverage_file = data_dir / "news_coverage.parquet"

output_file = data_dir / "model_data_with_sentiment.parquet"


model_data = pd.read_parquet(model_file)

daily_sentiment = pd.read_parquet(sentiment_file)

coverage = pd.read_parquet(coverage_file)


daily_sentiment = daily_sentiment[daily_sentiment["ticker"] == "AAPL"].copy()

coverage = coverage[coverage["ticker"] == "AAPL"].copy()


daily_sentiment["session_date"] = pd.to_datetime(daily_sentiment["session_date"])

coverage["session_date"] = pd.to_datetime(coverage["session_date"])


coverage = coverage[coverage["downloaded"]].copy()


downloaded_dates = coverage["session_date"]


model_data = model_data[model_data.index.isin(downloaded_dates)].copy()


print("\nRows inside downloaded news period:")

print(len(model_data))


sentiment_columns = [
    "session_date",
    "sentiment_mean",
    "sentiment_std",
    "positive_ratio",
    "negative_ratio",
    "neutral_ratio",
    "sentiment_balance",
    "article_count",
    "log_article_count",
]


sentiment = daily_sentiment[sentiment_columns].copy()


sentiment = sentiment.rename(
    columns={
        "session_date": "Date",
        "sentiment_mean": "AAPL_sentiment_mean",
        "sentiment_std": "AAPL_sentiment_std",
        "positive_ratio": "AAPL_positive_ratio",
        "negative_ratio": "AAPL_negative_ratio",
        "neutral_ratio": "AAPL_neutral_ratio",
        "sentiment_balance": "AAPL_sentiment_balance",
        "article_count": "AAPL_article_count",
        "log_article_count": "AAPL_log_article_count",
    }
)


sentiment = sentiment.set_index("Date")


combined = model_data.join(sentiment, how="left")


sentiment_feature_names = [
    "AAPL_sentiment_mean",
    "AAPL_sentiment_std",
    "AAPL_positive_ratio",
    "AAPL_negative_ratio",
    "AAPL_neutral_ratio",
    "AAPL_sentiment_balance",
    "AAPL_article_count",
    "AAPL_log_article_count",
]


combined[sentiment_feature_names] = combined[sentiment_feature_names].fillna(0.0)


combined["AAPL_has_news"] = (combined["AAPL_article_count"] > 0).astype(int)


before = len(combined)

combined = combined.dropna()

removed = before - len(combined)


combined.to_parquet(output_file)


print("\nSentiment dataset created!")

print("\nFinal shape:")

print(combined.shape)

print("\nDate range:")

print(combined.index.min(), "to", combined.index.max())

print("\nRows removed because of technical-feature warmup:")

print(removed)

print("\nDays with news:")

print(combined["AAPL_has_news"].sum())

print("\nDays with no news:")

print((combined["AAPL_has_news"] == 0).sum())

print("\nTarget distribution:")

print(combined["target"].value_counts(normalize=True))

print(f"\nSaved to: {output_file}")
