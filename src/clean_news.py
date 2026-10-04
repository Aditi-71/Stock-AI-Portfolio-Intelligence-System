from pathlib import Path

import pandas as pd
import pandas_market_calendars as mcal

input_file = Path("data/raw/news_raw.parquet")

output_dir = Path("data/processed")

output_dir.mkdir(parents=True, exist_ok=True)

output_file = output_dir / "clean_news.parquet"


news = pd.read_parquet(input_file)

print("\nRaw news rows:")
print(len(news))


news["time_published"] = pd.to_datetime(news["time_published"], utc=True, errors="coerce")


news = news.dropna(subset=["ticker", "time_published", "title"]).copy()


news["summary"] = news["summary"].fillna("").astype(str)

news["title"] = news["title"].fillna("").astype(str)


news["text"] = news["title"].str.strip() + ". " + news["summary"].str.strip()

news["text"] = news["text"].str.replace(r"\s+", " ", regex=True).str.strip()


news["time_et"] = news["time_published"].dt.tz_convert("America/New_York")


nyse = mcal.get_calendar("NYSE")


start_date = news["time_et"].min().date() - pd.Timedelta(days=10)

end_date = news["time_et"].max().date() + pd.Timedelta(days=10)


schedule = nyse.schedule(start_date=start_date, end_date=end_date)


schedule["market_close_et"] = schedule["market_close"].dt.tz_convert("America/New_York")


trading_dates = list(schedule.index)


market_close_by_date = {
    date: close for date, close in zip(schedule.index, schedule["market_close_et"])
}


def assign_trading_session(timestamp_et):

    article_date = pd.Timestamp(timestamp_et.date())

    if article_date in market_close_by_date:
        close_time = market_close_by_date[article_date]

        if timestamp_et <= close_time:
            return article_date

    future_dates = [date for date in trading_dates if date > article_date]

    if len(future_dates) == 0:
        return pd.NaT

    return future_dates[0]


news["session_date"] = news["time_et"].apply(assign_trading_session)


before = len(news)

news = news.dropna(subset=["session_date"]).copy()

removed = before - len(news)


news = news.drop_duplicates(subset=["ticker", "time_published", "title"])


news = news.sort_values(["session_date", "time_published", "ticker"]).reset_index(drop=True)


news.to_parquet(output_file, index=False)


print("\nNews cleaning completed!")

print("\nClean rows:")

print(len(news))

print("\nRows removed because session could not be mapped:")

print(removed)

print("\nSession date range:")

print(news["session_date"].min(), "to", news["session_date"].max())


print("\nArticles by ticker:")

print(news["ticker"].value_counts().sort_index())


after_hours = (
    news["time_et"].dt.normalize()
    != pd.to_datetime(news["session_date"]).dt.tz_localize("America/New_York").dt.normalize()
)


print("\nArticles mapped to a later trading date:")

print(after_hours.sum())


print("\nExample mapped articles:")

print(
    news[["time_published", "time_et", "session_date", "ticker", "title"]]
    .head(10)
    .to_string(index=False)
)


print(f"\nSaved to: {output_file}")
