from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

data_dir = Path("data/processed")

clean_file = data_dir / "clean_news.parquet"
sentiment_file = data_dir / "news_sentiment.parquet"

MODEL_NAME = "ProsusAI/finbert"

BATCH_SIZE = 16
MAX_LENGTH = 256


clean = pd.read_parquet(clean_file)

clean["time_published"] = pd.to_datetime(clean["time_published"], utc=True)

print("\nClean articles:")
print(len(clean))


if sentiment_file.exists():
    old = pd.read_parquet(sentiment_file)

    old["time_published"] = pd.to_datetime(old["time_published"], utc=True)

else:
    old = pd.DataFrame()


print("\nAlready scored:")
print(len(old))


key_columns = ["ticker", "time_published", "title"]


if len(old) > 0:
    old_keys = set(old[key_columns].itertuples(index=False, name=None))

    mask = [key not in old_keys for key in clean[key_columns].itertuples(index=False, name=None)]

    new_articles = clean[mask].copy()

else:
    new_articles = clean.copy()


print("\nNew articles requiring FinBERT:")
print(len(new_articles))


if len(new_articles) == 0:
    print("\nNothing new to score.")

    raise SystemExit


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("\nDevice:")
print(device)


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

model.to(device)

model.eval()


id2label = {int(k): str(v).lower() for k, v in model.config.id2label.items()}


print("\nFinBERT labels:")
print(id2label)


label_to_id = {label: idx for idx, label in id2label.items()}


positive_id = label_to_id["positive"]
negative_id = label_to_id["negative"]
neutral_id = label_to_id["neutral"]


positive_probs = []
negative_probs = []
neutral_probs = []
sentiment_scores = []
sentiment_labels = []


texts = new_articles["text"].fillna("").astype(str).tolist()


total = len(texts)


for start in range(0, total, BATCH_SIZE):
    end = min(start + BATCH_SIZE, total)

    batch_texts = texts[start:end]

    encoded = tokenizer(
        batch_texts, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt"
    )

    encoded = {key: value.to(device) for key, value in encoded.items()}

    with torch.no_grad():
        outputs = model(**encoded)

        probs = torch.softmax(outputs.logits, dim=1)

    probs = probs.cpu().numpy()

    for row in probs:
        pos = float(row[positive_id])

        neg = float(row[negative_id])

        neu = float(row[neutral_id])

        score = pos - neg

        predicted_id = int(np.argmax(row))

        label = id2label[predicted_id]

        positive_probs.append(pos)
        negative_probs.append(neg)
        neutral_probs.append(neu)

        sentiment_scores.append(score)

        sentiment_labels.append(label)

    if end % 500 < BATCH_SIZE or end == total:
        print(f"Scored {end}/{total}")


new_articles["positive_prob"] = positive_probs

new_articles["negative_prob"] = negative_probs

new_articles["neutral_prob"] = neutral_probs

new_articles["sentiment_score"] = sentiment_scores

new_articles["sentiment_label"] = sentiment_labels


if len(old) > 0:
    combined = pd.concat([old, new_articles], ignore_index=True, sort=False)

else:
    combined = new_articles.copy()


combined = combined.drop_duplicates(subset=key_columns, keep="last")


combined = combined.sort_values("time_published")


combined.to_parquet(sentiment_file, index=False)


print()
print("=" * 60)

print("INCREMENTAL SENTIMENT SCORING COMPLETE")

print("=" * 60)

print("Previously scored:", len(old))

print("Newly scored:", len(new_articles))

print("Final sentiment rows:", len(combined))


print("\nTickers:")

print(combined["ticker"].value_counts())


print("\nPublication range:")

print(combined["time_published"].min(), "to", combined["time_published"].max())


print("\nSaved to:", sentiment_file)
