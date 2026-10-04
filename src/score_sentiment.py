import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_NAME = "ProsusAI/finbert"

BATCH_SIZE = 16

MAX_LENGTH = 256


input_file = Path("data/processed/clean_news.parquet")

output_file = Path("data/processed/news_sentiment.parquet")


news = pd.read_parquet(input_file)


print("\nArticles loaded:")
print(len(news))


print("\nLoading FinBERT...")


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model.to(device)

model.eval()


print("\nDevice:", device)


print("\nModel labels:")

print(model.config.id2label)


positive_scores = []
negative_scores = []
neutral_scores = []


texts = news["text"].fillna("").astype(str).tolist()


print("\nScoring sentiment...")


for start in range(0, len(texts), BATCH_SIZE):
    end = min(start + BATCH_SIZE, len(texts))

    batch_texts = texts[start:end]

    encoded = tokenizer(
        batch_texts, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt"
    )

    encoded = {key: value.to(device) for key, value in encoded.items()}

    with torch.no_grad():
        outputs = model(**encoded)

        probabilities = torch.softmax(outputs.logits, dim=1)

    probabilities = probabilities.cpu().numpy()

    positive_scores.extend(probabilities[:, 0])

    negative_scores.extend(probabilities[:, 1])

    neutral_scores.extend(probabilities[:, 2])

    print(f"Processed {end}/{len(texts)}", end="\r")


news["sentiment_positive"] = positive_scores

news["sentiment_negative"] = negative_scores

news["sentiment_neutral"] = neutral_scores


news["sentiment_score"] = news["sentiment_positive"] - news["sentiment_negative"]


probability_columns = ["sentiment_positive", "sentiment_negative", "sentiment_neutral"]


label_names = np.array(["positive", "negative", "neutral"])


highest_probability = news[probability_columns].values.argmax(axis=1)


news["sentiment_label"] = label_names[highest_probability]


news.to_parquet(output_file, index=False)


print("\n\nSentiment scoring completed!")


print("\nShape:")

print(news.shape)


print("\nSentiment distribution:")

print(news["sentiment_label"].value_counts())


print("\nSentiment score statistics:")

print(news["sentiment_score"].describe())


print("\nExample articles:")


print(
    news[
        [
            "session_date",
            "ticker",
            "title",
            "sentiment_label",
            "sentiment_positive",
            "sentiment_negative",
            "sentiment_neutral",
            "sentiment_score",
        ]
    ]
    .head(10)
    .to_string(index=False)
)


print(f"\nSaved to: {output_file}")
