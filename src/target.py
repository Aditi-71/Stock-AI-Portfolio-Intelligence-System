from pathlib import Path

import pandas as pd

features_file = Path("data/processed/features.parquet")

features = pd.read_parquet(features_file)


current_return = features["AAPL_ret_1d"]

next_day_return = current_return.shift(-1)


target = pd.Series(pd.NA, index=features.index, dtype="Int64")

target.loc[next_day_return > 0] = 1
target.loc[next_day_return <= 0] = 0


model_data = features.copy()

model_data["target"] = target


model_data = model_data.dropna(subset=["target"])


output_file = Path("data/processed/model_data.parquet")

model_data.to_parquet(output_file)


print("Target creation completed!")
print()
print("Model dataset shape:")
print(model_data.shape)

print()
print("Target distribution:")
print(model_data["target"].value_counts())

print()
print("Target percentages:")
print(model_data["target"].value_counts(normalize=True))

print()
print(f"Saved to: {output_file}")
