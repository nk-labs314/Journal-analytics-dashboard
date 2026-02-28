import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

from synthetic_dataset import generate_synthetic_text_dataset
from lexicon_engine import (
    build_global_lexicon,
    predict_mood_from_text
)


# -----------------------------
# Generate Data
# -----------------------------

df = generate_synthetic_text_dataset(
    n_users=6,
    n_entries=2000,
    random_state=42
)

df = df.sort_values(["user_id", "entry_index"])


# -----------------------------
# Chronological Train/Test Split (per user)
# -----------------------------

train_list = []
test_list = []

for user_id, group in df.groupby("user_id"):
    split_idx = int(len(group) * 0.7)
    train_list.append(group.iloc[:split_idx])
    test_list.append(group.iloc[split_idx:])

train_df = pd.concat(train_list)
test_df = pd.concat(test_list)


# -----------------------------
# Build Lexicon ONLY on Train
# -----------------------------

global_lexicon, global_counts, global_mean = build_global_lexicon(train_df)


# -----------------------------
# Predict on Test
# -----------------------------

predictions = []

for _, row in test_df.iterrows():
    pred, _ = predict_mood_from_text(
        row["text"],
        global_lexicon,
        global_counts,
        {},         # no personalization yet
        {},
        global_mean,
        k=10
    )
    predictions.append(pred)

test_df["predicted"] = predictions


# -----------------------------
# Metrics
# -----------------------------

mae = mean_absolute_error(test_df["mood_score"], test_df["predicted"])

baseline_mae = mean_absolute_error(
    test_df["mood_score"],
    np.full(len(test_df), global_mean)
)

r2 = r2_score(test_df["mood_score"], test_df["predicted"])

corr = np.corrcoef(
    test_df["mood_score"],
    test_df["predicted"]
)[0, 1]


print("\n--- Lexicon Evaluation ---")
print("Test MAE:", round(mae, 4))
print("Baseline MAE:", round(baseline_mae, 4))
print("R²:", round(r2, 4))
print("Correlation:", round(corr, 4))
print(test_df["mood_score"].min(), test_df["mood_score"].max())
print(test_df["predicted"].min(), test_df["predicted"].max())