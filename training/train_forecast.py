import os
import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge

from models.feature_builder import build_features, FEATURE_COLUMNS
from models.lexicon_model import build_global_lexicon


HORIZONS = [3, 7, 14]
ARTIFACT_PATH = os.path.join("artifacts", "ridge_multi_output.pkl")
LEXICON_ARTIFACT_PATH = os.path.join("artifacts", "global_lexicon.pkl")


def build_targets(df):
    df = df.copy()

    for h in HORIZONS:
        df[f"target_{h}"] = (
            df["mood_score"]
            .shift(-1)
            .rolling(window=h)
            .mean()
        )

    return df


def train_model(df, lexicon):
    df = df.sort_values(["user_id", "entry_index"])

    train_rows = []

    for user_id, user_df in df.groupby("user_id"):
        user_df = build_features(user_df, lexicon=lexicon)
        user_df = build_targets(user_df)
        user_df = user_df.dropna()
        train_rows.append(user_df)

    full_df = pd.concat(train_rows)

    X = full_df[FEATURE_COLUMNS]
    Y = full_df[[f"target_{h}" for h in HORIZONS]]

    model = Ridge(alpha=1.0)
    model.fit(X, Y)

    return model


if __name__ == "__main__":
    # Load synthetic dataset
    df = pd.read_csv("data/synthetic_dataset_v3.csv")

    # Build lexicon from training data
    print("Building global lexicon...")
    global_lexicon, global_counts, global_mean = build_global_lexicon(df)
    lexicon = {
        "global_lexicon": global_lexicon,
        "global_counts": global_counts,
        "global_mean": global_mean,
    }
    print(f"  Lexicon size: {len(global_lexicon)} words, global mean: {global_mean:.2f}")

    # Train model with lexicon scoring
    print("Training forecast model with lexicon integration...")
    model = train_model(df, lexicon)

    os.makedirs("artifacts", exist_ok=True)

    # Save everything in a single artifact
    artifact = {
        "model": model,
        "feature_columns": FEATURE_COLUMNS,
        "horizons": HORIZONS,
        "global_lexicon": global_lexicon,
        "global_counts": global_counts,
        "global_mean": global_mean,
    }

    joblib.dump(artifact, ARTIFACT_PATH)
    print("Model + lexicon saved to:", ARTIFACT_PATH)

    # Also update the standalone lexicon artifact
    joblib.dump({
        "global_lexicon": global_lexicon,
        "global_counts": global_counts,
        "global_mean": global_mean,
    }, LEXICON_ARTIFACT_PATH)
    print("Lexicon artifact updated:", LEXICON_ARTIFACT_PATH)