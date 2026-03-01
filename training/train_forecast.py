import os
import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge

from models.feature_builder import build_features, FEATURE_COLUMNS


HORIZONS = [3, 7, 14]
ARTIFACT_PATH = os.path.join("artifacts", "ridge_multi_output.pkl")


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


def train_model(df):
    df = df.sort_values(["user_id", "entry_index"])

    train_rows = []

    for user_id, user_df in df.groupby("user_id"):
        user_df = build_features(user_df)
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
    # Use synthetic dataset for now
    df = pd.read_csv("data/synthetic_dataset_v3.csv")

    model = train_model(df)

    os.makedirs("artifacts", exist_ok=True)

    joblib.dump({
        "model": model,
        "feature_columns": FEATURE_COLUMNS,
        "horizons": HORIZONS
    }, ARTIFACT_PATH)

    print("Model saved to:", ARTIFACT_PATH)