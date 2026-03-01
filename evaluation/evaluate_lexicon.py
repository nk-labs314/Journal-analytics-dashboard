import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score

from models.lexicon_engine import (
    build_global_lexicon,
    build_user_lexicon,
    predict_mood_from_text
)

from utils.experiment_logger import log_experiment


DATASET_VERSION = "synthetic_dataset_v1"
DATA_PATH = f"data/{DATASET_VERSION}.csv"


def chronological_split(df, train_ratio=0.8):
    split_index = int(len(df) * train_ratio)
    train = df.iloc[:split_index]
    test = df.iloc[split_index:]
    return train, test


def evaluate():

    df = pd.read_csv(DATA_PATH)

    # enforce chronological order
    df = df.sort_values(["user_id", "entry_index"]).reset_index(drop=True)

    train_df, test_df = chronological_split(df)

    # ---------------------------
    # Build GLOBAL lexicon (train only)
    # ---------------------------
    global_lexicon, global_counts, global_mean = build_global_lexicon(
        train_df,
        min_freq=5
    )

    predictions = []
    actuals = []

    # ---------------------------
    # Loop per user
    # ---------------------------
    for user_id in test_df["user_id"].unique():

        user_train_df = train_df[train_df["user_id"] == user_id]
        user_test_df = test_df[test_df["user_id"] == user_id]

        # Build user lexicon (train only)
        user_lexicon, user_counts = build_user_lexicon(
            train_df,
            user_id,
            min_freq=3
        )

        for _, row in user_test_df.iterrows():

            pred, _ = predict_mood_from_text(
                text=row["text"],
                global_lexicon=global_lexicon,
                global_counts=global_counts,
                user_lexicon=user_lexicon,
                user_counts=user_counts,
                global_mean=global_mean,
                k=10
            )

            predictions.append(pred)
            actuals.append(row["mood_score"])

    predictions = np.array(predictions)
    actuals = np.array(actuals)

    # ---------------------------
    # Metrics
    # ---------------------------
    mae = mean_absolute_error(actuals, predictions)

    baseline_pred = train_df["mood_score"].mean()
    baseline_mae = mean_absolute_error(
        actuals,
        np.full(len(actuals), baseline_pred)
    )

    r2 = r2_score(actuals, predictions)
    corr = np.corrcoef(actuals, predictions)[0, 1]

    print("MAE:", mae)
    print("Baseline MAE:", baseline_mae)
    print("R2:", r2)
    print("Correlation:", corr)

    # ---------------------------
    # Log experiment
    # ---------------------------
    hyperparameters = {
        "global_min_freq": 5,
        "user_min_freq": 3,
        "k": 10,
        "train_ratio": 0.8
    }

    metrics = {
        "MAE": mae,
        "R2": r2,
        "correlation": corr
    }

    baseline_metrics = {
        "MAE_baseline": baseline_mae
    }

    log_experiment(
        dataset_version=DATASET_VERSION,
        experiment_type="lexicon",
        horizon=0,
        model_name="hybrid_lexicon",
        hyperparameters=hyperparameters,
        metrics=metrics,
        baseline_metrics=baseline_metrics,
        train_size=len(train_df),
        test_size=len(test_df),
        notes="First official run on frozen dataset v1"
    )


if __name__ == "__main__":
    evaluate()