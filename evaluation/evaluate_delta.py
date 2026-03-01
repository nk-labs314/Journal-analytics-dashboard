import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from models.lexicon_model import (
    build_global_lexicon,
    build_user_lexicon,
    predict_mood_from_text
)
from utils.experiment_logger import log_experiment


DATASET_VERSION = "synthetic_dataset_v3"
DATA_PATH = f"data/{DATASET_VERSION}.csv"
HORIZON = 7


def build_delta_dataset(df, horizon=7, train_ratio=0.8):

    train_rows = []
    test_rows = []

    for user_id, user_df in df.groupby("user_id"):

        user_df = user_df.sort_values("entry_index").reset_index(drop=True)

        # Compute delta
        user_df["future_mood"] = user_df["mood_score"].shift(-horizon)
        user_df["delta"] = user_df["future_mood"] - user_df["mood_score"]

        # Drop last horizon rows
        user_df = user_df.dropna().reset_index(drop=True)

        split_index = int(len(user_df) * train_ratio)

        train_rows.append(user_df.iloc[:split_index])
        test_rows.append(user_df.iloc[split_index:])

    train_df = pd.concat(train_rows).reset_index(drop=True)
    test_df = pd.concat(test_rows).reset_index(drop=True)

    return train_df, test_df



def run_tfidf_delta(train_df, test_df):

    vectorizer = TfidfVectorizer(max_features=5000)

    X_train = vectorizer.fit_transform(train_df["text"])
    X_test = vectorizer.transform(test_df["text"])

    y_train = train_df["delta"]
    y_test = test_df["delta"]

    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    return mae, r2

def run_lexicon_delta(train_df, test_df):

    # --------------------------
    # Build global lexicon (train only)
    # --------------------------
    global_lexicon, global_counts, global_mean = build_global_lexicon(
        train_df,
        min_freq=5
    )

    train_preds = []
    train_deltas = []

    # --------------------------
    # Build per-user lexicons
    # --------------------------
    for user_id in train_df["user_id"].unique():

        user_train = train_df[train_df["user_id"] == user_id]
        user_test = test_df[test_df["user_id"] == user_id]

        user_lexicon, user_counts = build_user_lexicon(
            train_df,
            user_id,
            min_freq=3
        )

        # Training predictions
        for _, row in user_train.iterrows():

            pred_mood, _ = predict_mood_from_text(
                text=row["text"],
                global_lexicon=global_lexicon,
                global_counts=global_counts,
                user_lexicon=user_lexicon,
                user_counts=user_counts,
                global_mean=global_mean,
                k=10
            )

            train_preds.append(pred_mood - global_mean)
            train_deltas.append(row["delta"])

    # Fit simple linear regression beta
    train_preds = np.array(train_preds).reshape(-1, 1)
    train_deltas = np.array(train_deltas)

    from sklearn.linear_model import LinearRegression
    lr = LinearRegression()
    lr.fit(train_preds, train_deltas)

    # --------------------------
    # Test phase
    # --------------------------
    test_preds = []
    test_deltas = []

    for user_id in test_df["user_id"].unique():

        user_train = train_df[train_df["user_id"] == user_id]
        user_test = test_df[test_df["user_id"] == user_id]

        user_lexicon, user_counts = build_user_lexicon(
            train_df,
            user_id,
            min_freq=3
        )

        for _, row in user_test.iterrows():

            pred_mood, _ = predict_mood_from_text(
                text=row["text"],
                global_lexicon=global_lexicon,
                global_counts=global_counts,
                user_lexicon=user_lexicon,
                user_counts=user_counts,
                global_mean=global_mean,
                k=10
            )

            x = np.array([[pred_mood - global_mean]])
            delta_pred = lr.predict(x)[0]

            test_preds.append(delta_pred)
            test_deltas.append(row["delta"])

    test_preds = np.array(test_preds)
    test_deltas = np.array(test_deltas)

    mae = mean_absolute_error(test_deltas, test_preds)
    r2 = r2_score(test_deltas, test_preds)

    return mae, r2

def evaluate_delta():

    df = pd.read_csv(DATA_PATH)
    df = df.sort_values(["user_id", "entry_index"]).reset_index(drop=True)

    train_df, test_df = build_delta_dataset(df, HORIZON)
    print(test_df["delta"].describe())

    # Baseline
    baseline_preds = np.zeros(len(test_df))
    baseline_mae = mean_absolute_error(test_df["delta"], baseline_preds)

    # TF-IDF Model
    mae_model, r2_model = run_tfidf_delta(train_df, test_df)

    print("Delta MAE:", mae_model)
    print("Baseline MAE:", baseline_mae)
    print("Delta R2:", r2_model)

    log_experiment(
        dataset_version=DATASET_VERSION,
        experiment_type="delta",
        horizon=HORIZON,
        model_name="tfidf_ridge",
        hyperparameters={
            "vectorizer": "TFIDF",
            "max_features": 5000,
            "alpha": 1.0
        },
        metrics={
            "MAE": mae_model,
            "R2": r2_model
        },
        baseline_metrics={
            "MAE_baseline": baseline_mae
        },
        train_size=len(train_df),
        test_size=len(test_df),
        notes="Text-only delta prediction"
    )
    # --------------------
    # Lexicon Delta Model
    # --------------------
    mae_lex, r2_lex = run_lexicon_delta(train_df, test_df)

    print("Lexicon Delta MAE:", mae_lex)
    print("Lexicon Delta R2:", r2_lex)

    log_experiment(
        dataset_version=DATASET_VERSION,
        experiment_type="delta",
        horizon=HORIZON,
        model_name="lexicon_delta",
        hyperparameters={
            "global_min_freq": 5,
            "user_min_freq": 3,
            "k": 10
        },
        metrics={
            "MAE": mae_lex,
            "R2": r2_lex
        },
        baseline_metrics={
            "MAE_baseline": baseline_mae
        },
        train_size=len(train_df),
        test_size=len(test_df),
        notes="Lexicon-based delta prediction"
    )

if __name__ == "__main__":
    evaluate_delta()