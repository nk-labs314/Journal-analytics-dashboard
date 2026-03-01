# forecasting.py

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error


HORIZONS = [3, 7, 14]


# --------------------------------------------------
# 1️⃣ Feature Construction
# --------------------------------------------------

def build_features(df, cycle_period=60):
    df = df.copy()

    # ----- Lag Features -----
    df["lag_1"] = df["mood_score"].shift(1)
    df["lag_2"] = df["mood_score"].shift(2)

    # ----- Rolling Features -----
    df["rolling_3"] = df["mood_score"].rolling(window=3).mean()
    df["rolling_7"] = df["mood_score"].rolling(window=7).mean()
    df["rolling_14"] = df["mood_score"].rolling(window=14).mean()

    # ----- Seasonal (Time-Based) Features -----
    df["sin_time"] = np.sin(2 * np.pi * df["entry_index"] / cycle_period)
    df["cos_time"] = np.cos(2 * np.pi * df["entry_index"] / cycle_period)

    return df

# --------------------------------------------------
# 2️⃣ Target Construction
# --------------------------------------------------

def build_target(df, horizon):
    df = df.copy()

    df[f"target_{horizon}"] = (
        df["mood_score"]
        .shift(-1)
        .rolling(window=horizon)
        .mean()
    )

    return df


# --------------------------------------------------
# 3️⃣ Train + Evaluate Per Horizon
# --------------------------------------------------

def train_multi_horizon_forecast(df, train_ratio=0.8):

    results = {}

    for horizon in HORIZONS:

        train_rows = []
        test_rows = []

        # Process each user independently
        for user_id, user_df in df.groupby("user_id"):

            user_df = user_df.sort_values("entry_index").reset_index(drop=True)

            user_df = build_features(user_df)
            user_df = build_target(user_df, horizon)

            user_df = user_df.dropna().reset_index(drop=True)

            split_index = int(len(user_df) * train_ratio)

            train_rows.append(user_df.iloc[:split_index])
            test_rows.append(user_df.iloc[split_index:])

        train_df = pd.concat(train_rows)
        test_df = pd.concat(test_rows)

        feature_cols = [
            "lag_1",
            "lag_2",
            "rolling_3",
            "rolling_7",
            "rolling_14",
            "sin_time",
            "cos_time"
        ]

        target_col = f"target_{horizon}"

        X_train = train_df[feature_cols]
        y_train = train_df[target_col]

        X_test = test_df[feature_cols]
        y_test = test_df[target_col]

        model = LinearRegression()
        model.fit(X_train, y_train)

        preds = model.predict(X_test)

        mae_model = mean_absolute_error(y_test, preds)

        baseline_preds = X_test["rolling_7"]
        mae_baseline = mean_absolute_error(y_test, baseline_preds)

        results[horizon] = {
            "model_mae": mae_model,
            "baseline_mae": mae_baseline,
            "model": model
        }

    return results