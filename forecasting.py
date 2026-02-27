# forecasting.py

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error


HORIZONS = [3, 7, 14]


# --------------------------------------------------
# 1️⃣ Feature Construction
# --------------------------------------------------

def build_features(df):
    df = df.copy()

    # Lag features
    df["lag_1"] = df["mood_score"].shift(1)
    df["lag_2"] = df["mood_score"].shift(2)

    # Rolling 7 mean (uses only past values)
    df["rolling_7"] = df["mood_score"].rolling(window=7).mean()

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

    df = build_features(df)

    results = {}

    for horizon in HORIZONS:

        df_h = build_target(df, horizon)

        # Drop rows with NaNs (lag + rolling + future target)
        df_h = df_h.dropna().reset_index(drop=True)

        feature_cols = ["lag_1", "lag_2", "rolling_7"]
        target_col = f"target_{horizon}"

        X = df_h[feature_cols]
        y = df_h[target_col]

        # Chronological split
        split_index = int(len(df_h) * train_ratio)

        X_train = X.iloc[:split_index]
        y_train = y.iloc[:split_index]

        X_test = X.iloc[split_index:]
        y_test = y.iloc[split_index:]

        # Model
        model = LinearRegression()
        model.fit(X_train, y_train)

        preds = model.predict(X_test)

        mae_model = mean_absolute_error(y_test, preds)

        # Baseline = current rolling_7
        baseline_preds = X_test["rolling_7"]
        mae_baseline = mean_absolute_error(y_test, baseline_preds)

        results[horizon] = {
            "model_mae": mae_model,
            "baseline_mae": mae_baseline,
            "model": model
        }

    return results