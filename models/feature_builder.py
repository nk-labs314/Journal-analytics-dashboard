import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "lag_1",
    "lag_2",
    "rolling_3",
    "rolling_7",
    "rolling_14",
    "sin_time",
    "cos_time"
]


def build_features(df, cycle_period=60):
    df = df.copy()

    df["lag_1"] = df["mood_score"].shift(1)
    df["lag_2"] = df["mood_score"].shift(2)

    df["rolling_3"] = df["mood_score"].rolling(window=3).mean()
    df["rolling_7"] = df["mood_score"].rolling(window=7).mean()
    df["rolling_14"] = df["mood_score"].rolling(window=14).mean()

    df["sin_time"] = np.sin(2 * np.pi * df["entry_index"] / cycle_period)
    df["cos_time"] = np.cos(2 * np.pi * df["entry_index"] / cycle_period)

    return df