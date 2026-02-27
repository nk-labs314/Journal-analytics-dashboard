# data_generator.py

import numpy as np
import pandas as pd


def generate_mood_series(
    n_entries=1000,
    baseline=6.5,
    mean_reversion=0.08,
    low_vol=0.3,
    high_vol=0.8,
    regime_switch_prob=0.05,
    random_seed=42
):
    """
    Generates a synthetic behavioral mood time-series using:
    - Mean reversion toward a fixed baseline
    - Gaussian noise
    - Volatility regime switching

    Returns:
        DataFrame with:
            entry_index
            mood_score
    """

    np.random.seed(random_seed)

    moods = []
    current_mood = baseline
    current_vol = low_vol

    for t in range(n_entries):

        # ---- Volatility Regime Switching ----
        if np.random.rand() < regime_switch_prob:
            current_vol = high_vol if current_vol == low_vol else low_vol

        # ---- Gaussian Noise ----
        noise = np.random.normal(0, current_vol)

        # ---- Mean Reversion ----
        reversion = mean_reversion * (baseline - current_mood)

        # ---- Update Mood ----
        current_mood = current_mood + noise + reversion

        # ---- Bound Between 1 and 10 ----
        current_mood = np.clip(current_mood, 1, 10)

        moods.append(current_mood)

    df = pd.DataFrame({
        "entry_index": np.arange(n_entries),
        "mood_score": moods
    })

    return df