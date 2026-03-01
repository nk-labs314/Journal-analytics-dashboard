# data_generator.py

import numpy as np
import pandas as pd


def generate_mood_series(
    n_entries=3000,
    baseline=6.5,
    mean_reversion=0.08,
    low_vol=0.3,
    high_vol=0.8,
    regime_switch_prob=0.05,
    cycle_amplitude=1.0,
    cycle_period=60,
    shock_prob=0.01,
    shock_magnitude=2.5,
    random_seed=42
):
    """
    Generates synthetic behavioral mood time-series with:
    - AR(1) mean reversion
    - Volatility regime switching
    - Slow sinusoidal baseline drift
    - Rare shock events
    """

    np.random.seed(random_seed)

    moods = []
    current_mood = baseline
    current_vol = low_vol

    for t in range(n_entries):

        # ----- Cyclical Baseline -----
        dynamic_baseline = baseline + cycle_amplitude * np.sin(
            2 * np.pi * t / cycle_period
        )

        # ----- Volatility Regime Switching -----
        if np.random.rand() < regime_switch_prob:
            current_vol = high_vol if current_vol == low_vol else low_vol

        # ----- Gaussian Noise -----
        noise = np.random.normal(0, current_vol)

        # ----- Mean Reversion Toward Dynamic Baseline -----
        reversion = mean_reversion * (dynamic_baseline - current_mood)

        current_mood = current_mood + noise + reversion

        # ----- Shock Events -----
        if np.random.rand() < shock_prob:
            shock_direction = -1 if np.random.rand() < 0.8 else 1
            current_mood += shock_direction * shock_magnitude

        # ----- Bound Between 1 and 10 -----
        current_mood = np.clip(current_mood, 1, 10)

        moods.append(current_mood)

    df = pd.DataFrame({
        "entry_index": np.arange(n_entries),
        "mood_score": moods
    })
    
    return df
