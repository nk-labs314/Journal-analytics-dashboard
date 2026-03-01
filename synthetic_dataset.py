# synthetic_text_dataset.py

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


# ===================================================
# DATASET VERSION
# ===================================================

DATASET_VERSION = "synthetic_dataset_v1"


# ===================================================
# Vocabulary Pools
# ===================================================

POS_STRONG = ["ecstatic", "thrilled", "confident", "energized", "inspired"]
POS_MILD = ["productive", "motivated", "focused", "grateful", "content"]
NEG_STRONG = ["depressed", "hopeless", "miserable", "overwhelmed", "worthless"]
NEG_MILD = ["tired", "stressed", "anxious", "irritated", "low"]
NEUTRAL = ["work", "meeting", "routine", "food", "weather", "family", "task"]

TEMPLATES = [
    "Today I felt {w1} and {w2}. It was mostly about {topic}.",
    "I have been feeling {w1}. The day involved {topic} and made me {w2}.",
    "Overall I felt {w1}, especially during {topic}. Still somewhat {w2}.",
    "My mood was {w1}. Things around {topic} made me feel {w2}."
]


# ===================================================
# Regime Sampler
# ===================================================

def sample_user_regime(rng):
    regime_type = rng.choice(["stable_high", "volatile", "low_mean", "cyclical"])

    if regime_type == "stable_high":
        regime = {
            "baseline": rng.uniform(7.0, 8.5),
            "phi": rng.uniform(0.85, 0.95),
            "volatility": 0.7,
            "amplitude": 0.5,
            "shock_prob": 0.01,
            "cycle_period": 120
        }

    elif regime_type == "volatile":
        regime = {
            "baseline": rng.uniform(5.5, 6.5),
            "phi": rng.uniform(0.6, 0.75),
            "volatility": 2.0,
            "amplitude": 1.0,
            "shock_prob": 0.05,
            "cycle_period": 60
        }

    elif regime_type == "low_mean":
        regime = {
            "baseline": rng.uniform(3.5, 5.0),
            "phi": rng.uniform(0.8, 0.9),
            "volatility": 1.2,
            "amplitude": 0.8,
            "shock_prob": 0.03,
            "cycle_period": 90
        }

    else:  # cyclical
        regime = {
            "baseline": rng.uniform(5.5, 7.0),
            "phi": rng.uniform(0.75, 0.9),
            "volatility": 1.0,
            "amplitude": 2.0,
            "shock_prob": 0.02,
            "cycle_period": rng.integers(40, 100)
        }

    return regime_type, regime


# ===================================================
# Mood Generator
# ===================================================

def generate_mood_series(n_entries, regime, rng):
    moods = []
    mood = regime["baseline"]

    for t in range(n_entries):

        seasonal = regime["amplitude"] * np.sin(
            2 * np.pi * t / regime["cycle_period"]
        )

        shock = 0
        if rng.random() < regime["shock_prob"]:
            shock = rng.uniform(-3, 3)

        noise = rng.normal(0, regime["volatility"])

        mood = (
            regime["baseline"]
            + regime["phi"] * (mood - regime["baseline"])
            + seasonal
            + noise
            + shock
        )

        mood = np.clip(mood, 1, 10)
        moods.append(mood)

    return moods


# ===================================================
# Text Emission Model
# ===================================================

def sample_words_from_mood(mood, rng):

    if mood <= 3:
        w1 = rng.choice(NEG_STRONG)
        w2 = rng.choice(NEG_MILD)

    elif mood <= 5:
        w1 = rng.choice(NEG_MILD)
        w2 = rng.choice(NEUTRAL)

    elif mood <= 7:
        w1 = rng.choice(POS_MILD)
        w2 = rng.choice(NEUTRAL)

    else:
        w1 = rng.choice(POS_STRONG)
        w2 = rng.choice(POS_MILD)

    topic = rng.choice(NEUTRAL)
    template = rng.choice(TEMPLATES)

    return template.format(w1=w1, w2=w2, topic=topic)


# ===================================================
# Full Dataset Generator
# ===================================================

def generate_synthetic_text_dataset(
    n_users=5,
    n_entries=1000,
    random_state=42
):
    rng = np.random.default_rng(random_state)

    rows = []
    start_date = datetime(2020, 1, 1)

    for user_id in range(1, n_users + 1):

        regime_label, regime = sample_user_regime(rng)
        moods = generate_mood_series(n_entries, regime, rng)

        for i, mood in enumerate(moods):

            text = sample_words_from_mood(mood, rng)

            rows.append({
                "user_id": user_id,
                "entry_index": i,
                "date": start_date + timedelta(days=i),
                "mood_score": mood,
                "text": text,
                "regime_label": regime_label
            })

    df = pd.DataFrame(rows)

    # Enforce strict chronological ordering
    df = df.sort_values(["user_id", "entry_index"]).reset_index(drop=True)

    return df


# ===================================================
# FREEZE DATASET TO CSV
# ===================================================

if __name__ == "__main__":

    os.makedirs("data", exist_ok=True)

    df = generate_synthetic_text_dataset(
        n_users=10,
        n_entries=500,
        random_state=42
    )

    output_path = f"data/{DATASET_VERSION}.csv"
    df.to_csv(output_path, index=False)

    print(f"[INFO] Dataset version: {DATASET_VERSION}")
    print(f"[INFO] Saved to: {output_path}")
    print(f"[INFO] Rows: {len(df)}")
    print(f"[INFO] Users: {df['user_id'].nunique()}")