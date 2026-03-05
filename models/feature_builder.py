import numpy as np
import pandas as pd
from models.lexicon_model import predict_mood_from_text


FEATURE_COLUMNS = [
    "lag_1",
    "lag_2",
    "rolling_3",
    "rolling_7",
    "rolling_14",
    "sin_time",
    "cos_time",
    "lexicon_score"
]


def _score_text(text, global_lexicon, global_counts, global_mean):
    """Score a single text using the global lexicon. Returns the centered prediction."""
    if not text or not isinstance(text, str):
        return 0.0
    prediction, _ = predict_mood_from_text(
        text, global_lexicon, global_counts, {}, {}, global_mean
    )
    return prediction - global_mean  # centered score


def build_features(df, cycle_period=60, lexicon=None):
    """Build features for the forecasting model.
    
    Args:
        df: DataFrame with mood_score, entry_index, and optionally text/journal_entry columns
        cycle_period: period for cyclical time encoding
        lexicon: dict with global_lexicon, global_counts, global_mean (optional)
    """
    df = df.copy()

    df["lag_1"] = df["mood_score"].shift(1)
    df["lag_2"] = df["mood_score"].shift(2)

    df["rolling_3"] = df["mood_score"].rolling(window=3).mean()
    df["rolling_7"] = df["mood_score"].rolling(window=7).mean()
    df["rolling_14"] = df["mood_score"].rolling(window=14).mean()

    df["sin_time"] = np.sin(2 * np.pi * df["entry_index"] / cycle_period)
    df["cos_time"] = np.cos(2 * np.pi * df["entry_index"] / cycle_period)

    # Lexicon score feature
    if lexicon is not None:
        text_col = "text" if "text" in df.columns else "journal_entry"
        if text_col in df.columns:
            df["lexicon_score"] = df[text_col].apply(
                lambda t: _score_text(
                    t,
                    lexicon["global_lexicon"],
                    lexicon["global_counts"],
                    lexicon["global_mean"]
                )
            )
        else:
            df["lexicon_score"] = 0.0
    else:
        df["lexicon_score"] = 0.0

    return df