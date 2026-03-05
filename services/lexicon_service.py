import os
import joblib
from models.lexicon_model import build_user_lexicon, predict_mood_from_text
import logging

logger = logging.getLogger(__name__)

FORECAST_ARTIFACT_PATH = os.path.join("artifacts", "ridge_multi_output.pkl")
LEXICON_ARTIFACT_PATH = os.path.join("artifacts", "global_lexicon.pkl")


class LexiconService:

    def __init__(self):
        # Try loading from forecast artifact first (single source of truth)
        if os.path.exists(FORECAST_ARTIFACT_PATH):
            artifact = joblib.load(FORECAST_ARTIFACT_PATH)
            if "global_lexicon" in artifact:
                self.global_lexicon = artifact["global_lexicon"]
                self.global_counts = artifact["global_counts"]
                self.global_mean = artifact["global_mean"]
                logger.info("Lexicon loaded from forecast artifact (single source of truth)")
                return

        # Fallback to standalone lexicon artifact
        if os.path.exists(LEXICON_ARTIFACT_PATH):
            artifact = joblib.load(LEXICON_ARTIFACT_PATH)
            self.global_lexicon = artifact["global_lexicon"]
            self.global_counts = artifact["global_counts"]
            self.global_mean = artifact["global_mean"]
            logger.warning("Lexicon loaded from standalone artifact (fallback)")
            return

        raise FileNotFoundError("No lexicon artifact found")

    def analyze_text(self, text, user_df=None, user_id=1):
        # Build user lexicon from their journal history if available
        if user_df is not None and not user_df.empty:
            # lexicon_model expects columns: user_id, text, mood_score
            user_df = user_df.rename(columns={"journal_entry": "text"})
            user_df["user_id"] = user_id
            user_lexicon, user_counts = build_user_lexicon(
                user_df, user_id=user_id, min_freq=2
            )
        else:
            user_lexicon, user_counts = {}, {}

        prediction, contributions = predict_mood_from_text(
            text,
            self.global_lexicon,
            self.global_counts,
            user_lexicon,
            user_counts,
            self.global_mean
        )

        return prediction, contributions