import os
import joblib
from models.lexicon_model import build_user_lexicon, predict_mood_from_text
import logging

logger = logging.getLogger(__name__)
ARTIFACT_PATH = "artifacts/global_lexicon.pkl"


class LexiconService:

    def __init__(self):
        if not os.path.exists(ARTIFACT_PATH):
            raise FileNotFoundError("Lexicon artifact missing")

        artifact = joblib.load(ARTIFACT_PATH)
        self.global_lexicon = artifact["global_lexicon"]
        self.global_counts = artifact["global_counts"]
        self.global_mean = artifact["global_mean"]

    def analyze_text(self, text, user_df=None):
        # Build user lexicon from their journal history if available
        if user_df is not None and not user_df.empty:
            # lexicon_model expects columns: user_id, text, mood_score
            user_df = user_df.rename(columns={"journal_entry": "text"})
            user_df["user_id"] = 1  # will be replaced once auth exists
            user_lexicon, user_counts = build_user_lexicon(
                user_df, user_id=1, min_freq=2  # lower threshold for real users with fewer entries
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