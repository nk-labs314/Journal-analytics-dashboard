import os
import joblib
from models.lexicon_model import build_user_lexicon, predict_mood_from_text
import logging
import hashlib

FORECAST_HASH = os.getenv("MODEL_SHA256_FORECAST")
LEXICON_HASH = os.getenv("MODEL_SHA256_LEXICON")
logger = logging.getLogger(__name__)

FORECAST_ARTIFACT_PATH = os.path.join("artifacts", "ridge_multi_output.pkl")
LEXICON_ARTIFACT_PATH = os.path.join("artifacts", "global_lexicon.pkl")


class LexiconService:
    def _verify(self, path, expected_hash):
        if not expected_hash:
            raise RuntimeError("Model hash not set")
        expected_hash =expected_hash.strip()
        import hashlib
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)

        if h.hexdigest() != expected_hash:
            raise RuntimeError(f"Model tampered: {path}")

    def __init__(self):
    # Try forecast artifact first
        if os.path.exists(FORECAST_ARTIFACT_PATH):
            self._verify(FORECAST_ARTIFACT_PATH, os.getenv("MODEL_SHA256_FORECAST"))
            artifact = joblib.load(FORECAST_ARTIFACT_PATH)

            if "global_lexicon" in artifact:
                self.global_lexicon = artifact["global_lexicon"]
                self.global_counts = artifact["global_counts"]
                self.global_mean = artifact["global_mean"]
                logger.info("Lexicon loaded from forecast artifact")
                return

        # Fallback to standalone lexicon
        if os.path.exists(LEXICON_ARTIFACT_PATH):
            self._verify(LEXICON_ARTIFACT_PATH,os.getenv("MODEL_SHA256_LEXICON"))
            artifact = joblib.load(LEXICON_ARTIFACT_PATH)

            self.global_lexicon = artifact["global_lexicon"]
            self.global_counts = artifact["global_counts"]
            self.global_mean = artifact["global_mean"]
            logger.warning("Lexicon loaded from standalone artifact")
            return

        raise FileNotFoundError("No lexicon artifact found")

    def analyze_text(self, text, user_df=None, user_id=1):
        # Build user lexicon from their journal history if available
        if user_df is not None and not user_df.empty and len(user_df) >= 10:
            user_df = user_df.copy()
            if "journal_entry" in user_df.columns and "text" not in user_df.columns:
                user_df = user_df.rename(columns={"journal_entry": "text"})
            user_df["user_id"] = user_id
            user_lexicon, user_counts = build_user_lexicon(
                user_df, user_id=user_id, min_freq=3
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