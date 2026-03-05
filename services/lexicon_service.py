import os
import joblib

from models.lexicon_model import predict_mood_from_text

ARTIFACT_PATH = "artifacts/global_lexicon.pkl"


class LexiconService:

    def __init__(self):

        if not os.path.exists(ARTIFACT_PATH):
            raise FileNotFoundError("Lexicon artifact missing")

        artifact = joblib.load(ARTIFACT_PATH)

        self.global_lexicon = artifact["global_lexicon"]
        self.global_counts = artifact["global_counts"]
        self.global_mean = artifact["global_mean"]

    def analyze_text(self, text):

        prediction, contributions = predict_mood_from_text(
            text,
            self.global_lexicon,
            self.global_counts,
            {},
            {},
            self.global_mean
        )

        return prediction, contributions