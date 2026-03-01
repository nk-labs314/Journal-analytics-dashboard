import os
import joblib
import pandas as pd

from models.feature_builder import build_features

ARTIFACT_PATH = os.path.join("artifacts", "ridge_multi_output.pkl")


class ForecastService:

    def __init__(self):
        self.model = None
        self.feature_columns = None
        self.horizons = None
        self.load_model()

    def load_model(self):
        if not os.path.exists(ARTIFACT_PATH):
            raise FileNotFoundError("Forecast model artifact not found.")

        artifact = joblib.load(ARTIFACT_PATH)
        self.model = artifact["model"]
        self.feature_columns = artifact["feature_columns"]
        self.horizons = artifact["horizons"]

    def predict(self, user_df):

        if user_df.empty:
            return None

        # Sort chronologically using DB timestamp
        user_df = user_df.sort_values("date").reset_index(drop=True)

        # Generate sequential entry_index dynamically
        user_df["entry_index"] = range(len(user_df))

        user_df = build_features(user_df)
        user_df = user_df.dropna()

        if user_df.empty:
            return None

        latest_row = user_df.iloc[-1]
        X = latest_row[self.feature_columns].values.reshape(1, -1)

        preds = self.model.predict(X)[0]

        return {
            str(h): float(pred)
            for h, pred in zip(self.horizons, preds)
    }