import os
import joblib
import pandas as pd

from models.lexicon_model import build_global_lexicon

DATA_PATH = "data/synthetic_dataset_v3.csv"
ARTIFACT_PATH = "artifacts/global_lexicon.pkl"


def train_lexicon():

    df = pd.read_csv(DATA_PATH)

    global_lexicon, global_counts, global_mean = build_global_lexicon(df)

    artifact = {
        "global_lexicon": global_lexicon,
        "global_counts": global_counts,
        "global_mean": global_mean
    }

    os.makedirs("artifacts", exist_ok=True)

    joblib.dump(artifact, ARTIFACT_PATH)

    print("Lexicon artifact saved:", ARTIFACT_PATH)


if __name__ == "__main__":
    train_lexicon()