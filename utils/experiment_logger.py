import os
import json
import uuid
import pandas as pd
from datetime import datetime


LOG_PATH = "experiments_log.csv"


def log_experiment(
    dataset_version: str,
    experiment_type: str,
    horizon: int,
    model_name: str,
    hyperparameters: dict,
    metrics: dict,
    baseline_metrics: dict,
    train_size: int,
    test_size: int,
    notes: str = ""
):

    exp_id = str(uuid.uuid4())[:8]
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    row = {
        "exp_id": exp_id,
        "timestamp": timestamp,
        "dataset_version": dataset_version,
        "experiment_type": experiment_type,
        "horizon": horizon,
        "model_name": model_name,
        "hyperparameters": json.dumps(hyperparameters),
        "metrics": json.dumps(metrics),
        "baseline_metrics": json.dumps(baseline_metrics),
        "train_size": train_size,
        "test_size": test_size,
        "notes": notes,
    }

    df_row = pd.DataFrame([row])

    if not os.path.exists(LOG_PATH):
        df_row.to_csv(LOG_PATH, index=False)
    else:
        df_row.to_csv(LOG_PATH, mode="a", header=False, index=False)

    print(f"[LOGGED] Experiment {exp_id} saved.")