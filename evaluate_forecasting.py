import pandas as pd

from forecasting import train_multi_horizon_forecast, HORIZONS
from utils.experiment_logger import log_experiment


DATASET_VERSION = "synthetic_dataset_v1"
DATA_PATH = f"data/{DATASET_VERSION}.csv"


def evaluate_forecasting():

    df = pd.read_csv(DATA_PATH)

    # enforce ordering safety
    df = df.sort_values(["user_id", "entry_index"]).reset_index(drop=True)

    results = train_multi_horizon_forecast(df)

    for horizon in HORIZONS:

        model_mae = results[horizon]["model_mae"]
        baseline_mae = results[horizon]["baseline_mae"]

        print(f"\nHorizon: {horizon}")
        print("Model MAE:", model_mae)
        print("Baseline MAE:", baseline_mae)

        hyperparameters = {
            "features": [
                "lag_1",
                "lag_2",
                "rolling_3",
                "rolling_7",
                "rolling_14",
                "sin_time",
                "cos_time"
            ],
            "train_ratio": 0.8,
            "model": "LinearRegression"
        }

        metrics = {
            "MAE": model_mae
        }

        baseline_metrics = {
            "MAE_baseline": baseline_mae
        }

        log_experiment(
            dataset_version=DATASET_VERSION,
            experiment_type="forecasting",
            horizon=horizon,
            model_name="linear_regression",
            hyperparameters=hyperparameters,
            metrics=metrics,
            baseline_metrics=baseline_metrics,
            train_size=None,  # optional: you can compute this if needed
            test_size=None,
            notes="Forecasting benchmark on frozen dataset v1"
        )


if __name__ == "__main__":
    evaluate_forecasting()