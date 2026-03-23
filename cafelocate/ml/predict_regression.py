from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
MODELS_DIR = BASE_DIR / "models"
DEFAULT_INPUT = PROJECT_DIR / "data" / "processed" / "preprocessed_dataset.csv"

SUITES = {
    "pseudo": {
        "feature_path": MODELS_DIR / "feature_columns_regression.pkl",
        "scaler_path": MODELS_DIR / "scaler_regression.pkl",
        "models": {
            "80_20": {
                "rf": MODELS_DIR / "rf_regressor_v3_80_20.pkl",
                "xgb": MODELS_DIR / "xgb_regressor_v3_80_20.pkl",
            },
            "85_15": {
                "rf": MODELS_DIR / "rf_regressor_v3_85_15.pkl",
                "xgb": MODELS_DIR / "xgb_regressor_v3_85_15.pkl",
            },
        },
    },
    "observed": {
        "feature_path": MODELS_DIR / "feature_columns_regression_observed.pkl",
        "scaler_path": MODELS_DIR / "scaler_regression_observed.pkl",
        "models": {
            "80_20": {
                "rf": MODELS_DIR / "rf_regressor_observed_v1_80_20.pkl",
                "xgb": MODELS_DIR / "xgb_regressor_observed_v1_80_20.pkl",
            },
            "85_15": {
                "rf": MODELS_DIR / "rf_regressor_observed_v1_85_15.pkl",
                "xgb": MODELS_DIR / "xgb_regressor_observed_v1_85_15.pkl",
            },
        },
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference using trained CafeLocate regression models.")
    parser.add_argument("--suite", choices=["pseudo", "observed"], default="pseudo")
    parser.add_argument("--split", choices=["80_20", "85_15"], default="85_15")
    parser.add_argument("--model", choices=["rf", "xgb", "ensemble"], default="ensemble")
    parser.add_argument("--input-csv", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-csv", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    suite = SUITES[args.suite]
    feature_columns = joblib.load(suite["feature_path"])
    scaler = joblib.load(suite["scaler_path"])
    model_paths = suite["models"][args.split]

    df = pd.read_csv(args.input_csv)
    for column in feature_columns:
        if column not in df.columns:
            raise ValueError(f"Missing required feature column: {column}")
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0.0)

    x_scaled = pd.DataFrame(scaler.transform(df[feature_columns]), columns=feature_columns)
    predictions = {}

    if args.model in {"rf", "ensemble"}:
        rf_model = joblib.load(model_paths["rf"])
        predictions["rf_prediction"] = np.clip(rf_model.predict(x_scaled), 0.0, 100.0)

    if args.model in {"xgb", "ensemble"}:
        xgb_model = joblib.load(model_paths["xgb"])
        predictions["xgb_prediction"] = np.clip(xgb_model.predict(x_scaled), 0.0, 100.0)

    if args.model == "ensemble":
        predictions["ensemble_prediction"] = (
            predictions["rf_prediction"] + predictions["xgb_prediction"]
        ) / 2.0

    output_df = df.copy()
    for key, value in predictions.items():
        output_df[key] = value

    output_path = Path(args.output_csv) if args.output_csv else Path(args.input_csv).with_name(
        f"{Path(args.input_csv).stem}_{args.suite}_{args.split}_{args.model}_predictions.csv"
    )
    output_df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"Predictions saved to: {output_path}")


if __name__ == "__main__":
    main()
