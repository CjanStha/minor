from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
SPLITS_DIR = PROJECT_DIR / "data" / "processed" / "splits"
MODELS_DIR = BASE_DIR / "models"
METRICS_PATH = MODELS_DIR / "regression_metrics.json"
FEATURES_PATH = MODELS_DIR / "feature_columns_regression.pkl"
SCALER_PATH = MODELS_DIR / "scaler_regression.pkl"

FEATURE_COLUMNS = [
    "competitors_within_500m",
    "competitors_within_200m",
    "competitors_min_distance",
    "competitors_avg_distance",
    "roads_within_500m",
    "roads_avg_distance",
    "schools_within_500m",
    "schools_within_200m",
    "schools_min_distance",
    "hospitals_within_500m",
    "hospitals_min_distance",
    "bus_stops_within_500m",
    "bus_stops_min_distance",
    "population_density_proxy",
    "accessibility_score",
    "foot_traffic_score",
    "competition_pressure",
]

SPLIT_CONFIGS = {
    "80_20": {
        "train_file": SPLITS_DIR / "preprocessed_train_80.csv",
        "test_file": SPLITS_DIR / "preprocessed_test_20.csv",
        "rf_model": MODELS_DIR / "rf_regressor_v3_80_20.pkl",
        "xgb_model": MODELS_DIR / "xgb_regressor_v3_80_20.pkl",
        "is_default": False,
    },
    "85_15": {
        "train_file": SPLITS_DIR / "preprocessed_train_85.csv",
        "test_file": SPLITS_DIR / "preprocessed_test_15.csv",
        "rf_model": MODELS_DIR / "rf_regressor_v3_85_15.pkl",
        "xgb_model": MODELS_DIR / "xgb_regressor_v3_85_15.pkl",
        "is_default": True,
    },
}


def derive_regression_target(df: pd.DataFrame) -> pd.Series:
    population_component = np.minimum(100.0, df["population_density_proxy"].astype(float) * 10.0) * 0.20
    accessibility_component = np.minimum(10.0, df["accessibility_score"].astype(float)) * 10.0 * 0.15
    foot_traffic_component = np.minimum(10.0, df["foot_traffic_score"].astype(float)) * 10.0 * 0.15
    schools_component = np.minimum(20.0, df["schools_within_500m"].astype(float) * 5.0) * 0.10
    bus_component = np.minimum(20.0, df["bus_stops_within_500m"].astype(float) * 5.0) * 0.10
    competition_penalty = np.minimum(100.0, df["competition_pressure"].astype(float) * 10.0) * 0.20
    competitor_penalty = np.minimum(100.0, df["competitors_within_200m"].astype(float) * 10.0) * 0.10

    score = (
        population_component
        + accessibility_component
        + foot_traffic_component
        + schools_component
        + bus_component
        - competition_penalty
        - competitor_penalty
    )
    return np.clip(score, 0.0, 100.0).round(4)


def load_split_frame(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Split file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    missing_columns = [column for column in FEATURE_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required feature columns in {csv_path.name}: {missing_columns}")

    for column in FEATURE_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0.0)

    df["derived_suitability_score"] = derive_regression_target(df)
    return df


def build_models() -> tuple[RandomForestRegressor, XGBRegressor]:
    rf_model = RandomForestRegressor(
        n_estimators=300,
        max_depth=14,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=1,
    )
    xgb_model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.0,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=1,
    )
    return rf_model, xgb_model


def evaluate_regressor(model, x_test: pd.DataFrame, y_test: pd.Series) -> dict:
    predictions = np.clip(model.predict(x_test), 0.0, 100.0)
    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions) ** 0.5
    r2 = r2_score(y_test, predictions)
    return {
        "mae": round(float(mae), 4),
        "rmse": round(float(rmse), 4),
        "r2": round(float(r2), 4),
        "prediction_min": round(float(np.min(predictions)), 4),
        "prediction_max": round(float(np.max(predictions)), 4),
    }


def train_for_split(split_name: str, config: dict) -> dict:
    train_df = load_split_frame(config["train_file"])
    test_df = load_split_frame(config["test_file"])

    x_train = train_df[FEATURE_COLUMNS].copy()
    x_test = test_df[FEATURE_COLUMNS].copy()
    y_train = train_df["derived_suitability_score"].copy()
    y_test = test_df["derived_suitability_score"].copy()

    scaler = StandardScaler()
    x_train_scaled = pd.DataFrame(scaler.fit_transform(x_train), columns=FEATURE_COLUMNS)
    x_test_scaled = pd.DataFrame(scaler.transform(x_test), columns=FEATURE_COLUMNS)

    rf_model, xgb_model = build_models()
    rf_model.fit(x_train_scaled, y_train)
    xgb_model.fit(x_train_scaled, y_train)

    rf_metrics = evaluate_regressor(rf_model, x_test_scaled, y_test)
    xgb_metrics = evaluate_regressor(xgb_model, x_test_scaled, y_test)

    rf_path = config["rf_model"]
    xgb_path = config["xgb_model"]
    joblib.dump(rf_model, rf_path)
    joblib.dump(xgb_model, xgb_path)

    if config["is_default"]:
        joblib.dump(scaler, SCALER_PATH)
        joblib.dump(FEATURE_COLUMNS, FEATURES_PATH)

    return {
        "split_name": split_name,
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "target_name": "derived_suitability_score",
        "target_description": "Pseudo-label derived from backend fallback suitability formula.",
        "rf_model_path": str(rf_path),
        "xgb_model_path": str(xgb_path),
        "random_forest": rf_metrics,
        "xgboost": xgb_metrics,
        "default_backend_artifacts": config["is_default"],
    }


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    metrics_report = {
        "feature_columns": FEATURE_COLUMNS,
        "default_scaler_path": str(SCALER_PATH),
        "default_feature_columns_path": str(FEATURES_PATH),
        "training_runs": {},
    }

    for split_name, config in SPLIT_CONFIGS.items():
        metrics_report["training_runs"][split_name] = train_for_split(split_name, config)

    METRICS_PATH.write_text(json.dumps(metrics_report, indent=2), encoding="utf-8")

    print(f"Saved metrics report to: {METRICS_PATH}")
    for split_name, result in metrics_report["training_runs"].items():
        print(
            f"{split_name}: "
            f"RF r2={result['random_forest']['r2']} rmse={result['random_forest']['rmse']} | "
            f"XGB r2={result['xgboost']['r2']} rmse={result['xgboost']['rmse']}"
        )


if __name__ == "__main__":
    main()
