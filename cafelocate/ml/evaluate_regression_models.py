from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
SPLITS_DIR = PROJECT_DIR / "data" / "processed" / "splits"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

PSEUDO_FEATURES_PATH = MODELS_DIR / "feature_columns_regression.pkl"
PSEUDO_SCALER_PATH = MODELS_DIR / "scaler_regression.pkl"
OBSERVED_FEATURES_PATH = MODELS_DIR / "feature_columns_regression_observed.pkl"
OBSERVED_SCALER_PATH = MODELS_DIR / "scaler_regression_observed.pkl"


def derive_pseudo_target(df: pd.DataFrame) -> pd.Series:
    population_component = np.minimum(100.0, df["population_density_proxy"].astype(float) * 10.0) * 0.20
    accessibility_component = np.minimum(10.0, df["accessibility_score"].astype(float)) * 10.0 * 0.15
    foot_traffic_component = np.minimum(10.0, df["foot_traffic_score"].astype(float)) * 10.0 * 0.15
    schools_component = np.minimum(20.0, df["schools_within_500m"].astype(float) * 5.0) * 0.10
    bus_component = np.minimum(20.0, df["bus_stops_within_500m"].astype(float) * 5.0) * 0.10
    competition_penalty = np.minimum(100.0, df["competition_pressure"].astype(float) * 10.0) * 0.20
    competitor_penalty = np.minimum(100.0, df["competitors_within_200m"].astype(float) * 10.0) * 0.10
    return np.clip(
        population_component
        + accessibility_component
        + foot_traffic_component
        + schools_component
        + bus_component
        - competition_penalty
        - competitor_penalty,
        0.0,
        100.0,
    )


SUITES = {
    "pseudo": {
        "feature_path": PSEUDO_FEATURES_PATH,
        "scaler_path": PSEUDO_SCALER_PATH,
        "splits": {
            "80_20": {
                "test_file": SPLITS_DIR / "preprocessed_test_20.csv",
                "target_kind": "pseudo",
                "rf_model": MODELS_DIR / "rf_regressor_v3_80_20.pkl",
                "xgb_model": MODELS_DIR / "xgb_regressor_v3_80_20.pkl",
            },
            "85_15": {
                "test_file": SPLITS_DIR / "preprocessed_test_15.csv",
                "target_kind": "pseudo",
                "rf_model": MODELS_DIR / "rf_regressor_v3_85_15.pkl",
                "xgb_model": MODELS_DIR / "xgb_regressor_v3_85_15.pkl",
            },
        },
    },
    "observed": {
        "feature_path": OBSERVED_FEATURES_PATH,
        "scaler_path": OBSERVED_SCALER_PATH,
        "splits": {
            "80_20": {
                "test_file": SPLITS_DIR / "labeled_test_20.csv",
                "target_kind": "observed",
                "rf_model": MODELS_DIR / "rf_regressor_observed_v1_80_20.pkl",
                "xgb_model": MODELS_DIR / "xgb_regressor_observed_v1_80_20.pkl",
            },
            "85_15": {
                "test_file": SPLITS_DIR / "labeled_test_15.csv",
                "target_kind": "observed",
                "rf_model": MODELS_DIR / "rf_regressor_observed_v1_85_15.pkl",
                "xgb_model": MODELS_DIR / "xgb_regressor_observed_v1_85_15.pkl",
            },
        },
    },
}


def ensure_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for column in columns:
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0.0)
    return df


def get_target(df: pd.DataFrame, target_kind: str) -> pd.Series:
    if target_kind == "pseudo":
        return derive_pseudo_target(df)
    return pd.to_numeric(df["observed_outcome_score"], errors="coerce").fillna(0.0)


def save_feature_importance(feature_names: list[str], model, out_prefix: Path) -> None:
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        return

    importance_df = pd.DataFrame(
        {"feature": feature_names, "importance": np.asarray(importances, dtype=float)}
    ).sort_values("importance", ascending=False)
    importance_df.to_csv(out_prefix.with_suffix(".csv"), index=False, encoding="utf-8")

    plt.figure(figsize=(10, 6))
    plt.barh(importance_df["feature"], importance_df["importance"], color="#2f6b5f")
    plt.gca().invert_yaxis()
    plt.title(out_prefix.stem.replace("_", " ").title())
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(out_prefix.with_suffix(".png"), dpi=160)
    plt.close()


def save_scatter_plot(actual: np.ndarray, predicted: np.ndarray, out_path: Path, title: str) -> None:
    plt.figure(figsize=(6, 6))
    plt.scatter(actual, predicted, alpha=0.65, color="#0f766e", edgecolors="none")
    min_val = float(min(actual.min(), predicted.min()))
    max_val = float(max(actual.max(), predicted.max()))
    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--", color="#b45309")
    plt.title(title)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    summary = {}

    for suite_name, suite_config in SUITES.items():
        feature_columns = joblib.load(suite_config["feature_path"])
        scaler = joblib.load(suite_config["scaler_path"])
        summary[suite_name] = {}

        for split_name, split_config in suite_config["splits"].items():
            df = pd.read_csv(split_config["test_file"])
            df = ensure_numeric(df, feature_columns)
            y_true = get_target(df, split_config["target_kind"]).to_numpy(dtype=float)
            x_scaled = pd.DataFrame(scaler.transform(df[feature_columns]), columns=feature_columns)

            rf_model = joblib.load(split_config["rf_model"])
            xgb_model = joblib.load(split_config["xgb_model"])

            rf_pred = np.clip(rf_model.predict(x_scaled), 0.0, 100.0)
            xgb_pred = np.clip(xgb_model.predict(x_scaled), 0.0, 100.0)
            ensemble_pred = np.clip((rf_pred + xgb_pred) / 2.0, 0.0, 100.0)

            predictions_df = pd.DataFrame(
                {
                    "place_id": df["place_id"],
                    "name": df.get("name", pd.Series([""] * len(df))),
                    "actual_score": y_true,
                    "rf_prediction": rf_pred,
                    "xgb_prediction": xgb_pred,
                    "ensemble_prediction": ensemble_pred,
                    "rf_abs_error": np.abs(y_true - rf_pred),
                    "xgb_abs_error": np.abs(y_true - xgb_pred),
                    "ensemble_abs_error": np.abs(y_true - ensemble_pred),
                }
            )
            predictions_path = REPORTS_DIR / f"{suite_name}_{split_name}_predictions.csv"
            predictions_df.to_csv(predictions_path, index=False, encoding="utf-8")

            save_scatter_plot(
                y_true,
                ensemble_pred,
                REPORTS_DIR / f"{suite_name}_{split_name}_actual_vs_ensemble.png",
                f"{suite_name.upper()} {split_name} Actual vs Ensemble",
            )

            save_feature_importance(
                feature_columns,
                rf_model,
                REPORTS_DIR / f"{suite_name}_{split_name}_rf_feature_importance",
            )
            save_feature_importance(
                feature_columns,
                xgb_model,
                REPORTS_DIR / f"{suite_name}_{split_name}_xgb_feature_importance",
            )

            summary[suite_name][split_name] = {
                "prediction_file": str(predictions_path),
                "actual_vs_pred_plot": str(REPORTS_DIR / f"{suite_name}_{split_name}_actual_vs_ensemble.png"),
                "rf_feature_importance_csv": str(REPORTS_DIR / f"{suite_name}_{split_name}_rf_feature_importance.csv"),
                "xgb_feature_importance_csv": str(REPORTS_DIR / f"{suite_name}_{split_name}_xgb_feature_importance.csv"),
                "rows": len(predictions_df),
                "ensemble_mae": round(float(np.mean(np.abs(y_true - ensemble_pred))), 4),
            }

    summary_path = REPORTS_DIR / "regression_report_manifest.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved regression reports manifest to: {summary_path}")


if __name__ == "__main__":
    main()
