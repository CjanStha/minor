from __future__ import annotations

import csv
import json
import subprocess
import sys
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
ML_DIR = PROJECT_ROOT / "ml"
PYTHON_EXE = Path(sys.executable)


def run_script(relative_path: str) -> subprocess.CompletedProcess[str]:
    script_path = PROJECT_ROOT / relative_path
    return subprocess.run(
        [str(PYTHON_EXE), str(script_path)],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        check=True,
    )


class DataPipelineTests(unittest.TestCase):
    def test_combine_dataset_output_exists_and_has_expected_columns(self) -> None:
        combined_path = DATA_DIR / "processed" / "combined_dataset.csv"
        self.assertTrue(combined_path.exists(), "combined_dataset.csv should exist")

        df = pd.read_csv(combined_path, nrows=10)
        expected_columns = {
            "place_id",
            "ward_number",
            "population_density_proxy",
            "competitors_within_500m",
            "roads_avg_distance",
            "schools_within_500m",
            "hospitals_within_500m",
            "bus_stops_within_500m",
            "accessibility_score",
            "foot_traffic_score",
            "competition_pressure",
        }
        self.assertTrue(expected_columns.issubset(df.columns))

    def test_preprocessed_dataset_has_engineered_columns(self) -> None:
        preprocessed_path = DATA_DIR / "processed" / "preprocessed_dataset.csv"
        self.assertTrue(preprocessed_path.exists(), "preprocessed_dataset.csv should exist")

        df = pd.read_csv(preprocessed_path, nrows=10)
        expected_columns = {
            "education_intensity_score",
            "location_density_score",
            "ward_population_log",
            "log_review_count",
            "rating_review_signal",
        }
        self.assertTrue(expected_columns.issubset(df.columns))

    def test_split_files_have_expected_row_counts(self) -> None:
        base = DATA_DIR / "processed" / "splits"
        counts = {
            "preprocessed_train_80.csv": 857,
            "preprocessed_test_20.csv": 215,
            "preprocessed_train_85.csv": 911,
            "preprocessed_test_15.csv": 161,
        }
        for name, expected in counts.items():
            path = base / name
            self.assertTrue(path.exists(), f"{name} should exist")
            with path.open("r", encoding="utf-8", newline="") as handle:
                row_count = sum(1 for _ in csv.reader(handle)) - 1
            self.assertEqual(row_count, expected)

    def test_label_manifest_matches_labeled_dataset(self) -> None:
        labeled_path = DATA_DIR / "processed" / "preprocessed_dataset_labeled.csv"
        manifest_path = DATA_DIR / "processed" / "label_manifest.json"
        self.assertTrue(labeled_path.exists())
        self.assertTrue(manifest_path.exists())

        labeled_df = pd.read_csv(labeled_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        self.assertEqual(len(labeled_df), manifest["rows"])
        self.assertIn("observed_outcome_score", labeled_df.columns)
        self.assertEqual(
            labeled_df["observed_outcome_tier"].value_counts().to_dict(),
            manifest["tier_distribution"],
        )


class MlArtifactsTests(unittest.TestCase):
    def test_regression_model_artifacts_exist(self) -> None:
        models_dir = ML_DIR / "models"
        required = [
            "rf_regressor_v3_80_20.pkl",
            "xgb_regressor_v3_80_20.pkl",
            "rf_regressor_v3_85_15.pkl",
            "xgb_regressor_v3_85_15.pkl",
            "rf_regressor_observed_v1_80_20.pkl",
            "xgb_regressor_observed_v1_80_20.pkl",
            "rf_regressor_observed_v1_85_15.pkl",
            "xgb_regressor_observed_v1_85_15.pkl",
            "scaler_regression.pkl",
            "feature_columns_regression.pkl",
            "scaler_regression_observed.pkl",
            "feature_columns_regression_observed.pkl",
        ]
        for name in required:
            self.assertTrue((models_dir / name).exists(), f"{name} should exist")

    def test_evaluation_outputs_exist(self) -> None:
        eval_dir = ML_DIR / "evaluation"
        class_dir = ML_DIR / "classification_evaluation"
        required = [
            eval_dir / "all_model_evaluation_metrics.csv",
            eval_dir / "mae_all_models.png",
            eval_dir / "rmse_all_models.png",
            eval_dir / "r2_all_models.png",
            class_dir / "classification_metrics_all_models.csv",
            class_dir / "accuracy_all_models.png",
            class_dir / "precision_macro_all_models.png",
            class_dir / "recall_macro_all_models.png",
            class_dir / "f1_macro_all_models.png",
        ]
        for path in required:
            self.assertTrue(path.exists(), f"{path.name} should exist")

    def test_classification_metrics_have_only_rf_and_xgb_rows(self) -> None:
        metrics_path = ML_DIR / "classification_evaluation" / "classification_metrics_all_models.csv"
        df = pd.read_csv(metrics_path)
        self.assertEqual(set(df["model"]), {"rf", "xgb"})

    def test_regression_metric_matrices_exclude_ensemble_column(self) -> None:
        mae_matrix = pd.read_csv(ML_DIR / "evaluation" / "mae_matrix.csv")
        self.assertNotIn("ensemble", mae_matrix.columns)
        self.assertEqual(set(mae_matrix.columns), {"suite", "split", "rf", "xgb"})


class ScriptSmokeTests(unittest.TestCase):
    def test_predict_regression_cli_runs_for_observed_suite(self) -> None:
        output_path = DATA_DIR / "processed" / "splits" / "smoke_observed_predictions.csv"
        result = subprocess.run(
            [
                str(PYTHON_EXE),
                str(ML_DIR / "predict_regression.py"),
                "--suite",
                "observed",
                "--split",
                "85_15",
                "--model",
                "ensemble",
                "--input-csv",
                str(DATA_DIR / "processed" / "splits" / "labeled_test_15.csv"),
                "--output-csv",
                str(output_path),
            ],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertIn("Predictions saved to:", result.stdout)
        self.assertTrue(output_path.exists())
        df = pd.read_csv(output_path, nrows=3)
        self.assertIn("ensemble_prediction", df.columns)


if __name__ == "__main__":
    unittest.main()
