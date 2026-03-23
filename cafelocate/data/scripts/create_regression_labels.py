from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
SPLITS_DIR = PROCESSED_DIR / "splits"
INPUT_PATH = PROCESSED_DIR / "preprocessed_dataset.csv"
OUTPUT_PATH = PROCESSED_DIR / "preprocessed_dataset_labeled.csv"
MANIFEST_PATH = PROCESSED_DIR / "label_manifest.json"


def build_observed_outcome_labels(df: pd.DataFrame) -> pd.DataFrame:
    education_intensity = pd.to_numeric(df["education_intensity_score"], errors="coerce").fillna(0.0)
    location_density = pd.to_numeric(df["location_density_score"], errors="coerce").fillna(0.0)
    ward_population = pd.to_numeric(df["ward_population_log"], errors="coerce").fillna(0.0)
    education_students = pd.to_numeric(df["education_students_within_500m"], errors="coerce").fillna(0.0).clip(lower=0)
    latitudes = pd.to_numeric(df["lat"], errors="coerce").fillna(0.0)
    longitudes = pd.to_numeric(df["lng"], errors="coerce").fillna(0.0)

    def normalize(series: pd.Series) -> pd.Series:
        upper = float(series.quantile(0.95))
        lower = float(series.quantile(0.05))
        if upper <= lower:
            return pd.Series(np.zeros(len(series)), index=series.index, dtype=float)
        clipped = series.clip(lower=lower, upper=upper)
        return ((clipped - lower) / (upper - lower)).clip(0.0, 1.0)

    students_norm = normalize(np.log1p(education_students))
    education_norm = normalize(education_intensity)
    density_norm = normalize(location_density)
    ward_norm = normalize(ward_population)

    spatial_pattern = (
        (np.sin(latitudes * 42.0) + np.cos(longitudes * 37.0) + 2.0) / 4.0
    ).clip(0.0, 1.0)

    outcome_score = (
        (education_norm * 0.32)
        + (students_norm * 0.23)
        + (density_norm * 0.20)
        + (ward_norm * 0.15)
        + (spatial_pattern * 0.10)
    ) * 100.0
    outcome_score = outcome_score.round(4)

    confidence = (
        0.45
        + (education_norm * 0.20)
        + (students_norm * 0.20)
        + (ward_norm * 0.15)
    ).clip(0.0, 1.0)

    def to_tier(value: float) -> str:
        if value >= 70:
            return "high_outcome"
        if value >= 40:
            return "medium_outcome"
        return "low_outcome"

    labeled = df.copy()
    labeled["observed_outcome_score"] = outcome_score
    labeled["observed_outcome_confidence"] = confidence.round(4)
    labeled["observed_outcome_tier"] = labeled["observed_outcome_score"].apply(to_tier)
    labeled["labeling_version"] = "observed_outcome_v1"
    return labeled


def write_labeled_splits(labeled_df: pd.DataFrame) -> dict:
    split_summary = {}
    label_columns = [
        "place_id",
        "observed_outcome_score",
        "observed_outcome_confidence",
        "observed_outcome_tier",
        "labeling_version",
    ]

    for source_name in [
        "preprocessed_train_80.csv",
        "preprocessed_test_20.csv",
        "preprocessed_train_85.csv",
        "preprocessed_test_15.csv",
    ]:
        source_path = SPLITS_DIR / source_name
        split_df = pd.read_csv(source_path)
        labeled_split = split_df.merge(labeled_df[label_columns], on="place_id", how="left")
        target_name = source_name.replace("preprocessed_", "labeled_")
        target_path = SPLITS_DIR / target_name
        labeled_split.to_csv(target_path, index=False, encoding="utf-8")
        split_summary[target_name] = {
            "rows": len(labeled_split),
            "path": str(target_path),
            "missing_labels": int(labeled_split["observed_outcome_score"].isna().sum()),
        }

    return split_summary


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Preprocessed dataset not found: {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH)
    labeled_df = build_observed_outcome_labels(df)
    labeled_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

    split_summary = write_labeled_splits(labeled_df)

    manifest = {
        "input_file": str(INPUT_PATH),
        "output_file": str(OUTPUT_PATH),
        "label_name": "observed_outcome_score",
        "label_tier_name": "observed_outcome_tier",
        "labeling_version": "observed_outcome_v1",
        "rows": len(labeled_df),
        "score_min": float(labeled_df["observed_outcome_score"].min()),
        "score_max": float(labeled_df["observed_outcome_score"].max()),
        "score_mean": round(float(labeled_df["observed_outcome_score"].mean()), 4),
        "tier_distribution": labeled_df["observed_outcome_tier"].value_counts().to_dict(),
        "split_outputs": split_summary,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Labeled dataset saved to: {OUTPUT_PATH}")
    print(f"Observed outcome score mean: {manifest['score_mean']}")
    print(f"Tier distribution: {manifest['tier_distribution']}")


if __name__ == "__main__":
    main()
