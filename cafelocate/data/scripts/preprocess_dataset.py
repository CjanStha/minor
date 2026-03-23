from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
INPUT_PATH = PROCESSED_DIR / "combined_dataset.csv"
OUTPUT_PATH = PROCESSED_DIR / "preprocessed_dataset.csv"

NUMERIC_COLUMNS = [
    "rating",
    "review_count",
    "price_level",
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
    "population",
    "households",
    "area_sqkm",
    "population_density",
    "population_density_proxy",
    "accessibility_score",
    "foot_traffic_score",
    "competition_pressure",
    "education_points_within_500m",
    "education_points_within_200m",
    "education_students_within_500m",
]


def normalize_text(series: pd.Series) -> pd.Series:
    return (
        series.fillna("")
        .astype(str)
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Combined dataset not found: {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH)
    df = df.drop_duplicates(subset=["place_id"]).copy()

    for column in ["name", "type", "source", "road_feature_source"]:
        if column in df.columns:
            df[column] = normalize_text(df[column])

    if "is_operational" in df.columns:
        df["is_operational"] = df["is_operational"].fillna(False).astype(bool)

    for column in NUMERIC_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    median_fill_columns = [
        "rating",
        "review_count",
        "price_level",
        "competitors_min_distance",
        "competitors_avg_distance",
        "roads_within_500m",
        "roads_avg_distance",
        "schools_min_distance",
        "hospitals_min_distance",
        "bus_stops_min_distance",
        "population",
        "households",
        "area_sqkm",
        "population_density",
        "population_density_proxy",
        "accessibility_score",
        "foot_traffic_score",
        "competition_pressure",
    ]
    for column in median_fill_columns:
        if column in df.columns:
            non_null = df[column].dropna()
            median_value = non_null.median() if not non_null.empty else np.nan
            df[column] = df[column].fillna(0 if pd.isna(median_value) else median_value)

    zero_fill_columns = [
        "competitors_within_500m",
        "competitors_within_200m",
        "schools_within_500m",
        "schools_within_200m",
        "hospitals_within_500m",
        "bus_stops_within_500m",
        "education_points_within_500m",
        "education_points_within_200m",
        "education_students_within_500m",
    ]
    for column in zero_fill_columns:
        if column in df.columns:
            df[column] = df[column].fillna(0)

    df["rating"] = df["rating"].clip(lower=0, upper=5)
    df["review_count"] = df["review_count"].clip(lower=0)
    df["price_level"] = df["price_level"].clip(lower=0)
    df["population_density_proxy"] = df["population_density"] / 1000.0

    df["has_rating"] = (df["rating"] > 0).astype(int)
    df["log_review_count"] = np.log1p(df["review_count"])
    df["rating_review_signal"] = (df["rating"] * df["log_review_count"]).round(4)
    df["education_intensity_score"] = (
        (df["education_points_within_500m"] * 0.35)
        + (df["education_points_within_200m"] * 0.45)
        + (np.log1p(df["education_students_within_500m"]) * 0.20)
    ).round(4)
    df["ward_population_log"] = np.log1p(df["population"]).round(4)
    df["location_density_score"] = (
        (df["population_density_proxy"] * 0.35)
        + (df["foot_traffic_score"] * 0.35)
        + (df["accessibility_score"] * 0.30)
    ).round(4)

    final_order = [
        "place_id",
        "name",
        "type",
        "source",
        "lat",
        "lng",
        "ward_number",
        "population",
        "households",
        "area_sqkm",
        "population_density",
        "population_density_proxy",
        "rating",
        "review_count",
        "price_level",
        "has_rating",
        "log_review_count",
        "rating_review_signal",
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
        "education_points_within_500m",
        "education_points_within_200m",
        "education_students_within_500m",
        "education_intensity_score",
        "accessibility_score",
        "foot_traffic_score",
        "competition_pressure",
        "location_density_score",
        "ward_population_log",
        "road_feature_source",
        "is_operational",
    ]

    remaining_columns = [column for column in df.columns if column not in final_order]
    df = df[final_order + remaining_columns]
    df = df.sort_values(["ward_number", "name"], na_position="last").reset_index(drop=True)
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

    print(f"Preprocessed dataset saved to: {OUTPUT_PATH}")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")


if __name__ == "__main__":
    main()
