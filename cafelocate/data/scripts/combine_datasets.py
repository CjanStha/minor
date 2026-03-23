from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from shapely.geometry import Point, shape


ROOT_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT_DIR / "data" / "raw_data"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
OUTPUT_PATH = PROCESSED_DIR / "combined_dataset.csv"

CAFE_PATH = RAW_DIR / "kathmandu_cafes.csv"
ENRICHED_PATH = RAW_DIR / "dataset_ft_enriched.csv"
CENSUS_PATH = RAW_DIR / "kathmandu_census.csv"
WARD_PATH = RAW_DIR / "kathmandu_wards_boundary_sorted.csv"
EDUCATION_PATH = RAW_DIR / "kathmandu_education_cleaned.csv"
AMENITIES_CLEAN_PATH = RAW_DIR / "amenities_clean.csv"
OSM_AMENITIES_PATH = RAW_DIR / "osm_amenities_kathmandu.csv"
ROADS_PATH = RAW_DIR / "osm_roads_kathmandu.csv"

SCHOOL_TYPES = {"school", "college", "university"}
HOSPITAL_TYPES = {"hospital", "health_post", "clinic", "pharmacy"}
BUS_TYPES = {"bus_station", "bus_stop"}


def haversine_vector(lat1: float, lng1: float, lat2: np.ndarray, lng2: np.ndarray) -> np.ndarray:
    radius_m = 6_371_000.0
    lat1_rad = np.radians(lat1)
    lng1_rad = np.radians(lng1)
    lat2_rad = np.radians(lat2.astype(float))
    lng2_rad = np.radians(lng2.astype(float))

    dlat = lat2_rad - lat1_rad
    dlng = lng2_rad - lng1_rad

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlng / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return radius_m * c


def compute_distance_features(base_df: pd.DataFrame, points_df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    result = pd.DataFrame(index=base_df.index)

    if points_df.empty:
        result[f"{prefix}_within_500m"] = 0
        result[f"{prefix}_within_200m"] = 0
        result[f"{prefix}_min_distance"] = np.nan
        return result

    point_lats = points_df["latitude"].to_numpy(dtype=float)
    point_lngs = points_df["longitude"].to_numpy(dtype=float)

    within_500 = []
    within_200 = []
    min_distance = []

    for lat, lng in base_df[["lat", "lng"]].itertuples(index=False):
        distances = haversine_vector(float(lat), float(lng), point_lats, point_lngs)
        within_500.append(int(np.sum(distances <= 500)))
        within_200.append(int(np.sum(distances <= 200)))
        min_distance.append(round(float(distances.min()), 2) if len(distances) else np.nan)

    result[f"{prefix}_within_500m"] = within_500
    result[f"{prefix}_within_200m"] = within_200
    result[f"{prefix}_min_distance"] = min_distance
    return result


def compute_competitor_features(cafes_df: pd.DataFrame) -> pd.DataFrame:
    features = pd.DataFrame(index=cafes_df.index)
    latitudes = cafes_df["lat"].to_numpy(dtype=float)
    longitudes = cafes_df["lng"].to_numpy(dtype=float)

    within_500 = []
    within_200 = []
    min_distance = []
    avg_distance = []

    for idx, (lat, lng) in enumerate(zip(latitudes, longitudes)):
        distances = haversine_vector(lat, lng, latitudes, longitudes)
        mask = np.ones(len(distances), dtype=bool)
        mask[idx] = False
        distances = distances[mask]

        within_500.append(int(np.sum(distances <= 500)))
        within_200.append(int(np.sum(distances <= 200)))
        min_distance.append(round(float(distances.min()), 2) if len(distances) else np.nan)
        avg_distance.append(round(float(distances.mean()), 2) if len(distances) else np.nan)

    features["computed_competitors_within_500m"] = within_500
    features["computed_competitors_within_200m"] = within_200
    features["computed_competitors_min_distance"] = min_distance
    features["computed_competitors_avg_distance"] = avg_distance
    return features


def load_ward_geometries() -> list[tuple[int, object]]:
    wards_df = pd.read_csv(WARD_PATH)
    ward_geometries = []
    for row in wards_df.itertuples(index=False):
        geometry = shape(json.loads(row.geometry_json))
        ward_geometries.append((int(row.ward_number), geometry))
    return ward_geometries


def assign_wards(cafes_df: pd.DataFrame, ward_geometries: list[tuple[int, object]]) -> list[int | None]:
    assigned = []
    for lat, lng in cafes_df[["lat", "lng"]].itertuples(index=False):
        point = Point(float(lng), float(lat))
        ward_number = None
        nearest_ward = None
        nearest_distance = None
        for current_ward, geom in ward_geometries:
            if geom.covers(point):
                ward_number = current_ward
                break
            distance = geom.distance(point)
            if nearest_distance is None or distance < nearest_distance:
                nearest_distance = distance
                nearest_ward = current_ward
        if ward_number is None:
            ward_number = nearest_ward
        assigned.append(ward_number)
    return assigned


def build_amenity_frame() -> pd.DataFrame:
    osm_amenities = pd.read_csv(OSM_AMENITIES_PATH).rename(columns={"amenity_type": "amenity"})
    osm_amenities = osm_amenities[["amenity", "name", "latitude", "longitude"]]

    clean_amenities = pd.read_csv(AMENITIES_CLEAN_PATH)[["amenity", "name", "latitude", "longitude"]]

    education = pd.read_csv(EDUCATION_PATH).rename(
        columns={"student:count": "student_count", "operator:type": "operator_type"}
    )
    education["amenity"] = education["amenity"].fillna("school")
    education = education[["amenity", "name", "latitude", "longitude", "student_count", "education_level", "operator_type"]]

    combined = pd.concat(
        [
            osm_amenities.assign(source_dataset="osm_amenities"),
            clean_amenities.assign(source_dataset="amenities_clean"),
            education.assign(source_dataset="education"),
        ],
        ignore_index=True,
        sort=False,
    )

    combined["name"] = combined["name"].fillna("").astype(str).str.strip()
    combined["amenity"] = combined["amenity"].fillna("").astype(str).str.strip().str.lower()
    combined["latitude"] = pd.to_numeric(combined["latitude"], errors="coerce")
    combined["longitude"] = pd.to_numeric(combined["longitude"], errors="coerce")
    combined = combined.dropna(subset=["latitude", "longitude"])
    combined["dedupe_key"] = (
        combined["amenity"]
        + "|"
        + combined["name"].str.lower()
        + "|"
        + combined["latitude"].round(5).astype(str)
        + "|"
        + combined["longitude"].round(5).astype(str)
    )
    combined = combined.drop_duplicates(subset=["dedupe_key"]).drop(columns=["dedupe_key"])
    return combined


def add_education_summary(base_df: pd.DataFrame, education_df: pd.DataFrame) -> pd.DataFrame:
    if education_df.empty:
        base_df["education_points_within_500m"] = 0
        base_df["education_points_within_200m"] = 0
        base_df["education_students_within_500m"] = 0
        return base_df

    edu_lats = education_df["latitude"].to_numpy(dtype=float)
    edu_lngs = education_df["longitude"].to_numpy(dtype=float)
    student_counts = pd.to_numeric(education_df.get("student_count"), errors="coerce").fillna(0).to_numpy(dtype=float)

    within_500 = []
    within_200 = []
    students_500 = []

    for lat, lng in base_df[["lat", "lng"]].itertuples(index=False):
        distances = haversine_vector(float(lat), float(lng), edu_lats, edu_lngs)
        mask_500 = distances <= 500
        mask_200 = distances <= 200
        within_500.append(int(np.sum(mask_500)))
        within_200.append(int(np.sum(mask_200)))
        students_500.append(int(student_counts[mask_500].sum()))

    base_df["education_points_within_500m"] = within_500
    base_df["education_points_within_200m"] = within_200
    base_df["education_students_within_500m"] = students_500
    return base_df


def derive_fallback_road_metrics(base_df: pd.DataFrame) -> pd.DataFrame:
    road_rows = len(pd.read_csv(ROADS_PATH))
    base_road_m = 2000.0 if road_rows else 1500.0
    original_road_distance = pd.to_numeric(base_df["roads_avg_distance"], errors="coerce")

    competitor_factor = np.minimum(
        1.5,
        0.5 + (pd.to_numeric(base_df["competitors_within_500m"], errors="coerce").fillna(0) / 30.0),
    )
    pop_factor = np.minimum(
        1.3,
        pd.to_numeric(base_df["population_density"], errors="coerce").fillna(0) / 10_000.0,
    ).clip(lower=0.4)

    coord_hash = (
        ((base_df["lat"].astype(float).round(3) * 1000).astype(int) * 31)
        + ((base_df["lng"].astype(float).round(3) * 1000).astype(int) * 17)
    ).abs() % 1000 / 1000.0
    location_factor = 0.8 + (coord_hash * 0.4)

    estimated_road_m = (base_road_m * competitor_factor * pop_factor * location_factor).clip(lower=1500, upper=3500)

    base_df["roads_avg_distance"] = original_road_distance.fillna(estimated_road_m.round(0))
    base_df["roads_within_500m"] = pd.to_numeric(base_df["roads_within_500m"], errors="coerce").fillna(
        np.minimum(20, np.maximum(0, np.round(base_df["roads_avg_distance"] / 200.0)))
    )
    base_df["road_feature_source"] = np.where(original_road_distance.notna(), "enriched", "fallback")
    return base_df


def derive_model_features(base_df: pd.DataFrame) -> pd.DataFrame:
    same_type_weight = 0.0
    same_type_weight_200m = 0.0

    base_df["competitors_within_500m"] = pd.to_numeric(base_df["competitors_within_500m"], errors="coerce").fillna(
        base_df["computed_competitors_within_500m"]
    )
    base_df["competitors_within_200m"] = pd.to_numeric(base_df["competitors_within_200m"], errors="coerce").fillna(
        base_df["computed_competitors_within_200m"] + same_type_weight_200m
    )
    base_df["competitors_min_distance"] = pd.to_numeric(base_df["competitors_min_distance"], errors="coerce").fillna(
        base_df["computed_competitors_min_distance"]
    )
    base_df["competitors_avg_distance"] = pd.to_numeric(base_df["competitors_avg_distance"], errors="coerce").fillna(
        base_df["computed_competitors_avg_distance"]
    )

    road_access_score = np.clip(10.0 - (base_df["roads_avg_distance"].astype(float) / 150.0), 0.0, 10.0)
    bus_access_bonus = np.minimum(2.5, base_df["bus_stops_within_500m"].astype(float) * 0.35)
    school_bonus = np.minimum(1.5, base_df["schools_within_500m"].astype(float) * 0.15)
    hospital_bonus = np.minimum(1.0, base_df["hospitals_within_500m"].astype(float) * 0.15)

    density_signal = np.minimum(4.0, base_df["population_density_proxy"].astype(float))
    transit_signal = np.minimum(3.0, base_df["bus_stops_within_500m"].astype(float) * 0.4)
    institutional_signal = np.minimum(2.0, base_df["schools_within_500m"].astype(float) * 0.2)
    commerce_signal = np.minimum(2.0, base_df["competitors_within_500m"].astype(float) * 0.12)
    road_signal = np.minimum(2.0, np.maximum(0.0, 2.0 - (base_df["roads_avg_distance"].astype(float) / 300.0)))

    avg_rating = pd.to_numeric(base_df["rating"], errors="coerce").fillna(0.0)

    base_df["accessibility_score"] = pd.to_numeric(base_df["accessibility_score"], errors="coerce").fillna(
        (road_access_score + bus_access_bonus + school_bonus + hospital_bonus).clip(0.0, 10.0).round(2)
    )
    base_df["foot_traffic_score"] = pd.to_numeric(base_df["foot_traffic_score"], errors="coerce").fillna(
        (density_signal + transit_signal + institutional_signal + commerce_signal + road_signal).clip(0.0, 10.0).round(2)
    )
    base_df["competition_pressure"] = pd.to_numeric(base_df["competition_pressure"], errors="coerce").fillna(
        (
            (base_df["competitors_within_500m"].astype(float) * 0.30)
            + (same_type_weight * 0.85)
            + np.minimum(2.0, avg_rating * 0.25)
            + np.minimum(2.5, avg_rating * 0.55)
        ).clip(0.0, 10.0).round(2)
    )

    return base_df


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    cafes_df = pd.read_csv(CAFE_PATH)
    enriched_df = pd.read_csv(ENRICHED_PATH)
    census_df = pd.read_csv(CENSUS_PATH)

    base_df = cafes_df.merge(
        enriched_df.drop(columns=["name", "lat", "lng", "type", "is_operational", "source"], errors="ignore"),
        on="place_id",
        how="left",
    )

    ward_geometries = load_ward_geometries()
    base_df["ward_number"] = assign_wards(base_df, ward_geometries)

    census_df = census_df.rename(columns={"ward_no": "ward_number"})
    base_df = base_df.merge(census_df, on="ward_number", how="left")
    base_df["population_density_proxy"] = pd.to_numeric(base_df["population_density_proxy"], errors="coerce").fillna(
        pd.to_numeric(base_df["population_density"], errors="coerce") / 1000.0
    )

    amenity_df = build_amenity_frame()
    education_df = amenity_df[amenity_df["source_dataset"] == "education"].copy()

    school_features = compute_distance_features(base_df, amenity_df[amenity_df["amenity"].isin(SCHOOL_TYPES)], "computed_schools")
    hospital_features = compute_distance_features(base_df, amenity_df[amenity_df["amenity"].isin(HOSPITAL_TYPES)], "computed_hospitals")
    bus_features = compute_distance_features(base_df, amenity_df[amenity_df["amenity"].isin(BUS_TYPES)], "computed_bus_stops")
    competitor_features = compute_competitor_features(base_df)

    base_df = pd.concat([base_df, school_features, hospital_features, bus_features, competitor_features], axis=1)

    base_df["schools_within_500m"] = pd.to_numeric(base_df["schools_within_500m"], errors="coerce").fillna(
        base_df["computed_schools_within_500m"]
    )
    base_df["schools_within_200m"] = pd.to_numeric(base_df["schools_within_200m"], errors="coerce").fillna(
        base_df["computed_schools_within_200m"]
    )
    base_df["schools_min_distance"] = pd.to_numeric(base_df["schools_min_distance"], errors="coerce").fillna(
        base_df["computed_schools_min_distance"]
    )

    base_df["hospitals_within_500m"] = pd.to_numeric(base_df["hospitals_within_500m"], errors="coerce").fillna(
        base_df["computed_hospitals_within_500m"]
    )
    base_df["hospitals_min_distance"] = pd.to_numeric(base_df["hospitals_min_distance"], errors="coerce").fillna(
        base_df["computed_hospitals_min_distance"]
    )

    base_df["bus_stops_within_500m"] = pd.to_numeric(base_df["bus_stops_within_500m"], errors="coerce").fillna(
        base_df["computed_bus_stops_within_500m"]
    )
    base_df["bus_stops_min_distance"] = pd.to_numeric(base_df["bus_stops_min_distance"], errors="coerce").fillna(
        base_df["computed_bus_stops_min_distance"]
    )

    base_df = add_education_summary(base_df, education_df)
    base_df = derive_fallback_road_metrics(base_df)
    base_df = derive_model_features(base_df)

    base_df["dataset_count_raw_files"] = 8
    base_df["combined_at"] = pd.Timestamp.utcnow().isoformat()
    base_df = base_df.sort_values(["ward_number", "name"], na_position="last").reset_index(drop=True)
    base_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

    print(f"Combined dataset saved to: {OUTPUT_PATH}")
    print(f"Rows: {len(base_df)}")
    print(f"Columns: {len(base_df.columns)}")


if __name__ == "__main__":
    main()
