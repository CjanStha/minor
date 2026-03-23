from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


ROOT_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
SPLITS_DIR = PROCESSED_DIR / "splits"
INPUT_PATH = PROCESSED_DIR / "preprocessed_dataset.csv"
RANDOM_STATE = 42


def choose_stratify_series(df: pd.DataFrame) -> pd.Series | None:
    for column in ["source", "is_operational"]:
        if column not in df.columns:
            continue
        counts = df[column].value_counts(dropna=False)
        if len(counts) > 1 and int(counts.min()) >= 2:
            return df[column]
    return None


def save_split(df: pd.DataFrame, train_size: float, train_label: str, test_label: str) -> dict:
    stratify_series = choose_stratify_series(df)
    train_df, test_df = train_test_split(
        df,
        train_size=train_size,
        random_state=RANDOM_STATE,
        shuffle=True,
        stratify=stratify_series,
    )

    train_df = train_df.sort_values("place_id").reset_index(drop=True)
    test_df = test_df.sort_values("place_id").reset_index(drop=True)

    train_path = SPLITS_DIR / f"preprocessed_train_{train_label}.csv"
    test_path = SPLITS_DIR / f"preprocessed_test_{test_label}.csv"
    train_df.to_csv(train_path, index=False, encoding="utf-8")
    test_df.to_csv(test_path, index=False, encoding="utf-8")

    return {
        "train_ratio": train_size,
        "test_ratio": round(1 - train_size, 2),
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "train_file": str(train_path),
        "test_file": str(test_path),
        "stratified_by": stratify_series.name if stratify_series is not None else None,
    }


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Preprocessed dataset not found: {INPUT_PATH}")

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(INPUT_PATH)

    split_manifest = {
        "input_file": str(INPUT_PATH),
        "total_rows": len(df),
        "random_state": RANDOM_STATE,
        "splits": {
            "80_20": save_split(df, train_size=0.80, train_label="80", test_label="20"),
            "85_15": save_split(df, train_size=0.85, train_label="85", test_label="15"),
        },
    }

    manifest_path = SPLITS_DIR / "split_manifest.json"
    manifest_path.write_text(json.dumps(split_manifest, indent=2), encoding="utf-8")

    print(f"Saved dataset splits to: {SPLITS_DIR}")
    for name, info in split_manifest["splits"].items():
        print(f"{name}: train={info['train_rows']} test={info['test_rows']} stratified_by={info['stratified_by']}")


if __name__ == "__main__":
    main()
