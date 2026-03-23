from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
EVAL_DIR = BASE_DIR / "evaluation"

PSEUDO_METRICS_PATH = MODELS_DIR / "regression_metrics.json"
OBSERVED_METRICS_PATH = MODELS_DIR / "regression_metrics_observed.json"


def load_training_metrics(metrics_path: Path, suite_name: str) -> list[dict]:
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    rows = []

    for split_name, split_metrics in payload["training_runs"].items():
        for model_key, model_label in [("random_forest", "rf"), ("xgboost", "xgb")]:
            metrics = split_metrics[model_key]
            rows.append(
                {
                    "suite": suite_name,
                    "split": split_name,
                    "model": model_label,
                    "target_name": split_metrics["target_name"],
                    "train_rows": split_metrics["train_rows"],
                    "test_rows": split_metrics["test_rows"],
                    "mae": metrics["mae"],
                    "rmse": metrics["rmse"],
                    "r2": metrics["r2"],
                    "prediction_min": metrics["prediction_min"],
                    "prediction_max": metrics["prediction_max"],
                }
            )

    return rows


def load_ensemble_metrics() -> list[dict]:
    manifest = json.loads((REPORTS_DIR / "regression_report_manifest.json").read_text(encoding="utf-8"))
    rows = []

    for suite_name, suite_data in manifest.items():
        for split_name, split_info in suite_data.items():
            pred_df = pd.read_csv(split_info["prediction_file"])
            actual = pred_df["actual_score"].to_numpy(dtype=float)
            ensemble = pred_df["ensemble_prediction"].to_numpy(dtype=float)
            mae = mean_absolute_error(actual, ensemble)
            rmse = mean_squared_error(actual, ensemble) ** 0.5
            r2 = r2_score(actual, ensemble)

            rows.append(
                {
                    "suite": suite_name,
                    "split": split_name,
                    "model": "ensemble",
                    "target_name": "actual_score",
                    "train_rows": np.nan,
                    "test_rows": int(len(pred_df)),
                    "mae": round(float(mae), 4),
                    "rmse": round(float(rmse), 4),
                    "r2": round(float(r2), 4),
                    "prediction_min": round(float(pred_df["ensemble_prediction"].min()), 4),
                    "prediction_max": round(float(pred_df["ensemble_prediction"].max()), 4),
                }
            )

    return rows


def save_grouped_metric_charts(summary_df: pd.DataFrame) -> None:
    plots = [
        ("mae", "Mean Absolute Error", "#b45309"),
        ("rmse", "Root Mean Square Error", "#0f766e"),
        ("r2", "R2 Score", "#1d4ed8"),
    ]

    order = summary_df[summary_df["model"] != "ensemble"].sort_values(["suite", "split", "model"]).copy()
    order["label"] = order["suite"] + " | " + order["split"] + " | " + order["model"].str.upper()

    for metric, title, color in plots:
        plt.figure(figsize=(12, 6))
        plt.bar(order["label"], order[metric], color=color)
        plt.xticks(rotation=40, ha="right")
        plt.ylabel(metric.upper())
        plt.title(f"All Model Runs - {title}")
        plt.tight_layout()
        plt.savefig(EVAL_DIR / f"{metric}_all_models.png", dpi=180)
        plt.close()


def save_suite_split_comparison(summary_df: pd.DataFrame) -> None:
    for suite_name in sorted(summary_df["suite"].unique()):
        suite_df = summary_df[(summary_df["suite"] == suite_name) & (summary_df["model"] != "ensemble")].copy()
        splits = sorted(suite_df["split"].unique())
        models = ["rf", "xgb"]
        metrics = ["mae", "rmse", "r2"]

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        for ax, metric in zip(axes, metrics):
            x = np.arange(len(splits))
            width = 0.32

            for idx, model in enumerate(models):
                values = []
                for split_name in splits:
                    row = suite_df[(suite_df["split"] == split_name) & (suite_df["model"] == model)]
                    values.append(float(row.iloc[0][metric]) if not row.empty else np.nan)

                ax.bar(x + (idx - 0.5) * width, values, width=width, label=model.upper())

            ax.set_xticks(x)
            ax.set_xticklabels(splits)
            ax.set_title(metric.upper())
            ax.grid(axis="y", alpha=0.2)

        axes[0].set_ylabel("Metric Value")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=2)
        fig.suptitle(f"{suite_name.upper()} Model Comparison by Split")
        fig.tight_layout(rect=[0, 0, 1, 0.92])
        fig.savefig(EVAL_DIR / f"{suite_name}_split_model_comparison.png", dpi=180)
        plt.close(fig)


def save_heatmap_like_tables(summary_df: pd.DataFrame) -> None:
    for metric in ["mae", "rmse", "r2"]:
        pivot = summary_df[summary_df["model"] != "ensemble"].pivot(index=["suite", "split"], columns="model", values=metric).reset_index()
        pivot.to_csv(EVAL_DIR / f"{metric}_matrix.csv", index=False, encoding="utf-8")

        display_df = pivot.copy()
        row_labels = display_df["suite"] + " | " + display_df["split"]
        value_df = display_df.drop(columns=["suite", "split"]).set_index(row_labels)

        fig, ax = plt.subplots(figsize=(7, 3 + len(value_df) * 0.45))
        ax.axis("off")
        table = ax.table(
            cellText=np.round(value_df.values, 4),
            rowLabels=value_df.index,
            colLabels=value_df.columns.str.upper(),
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        plt.title(f"{metric.upper()} Evaluation Matrix", pad=16)
        plt.tight_layout()
        plt.savefig(EVAL_DIR / f"{metric}_matrix.png", dpi=180, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    rows.extend(load_training_metrics(PSEUDO_METRICS_PATH, "pseudo"))
    rows.extend(load_training_metrics(OBSERVED_METRICS_PATH, "observed"))
    rows.extend(load_ensemble_metrics())

    summary_df = pd.DataFrame(rows).sort_values(["suite", "split", "model"]).reset_index(drop=True)
    summary_csv = EVAL_DIR / "all_model_evaluation_metrics.csv"
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8")

    save_grouped_metric_charts(summary_df)
    save_suite_split_comparison(summary_df)
    save_heatmap_like_tables(summary_df)

    manifest = {
        "summary_csv": str(summary_csv),
        "generated_files": sorted(str(path) for path in EVAL_DIR.iterdir() if path.is_file()),
        "total_model_rows": int(len(summary_df)),
    }
    (EVAL_DIR / "evaluation_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Saved evaluation summary to: {summary_csv}")
    print(f"Generated files: {len(manifest['generated_files'])}")


if __name__ == "__main__":
    main()
