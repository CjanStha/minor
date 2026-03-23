from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


BASE_DIR = Path(__file__).resolve().parent
REPORTS_DIR = BASE_DIR / "reports"
CLASSIFICATION_DIR = BASE_DIR / "classification_evaluation"
LABELS = ["low", "medium", "high"]


def score_to_level(score: float) -> str:
    if score >= 70:
        return "high"
    if score >= 40:
        return "medium"
    return "low"


def evaluate_predictions(pred_df: pd.DataFrame, suite: str, split: str) -> list[dict]:
    actual_labels = pred_df["actual_score"].apply(score_to_level)
    rows = []

    for model_col, model_name in [
        ("rf_prediction", "rf"),
        ("xgb_prediction", "xgb"),
    ]:
        pred_labels = pred_df[model_col].apply(score_to_level)

        accuracy = accuracy_score(actual_labels, pred_labels)
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            actual_labels,
            pred_labels,
            labels=LABELS,
            average="macro",
            zero_division=0,
        )
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            actual_labels,
            pred_labels,
            labels=LABELS,
            average="weighted",
            zero_division=0,
        )
        cm = confusion_matrix(actual_labels, pred_labels, labels=LABELS)

        rows.append(
            {
                "suite": suite,
                "split": split,
                "model": model_name,
                "accuracy": round(float(accuracy), 4),
                "precision_macro": round(float(precision_macro), 4),
                "recall_macro": round(float(recall_macro), 4),
                "f1_macro": round(float(f1_macro), 4),
                "precision_weighted": round(float(precision_weighted), 4),
                "recall_weighted": round(float(recall_weighted), 4),
                "f1_weighted": round(float(f1_weighted), 4),
                "support": int(len(pred_df)),
                "confusion_matrix_json": json.dumps(cm.tolist()),
            }
        )

        save_confusion_matrix(cm, suite, split, model_name)

    return rows


def save_confusion_matrix(cm: np.ndarray, suite: str, split: str, model: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(np.arange(len(LABELS)))
    ax.set_yticks(np.arange(len(LABELS)))
    ax.set_xticklabels([label.title() for label in LABELS])
    ax.set_yticklabels([label.title() for label in LABELS])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"{suite.upper()} {split} {model.upper()} Confusion Matrix")

    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", color="black")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(CLASSIFICATION_DIR / f"{suite}_{split}_{model}_confusion_matrix.png", dpi=180)
    plt.close(fig)


def save_metric_bars(summary_df: pd.DataFrame) -> None:
    metrics = [
        ("accuracy", "Accuracy", "#1d4ed8"),
        ("precision_macro", "Precision (Macro)", "#0f766e"),
        ("recall_macro", "Recall (Macro)", "#b45309"),
        ("f1_macro", "F1 Score (Macro)", "#7c3aed"),
    ]

    plot_df = summary_df.sort_values(["suite", "split", "model"]).copy()
    plot_df["label"] = plot_df["suite"] + " | " + plot_df["split"] + " | " + plot_df["model"].str.upper()

    for metric, title, color in metrics:
        plt.figure(figsize=(12, 6))
        plt.bar(plot_df["label"], plot_df[metric], color=color)
        plt.xticks(rotation=40, ha="right")
        plt.ylim(0, 1.05)
        plt.ylabel(metric)
        plt.title(f"Classification View - {title}")
        plt.tight_layout()
        plt.savefig(CLASSIFICATION_DIR / f"{metric}_all_models.png", dpi=180)
        plt.close()


def save_metric_tables(summary_df: pd.DataFrame) -> None:
    for metric in ["accuracy", "precision_macro", "recall_macro", "f1_macro"]:
        pivot = summary_df.pivot(index=["suite", "split"], columns="model", values=metric).reset_index()
        pivot.to_csv(CLASSIFICATION_DIR / f"{metric}_matrix.csv", index=False, encoding="utf-8")

        row_labels = pivot["suite"] + " | " + pivot["split"]
        value_df = pivot.drop(columns=["suite", "split"]).set_index(row_labels)

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
        plt.title(f"{metric.replace('_', ' ').title()} Matrix", pad=16)
        plt.tight_layout()
        plt.savefig(CLASSIFICATION_DIR / f"{metric}_matrix.png", dpi=180, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    CLASSIFICATION_DIR.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((REPORTS_DIR / "regression_report_manifest.json").read_text(encoding="utf-8"))

    rows = []
    for suite, suite_info in manifest.items():
        for split, split_info in suite_info.items():
            pred_df = pd.read_csv(split_info["prediction_file"])
            rows.extend(evaluate_predictions(pred_df, suite, split))

    summary_df = pd.DataFrame(rows).sort_values(["suite", "split", "model"]).reset_index(drop=True)
    summary_csv = CLASSIFICATION_DIR / "classification_metrics_all_models.csv"
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8")

    save_metric_bars(summary_df)
    save_metric_tables(summary_df)

    out_manifest = {
        "summary_csv": str(summary_csv),
        "generated_files": sorted(str(path) for path in CLASSIFICATION_DIR.iterdir() if path.is_file()),
        "labels": LABELS,
        "total_model_rows": int(len(summary_df)),
    }
    (CLASSIFICATION_DIR / "classification_manifest.json").write_text(json.dumps(out_manifest, indent=2), encoding="utf-8")

    print(f"Saved classification metrics to: {summary_csv}")
    print(f"Generated files: {len(out_manifest['generated_files'])}")


if __name__ == "__main__":
    main()
