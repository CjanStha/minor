import json
from pathlib import Path
from textwrap import dedent


def md_cell(text):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": dedent(text).strip().splitlines(keepends=True),
    }


def code_cell(text):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": dedent(text).strip().splitlines(keepends=True),
    }


cells = [
    md_cell(
        """
        # CafeLocate ML Pipeline Notebook

        This notebook documents the machine learning workflow for the combined cafe-location dataset:

        - collection overview of the project datasets
        - processing of the combined dataset
        - train/test split strategies: **80:20** and **85:15**
        - training of **Random Forest** and **XGBoost**
        - evaluation of each model
        - comparison in **tabular** and **graphical** form
        - final conclusion for the ML pipelines
        """
    ),
    code_cell(
        """
        from pathlib import Path
        import warnings
        import json

        from IPython import get_ipython
        import matplotlib
        import joblib
        import numpy as np
        import pandas as pd
        import seaborn as sns
        import matplotlib.pyplot as plt

        from IPython.display import display, Markdown
        from sklearn.preprocessing import StandardScaler, LabelEncoder
        from sklearn.model_selection import train_test_split, cross_val_score
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import (
            accuracy_score,
            precision_score,
            recall_score,
            f1_score,
            confusion_matrix,
            classification_report,
        )
        from xgboost import XGBClassifier

        ipython = get_ipython()
        if ipython is not None:
            ipython.run_line_magic("matplotlib", "inline")
        else:
            matplotlib.use("Agg")

        warnings.filterwarnings("ignore")
        sns.set_theme(style="whitegrid")
        plt.rcParams["figure.figsize"] = (10, 6)

        def find_ml_dir(start: Path) -> Path:
            start = start.resolve()
            candidates = [start, *start.parents]
            for candidate in candidates:
                if (candidate / "models").exists() and (candidate.parent / "data").exists():
                    return candidate
                if (candidate / "cafelocate" / "ml").exists() and (candidate / "cafelocate" / "data").exists():
                    return candidate / "cafelocate" / "ml"
            return start

        ML_DIR = find_ml_dir(Path.cwd())
        DATA_DIR = (ML_DIR.parent / "data").resolve()
        RAW_DIR = (DATA_DIR / "raw_data").resolve()
        MODELS_DIR = (ML_DIR / "models").resolve()
        SPLITS_DIR = (DATA_DIR / "splits").resolve()

        COMBINED_DATASET = DATA_DIR / "combined_comprehensive_dataset.csv"
        PREPROCESSED_DATASET = DATA_DIR / "preprocessed_combined_dataset.csv"

        RANDOM_STATE = 42

        print("ML directory:", ML_DIR)
        print("Data directory:", DATA_DIR)
        print("Combined dataset:", COMBINED_DATASET)
        """
    ),
    md_cell(
        """
        ## 1. Dataset Collection Overview

        The project combines several raw and intermediate data sources before training the final classifiers.
        """
    ),
    code_cell(
        """
        dataset_candidates = [
            RAW_DIR / "kathmandu_cafes.csv",
            RAW_DIR / "osm_roads_kathmandu.csv",
            RAW_DIR / "kathmandu_census.csv",
            RAW_DIR / "kathmandu_education_cleaned.csv",
            RAW_DIR / "combined_amenities_clean.csv",
            DATA_DIR / "combined_comprehensive_dataset.csv",
            DATA_DIR / "preprocessed_combined_dataset.csv",
        ]

        inventory_rows = []
        for path in dataset_candidates:
            if path.exists():
                df = pd.read_csv(path)
                inventory_rows.append({
                    "file_name": path.name,
                    "rows": len(df),
                    "columns": len(df.columns),
                    "sample_columns": ", ".join(df.columns[:6]),
                })

        inventory_df = pd.DataFrame(inventory_rows)
        display(inventory_df)
        """
    ),
    code_cell(
        """
        def summarize_dataset(path: Path, max_cols: int = 12):
            df = pd.read_csv(path)
            print(f"Dataset: {path.name}")
            print(f"Shape: {df.shape}")
            print("Columns:")
            for col in df.columns[:max_cols]:
                print(f" - {col}")
            if len(df.columns) > max_cols:
                print(f" ... and {len(df.columns) - max_cols} more columns")
            display(df.head(2))
            print("-" * 80)

        for path in dataset_candidates[:5]:
            if path.exists():
                summarize_dataset(path, max_cols=10)
        """
    ),
    md_cell(
        """
        ## 2. Load the Combined Dataset

        This is the main dataset used for the classification pipeline.
        """
    ),
    code_cell(
        """
        df = pd.read_csv(COMBINED_DATASET)
        print("Combined dataset shape:", df.shape)
        display(df.head())

        print("Columns in combined dataset:")
        for col in df.columns:
            print(" -", col)
        """
    ),
    code_cell(
        """
        target_counts = df["suitability"].fillna("Missing").value_counts(dropna=False).rename_axis("suitability").reset_index(name="count")
        display(target_counts)

        plt.figure(figsize=(8, 5))
        plot_target_df = df.copy()
        plot_target_df["suitability_plot"] = plot_target_df["suitability"].fillna("Missing")
        sns.countplot(
            data=plot_target_df,
            x="suitability_plot",
            order=plot_target_df["suitability_plot"].value_counts().index,
            palette="viridis",
        )
        plt.title("Suitability Class Distribution in Combined Dataset")
        plt.xlabel("Suitability Class")
        plt.ylabel("Count")
        plt.show()
        """
    ),
    md_cell(
        """
        ## 3. Dataset Processing

        Steps performed here:

        - remove identifier and metadata columns from the feature matrix
        - handle missing values
        - encode the suitability class labels
        - standardize numeric features
        - save the processed dataset
        """
    ),
    code_cell(
        """
        exclude_cols = [
            "place_id",
            "name",
            "lat",
            "lng",
            "type",
            "source",
            "suitability",
            "rating",
            "review_count",
            "price_level",
        ]

        feature_cols = [col for col in df.columns if col not in exclude_cols]
        clean_df = df.dropna(subset=["suitability"]).copy()
        X = clean_df[feature_cols].copy()
        y = clean_df["suitability"].copy()

        print("Feature columns used for training:")
        for col in feature_cols:
            print(" -", col)

        print("\\nRows before dropping missing suitability:", len(df))
        print("Rows after dropping missing suitability:", len(clean_df))

        missing_summary = X.isna().sum()
        print("\\nTotal missing feature values:", int(missing_summary.sum()))
        display(missing_summary[missing_summary > 0].to_frame("missing_count"))

        if missing_summary.sum() > 0:
            X = X.fillna(X.mean(numeric_only=True))

        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled_df = pd.DataFrame(X_scaled, columns=feature_cols)

        preprocessed_df = X_scaled_df.copy()
        preprocessed_df["suitability"] = y_encoded
        preprocessed_df.to_csv(PREPROCESSED_DATASET, index=False)

        print("\\nEncoded classes:", dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))))
        print("Preprocessed dataset saved to:", PREPROCESSED_DATASET)
        display(preprocessed_df.head())
        """
    ),
    md_cell(
        """
        ## 4. Split the Dataset into 80:20 and 85:15
        """
    ),
    code_cell(
        """
        split_configs = {
            "v2_80_20": 0.20,
            "v3_85_15": 0.15,
        }

        split_data = {}
        split_rows = []

        for split_name, test_size in split_configs.items():
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled_df,
                y_encoded,
                test_size=test_size,
                random_state=RANDOM_STATE,
                stratify=y_encoded,
            )

            split_data[split_name] = {
                "X_train": X_train,
                "X_test": X_test,
                "y_train": y_train,
                "y_test": y_test,
            }

            split_rows.append({
                "split_name": split_name,
                "train_ratio": round(1 - test_size, 2),
                "test_ratio": round(test_size, 2),
                "train_samples": len(X_train),
                "test_samples": len(X_test),
            })

        split_summary_df = pd.DataFrame(split_rows)
        display(split_summary_df)
        """
    ),
    md_cell(
        """
        ## 5. Train Random Forest and XGBoost on Each Split
        """
    ),
    code_cell(
        """
        from sklearn.preprocessing import StandardScaler, LabelEncoder
        from sklearn.model_selection import train_test_split, cross_val_score
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import (
            accuracy_score,
            precision_score,
            recall_score,
            f1_score,
            confusion_matrix,
            classification_report,
        )
        from xgboost import XGBClassifier

        split_data = globals().get("split_data")
        label_encoder = globals().get("label_encoder")
        feature_cols = globals().get("feature_cols")

        if split_data is None or label_encoder is None or feature_cols is None:
            print("Rebuilding preprocessing state for training section...")
            training_df = pd.read_csv(COMBINED_DATASET)
            exclude_cols = [
                "place_id",
                "name",
                "lat",
                "lng",
                "type",
                "source",
                "suitability",
                "rating",
                "review_count",
                "price_level",
            ]
            feature_cols = [col for col in training_df.columns if col not in exclude_cols]
            clean_df = training_df.dropna(subset=["suitability"]).copy()
            X = clean_df[feature_cols].copy().fillna(clean_df[feature_cols].mean(numeric_only=True))
            y = clean_df["suitability"].copy()

            label_encoder = LabelEncoder()
            y_encoded = label_encoder.fit_transform(y)

            scaler = StandardScaler()
            X_scaled_df = pd.DataFrame(scaler.fit_transform(X), columns=feature_cols)

            split_configs = {
                "v2_80_20": 0.20,
                "v3_85_15": 0.15,
            }
            split_data = {}
            for split_name, test_size in split_configs.items():
                X_train, X_test, y_train, y_test = train_test_split(
                    X_scaled_df,
                    y_encoded,
                    test_size=test_size,
                    random_state=RANDOM_STATE,
                    stratify=y_encoded,
                )
                split_data[split_name] = {
                    "X_train": X_train,
                    "X_test": X_test,
                    "y_train": y_train,
                    "y_test": y_test,
                }

        def evaluate_classifier(model_name, model, split_name, X_train, X_test, y_train, y_test):
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            all_labels = np.arange(len(label_encoder.classes_))
            min_class_count = int(pd.Series(y_train).value_counts().min())
            cv_folds = max(2, min(5, min_class_count))

            metrics = {
                "split_name": split_name,
                "model": model_name,
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
                "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
                "f1_score": f1_score(y_test, y_pred, average="weighted", zero_division=0),
                "cv_score": cross_val_score(model, X_train, y_train, cv=cv_folds).mean(),
                "cv_folds_used": cv_folds,
            }

            details = {
                "confusion_matrix": confusion_matrix(y_test, y_pred, labels=all_labels),
                "classification_report": classification_report(
                    y_test,
                    y_pred,
                    labels=all_labels,
                    target_names=label_encoder.classes_,
                    output_dict=True,
                    zero_division=0,
                ),
            }

            return model, metrics, details

        comparison_rows = []
        detailed_results = {}
        trained_models = {}

        for split_name, split_values in split_data.items():
            X_train = split_values["X_train"]
            X_test = split_values["X_test"]
            y_train = split_values["y_train"]
            y_test = split_values["y_test"]

            rf_model = RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=RANDOM_STATE,
                n_jobs=1,
            )

            xgb_model = XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=RANDOM_STATE,
                n_jobs=1,
                eval_metric="logloss" if len(label_encoder.classes_) == 2 else "mlogloss",
            )

            for model_name, model in [("Random Forest", rf_model), ("XGBoost", xgb_model)]:
                trained_model, metrics, details = evaluate_classifier(
                    model_name,
                    model,
                    split_name,
                    X_train,
                    X_test,
                    y_train,
                    y_test,
                )
                comparison_rows.append(metrics)
                detailed_results[(split_name, model_name)] = details
                trained_models[(split_name, model_name)] = trained_model

        comparison_df = pd.DataFrame(comparison_rows).sort_values(
            by=["accuracy", "f1_score", "cv_score"], ascending=False
        ).reset_index(drop=True)

        display(comparison_df)
        """
    ),
    code_cell(
        """
        comparison_output = MODELS_DIR / "notebook_model_comparison.csv"
        comparison_df.to_csv(comparison_output, index=False)
        print("Comparison table saved to:", comparison_output)

        report_output = MODELS_DIR / "notebook_model_comparison.json"
        label_mapping = {
            str(label): int(code)
            for label, code in zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))
        }
        with open(report_output, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "feature_columns": feature_cols,
                    "label_mapping": label_mapping,
                    "results": [
                        {
                            key: (float(value) if isinstance(value, (np.floating, np.integer)) else value)
                            for key, value in row.items()
                        }
                        for row in comparison_rows
                    ],
                },
                f,
                indent=2,
            )
        print("JSON summary saved to:", report_output)
        """
    ),
    md_cell(
        """
        ## 6. Tabular Performance Comparison
        """
    ),
    code_cell(
        """
        metric_columns = ["accuracy", "precision", "recall", "f1_score", "cv_score"]
        display(comparison_df.copy().round({col: 4 for col in metric_columns}))
        """
    ),
    md_cell(
        """
        ## 7. Graphical Performance Comparison
        """
    ),
    code_cell(
        """
        plot_df = comparison_df.copy()
        plot_df["label"] = plot_df["split_name"] + " | " + plot_df["model"]

        melted_df = plot_df.melt(
            id_vars=["label"],
            value_vars=["accuracy", "precision", "recall", "f1_score", "cv_score"],
            var_name="metric",
            value_name="score",
        )

        plt.figure(figsize=(14, 7))
        sns.barplot(data=melted_df, x="metric", y="score", hue="label", palette="Set2")
        plt.title("Model Performance Comparison Across Splits")
        plt.xlabel("Metric")
        plt.ylabel("Score")
        plt.ylim(0, 1.05)
        plt.legend(title="Split | Model", bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.tight_layout()
        plt.show()
        """
    ),
    code_cell(
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 11))
        axes = axes.flatten()

        ordered_items = [
            ("v2_80_20", "Random Forest"),
            ("v2_80_20", "XGBoost"),
            ("v3_85_15", "Random Forest"),
            ("v3_85_15", "XGBoost"),
        ]

        for ax, key in zip(axes, ordered_items):
            split_name, model_name = key
            cm = detailed_results[key]["confusion_matrix"]
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_,
                ax=ax,
            )
            ax.set_title(f"{split_name} - {model_name}")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")

        plt.tight_layout()
        plt.show()
        """
    ),
    md_cell(
        """
        ## 8. Feature Importance (Random Forest Example)
        """
    ),
    code_cell(
        """
        best_rf_key = None
        for key in [("v3_85_15", "Random Forest"), ("v2_80_20", "Random Forest")]:
            if key in trained_models:
                best_rf_key = key
                break

        if best_rf_key is not None:
            rf_model = trained_models[best_rf_key]
            importance_df = pd.DataFrame({
                "feature": feature_cols,
                "importance": rf_model.feature_importances_,
            }).sort_values("importance", ascending=False)

            display(importance_df.head(10))

            plt.figure(figsize=(10, 6))
            sns.barplot(data=importance_df.head(10), x="importance", y="feature", palette="crest")
            plt.title(f"Top 10 Feature Importances - {best_rf_key[0]} Random Forest")
            plt.xlabel("Importance")
            plt.ylabel("Feature")
            plt.tight_layout()
            plt.show()
        """
    ),
    md_cell(
        """
        ## 9. Conclusion for the ML Pipelines
        """
    ),
    code_cell(
        """
        best_model_row = comparison_df.sort_values(
            by=["accuracy", "f1_score", "cv_score"], ascending=False
        ).iloc[0]

        best_model_name = best_model_row["model"]
        best_split_name = best_model_row["split_name"]

        conclusion_text = f'''
        ### Conclusion

        - The notebook successfully documented dataset collection, preprocessing, model training, evaluation, and comparison for the combined dataset pipeline.
        - Two split strategies were tested: **80:20** and **85:15**.
        - Two models were trained on each split: **Random Forest** and **XGBoost**.
        - Based on the comparison table, the best-performing setup in this notebook is **{best_model_name}** on the **{best_split_name}** split.
        - The final choice should consider not only accuracy, but also precision, recall, F1-score, and cross-validation consistency.
        - This pipeline provides a clear experimental basis for reporting which split and model combination works best for the processed combined cafe-location dataset.
        '''

        display(Markdown(conclusion_text))
        """
    ),
]


notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.11",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


output_path = Path(__file__).with_name("combined_dataset_ml_pipeline.ipynb")
with output_path.open("w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)
    f.write("\n")

print(output_path)
