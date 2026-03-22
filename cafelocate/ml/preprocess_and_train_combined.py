"""
Comprehensive Data Preprocessing & Model Training
Handles combined dataset with multiple train-test splits (80-20 and 85-15)
Versions models as v1, v2, v3 based on split strategy
"""

import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
from pathlib import Path

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import xgboost as xgb
import joblib

print("=" * 100)
print("COMBINED DATASET PREPROCESSING & MODEL TRAINING")
print("=" * 100)

# Step 1: Load combined dataset
print("\n[Step 1] Loading combined dataset...")
data_path = "../data/combined_comprehensive_dataset.csv"
df = pd.read_csv(data_path)

print(f"✓ Dataset loaded: {df.shape}")
print(f"  Columns: {len(df.columns)}")
print(f"  Target variable: suitability")

# Step 2: Data preprocessing
print("\n[Step 2] Preprocessing data...")

# Identify feature columns (exclude location identifiers and target)
exclude_cols = ['place_id', 'name', 'lat', 'lng', 'type', 'source', 'suitability', 'rating', 'review_count', 'price_level']
feature_cols = [col for col in df.columns if col not in exclude_cols]

print(f"  Total columns: {len(df.columns)}")
print(f"  Excluded (non-features): {len(exclude_cols)}")
print(f"  Features to use: {len(feature_cols)}")
print(f"  Feature columns: {feature_cols}")

# Prepare features and target
X = df[feature_cols].copy()
y = df['suitability'].copy()

# Handle missing values
print(f"\n  Missing values before handling:")
missing = X.isnull().sum()
print(f"    Total missing: {missing.sum()}")
if missing.sum() > 0:
    print(f"    Columns with missing: {missing[missing > 0].to_dict()}")
    X = X.fillna(X.mean())  # Fill with mean
    print(f"    Filled with mean values")

if y.isnull().sum() > 0:
    valid_idx = ~y.isnull()
    X = X[valid_idx]
    y = y[valid_idx]
    print(f"    Removed {y.isnull().sum()} rows with missing target")

print(f"  Data cleaned: X {X.shape}, y {y.shape}")

# Standardize features
print(f"\n  Scaling features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=feature_cols)

print(f"  ✓ Features scaled (mean=0, std=1)")

# Encode target variable if string
print(f"\n  Encoding target variable...")
le = LabelEncoder()
y_encoded = le.fit_transform(y)
print(f"  ✓ Target encoded: {len(le.classes_)} classes - {list(le.classes_)}")

# Ensure y_encoded is numeric (numpy array)
y_encoded = np.array(y_encoded, dtype=np.int64)

print(f"\n[Step 2] Preprocessing complete!")
print(f"  Final features: {X_scaled.shape[1]}")
print(f"  Final samples: {X_scaled.shape[0]}")
print(f"  Target distribution:")
unique, counts = np.unique(y_encoded, return_counts=True)
for u, c in zip(unique, counts):
    class_label = le.classes_[u] if u < len(le.classes_) else str(u)
    print(f"    Class {u} ({class_label}): {c} ({c/len(y_encoded)*100:.1f}%)")

# Step 3: Create train-test splits with different ratios
print("\n" + "=" * 100)
print("CREATING TRAIN-TEST SPLITS")
print("=" * 100)

splits = {
    'v2_80_20': {'test_size': 0.20, 'train_size': 0.80},
    'v3_85_15': {'test_size': 0.15, 'train_size': 0.85}
}

split_data = {}

for split_name, split_config in splits.items():
    print(f"\n[{split_name.upper()}]")
    test_size = split_config['test_size']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded,
        test_size=test_size,
        random_state=42,
        stratify=y_encoded
    )
    
    split_data[split_name] = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'scaler': scaler,
        'feature_cols': feature_cols
    }
    
    print(f"  Train split: {len(X_train)} samples ({len(X_train)/len(X_scaled)*100:.1f}%)")
    print(f"  Test split:  {len(X_test)} samples ({len(X_test)/len(X_scaled)*100:.1f}%)")
    print(f"  Total samples: {len(X_scaled)}")

# Step 4: Train models with both splits
print("\n" + "=" * 100)
print("TRAINING MODELS WITH DIFFERENT SPLITS")
print("=" * 100)

models_dir = Path("models")
models_dir.mkdir(exist_ok=True)
results = {}

for split_name, data in split_data.items():
    print(f"\n{'='*100}")
    print(f"TRAINING MODELS FOR: {split_name}")
    print(f"{'='*100}")
    
    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']
    
    results[split_name] = {}
    
    # Train Random Forest
    print(f"\n[Algo 1] Training Random Forest ({split_name})...")
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    
    rf_metrics = {
        'accuracy': accuracy_score(y_test, rf_pred),
        'precision': precision_score(y_test, rf_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_test, rf_pred, average='weighted', zero_division=0),
        'f1': f1_score(y_test, rf_pred, average='weighted', zero_division=0),
        'cv_score': cross_val_score(rf_model, X_train, y_train, cv=5).mean()
    }
    
    print(f"  ✓ Random Forest trained")
    print(f"    - Accuracy: {rf_metrics['accuracy']:.4f}")
    print(f"    - Precision: {rf_metrics['precision']:.4f}")
    print(f"    - Recall: {rf_metrics['recall']:.4f}")
    print(f"    - F1-Score: {rf_metrics['f1']:.4f}")
    print(f"    - CV Score (5-fold): {rf_metrics['cv_score']:.4f}")
    
    # Save RF model
    rf_model_path = models_dir / f"rf_model_{split_name}.pkl"
    joblib.dump(rf_model, rf_model_path)
    print(f"  ✓ Saved: {rf_model_path}")
    
    results[split_name]['random_forest'] = rf_metrics
    
    # Train XGBoost
    print(f"\n[Algo 2] Training XGBoost ({split_name})...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        eval_metric='multi_logloss' if len(np.unique(y_train)) > 2 else 'logloss'
    )
    
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_test)
    
    xgb_metrics = {
        'accuracy': accuracy_score(y_test, xgb_pred),
        'precision': precision_score(y_test, xgb_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_test, xgb_pred, average='weighted', zero_division=0),
        'f1': f1_score(y_test, xgb_pred, average='weighted', zero_division=0),
        'cv_score': cross_val_score(xgb_model, X_train, y_train, cv=5).mean()
    }
    
    print(f"  ✓ XGBoost trained")
    print(f"    - Accuracy: {xgb_metrics['accuracy']:.4f}")
    print(f"    - Precision: {xgb_metrics['precision']:.4f}")
    print(f"    - Recall: {xgb_metrics['recall']:.4f}")
    print(f"    - F1-Score: {xgb_metrics['f1']:.4f}")
    print(f"    - CV Score (5-fold): {xgb_metrics['cv_score']:.4f}")
    
    # Save XGBoost model
    xgb_model_path = models_dir / f"xgb_model_{split_name}.pkl"
    joblib.dump(xgb_model, xgb_model_path)
    print(f"  ✓ Saved: {xgb_model_path}")
    
    results[split_name]['xgboost'] = xgb_metrics
    
    # Compare
    print(f"\n[Comparison] {split_name}")
    print(f"  Random Forest Accuracy: {rf_metrics['accuracy']:.4f}")
    print(f"  XGBoost Accuracy:       {xgb_metrics['accuracy']:.4f}")
    if xgb_metrics['accuracy'] >= rf_metrics['accuracy']:
        winner = "XGBoost ✓"
    else:
        winner = "Random Forest ✓"
    print(f"  Winner: {winner}")

# Step 5: Save preprocessed data and models
print(f"\n{'='*100}")
print("SAVING ARTIFACTS")
print(f"{'='*100}")

# Save scaler
scaler_path = models_dir / "scaler_combined.pkl"
joblib.dump(scaler, scaler_path)
print(f"✓ Scaler saved: {scaler_path}")

# Save feature columns
features_path = models_dir / "feature_columns.pkl"
joblib.dump(feature_cols, features_path)
print(f"✓ Feature columns saved: {features_path}")

# Save label encoder if needed
le_path = models_dir / "label_encoder_suitability.pkl"
if 'le' in locals():
    joblib.dump(le, le_path)
    print(f"✓ Label encoder saved: {le_path}")

# Save preprocessed dataset
preprocessed_path = Path("../data/preprocessed_combined_dataset.csv")
preprocessed_df = X_scaled.copy()
preprocessed_df['suitability'] = y_encoded
preprocessed_df.to_csv(preprocessed_path, index=False)
print(f"✓ Preprocessed dataset saved: {preprocessed_path}")

# Step 6: Generate comprehensive report
print(f"\n{'='*100}")
print("GENERATING REPORT")
print(f"{'='*100}")

report = {
    'timestamp': datetime.now().isoformat(),
    'dataset': {
        'total_samples': len(df),
        'total_features': len(feature_cols),
        'feature_names': feature_cols,
        'target_classes': len(le.classes_),
        'missing_values_handled': True,
        'features_scaled': True
    },
    'splits': {
        'v2_80_20': {
            'train_ratio': 0.80,
            'test_ratio': 0.20,
            'train_samples': len(split_data['v2_80_20']['X_train']),
            'test_samples': len(split_data['v2_80_20']['X_test'])
        },
        'v3_85_15': {
            'train_ratio': 0.85,
            'test_ratio': 0.15,
            'train_samples': len(split_data['v3_85_15']['X_train']),
            'test_samples': len(split_data['v3_85_15']['X_test'])
        }
    },
    'models': results,
    'files_generated': {
        'preprocessed_data': str(preprocessed_path),
        'models': [
            'rf_model_v2_80_20.pkl',
            'xgb_model_v2_80_20.pkl',
            'rf_model_v3_85_15.pkl',
            'xgb_model_v3_85_15.pkl'
        ],
        'preprocessing': [
            'scaler_combined.pkl',
            'feature_columns.pkl',
            'label_encoder_suitability.pkl'
        ]
    }
}

# Save report as JSON
report_path = models_dir / "training_report.json"
with open(report_path, 'w') as f:
    json.dump(report, f, indent=2)
print(f"✓ Report saved: {report_path}")

# Generate summary table
print(f"\n{'='*100}")
print("PERFORMANCE COMPARISON SUMMARY")
print(f"{'='*100}\n")

comparison_data = []
for split_name, models_results in results.items():
    for algo_name, metrics in models_results.items():
        comparison_data.append({
            'Split': split_name,
            'Algorithm': algo_name,
            'Accuracy': f"{metrics['accuracy']:.4f}",
            'Precision': f"{metrics['precision']:.4f}",
            'Recall': f"{metrics['recall']:.4f}",
            'F1-Score': f"{metrics['f1']:.4f}"
        })

comparison_df = pd.DataFrame(comparison_data)
print(comparison_df.to_string(index=False))

# Save comparison
comparison_path = models_dir / "model_comparison.csv"
comparison_df.to_csv(comparison_path, index=False)
print(f"\n✓ Comparison saved: {comparison_path}")

print(f"\n{'='*100}")
print("✅ PREPROCESSING & TRAINING COMPLETED!")
print(f"{'='*100}")
print(f"\nModels created:")
print(f"  V2 (80-20 split):")
print(f"    - rf_model_v2_80_20.pkl")
print(f"    - xgb_model_v2_80_20.pkl")
print(f"\n  V3 (85-15 split):")
print(f"    - rf_model_v3_85_15.pkl")
print(f"    - xgb_model_v3_85_15.pkl")
print(f"\nPreprocessing artifacts:")
print(f"  - preprocessed_combined_dataset.csv")
print(f"  - scaler_combined.pkl")
print(f"  - feature_columns.pkl")
print(f"\nReports:")
print(f"  - training_report.json")
print(f"  - model_comparison.csv")
print(f"\nAll files saved in: cafelocate/ml/models/")
