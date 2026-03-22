"""
Train and Compare XGBoost vs Random Forest for Cafe Location Suitability Prediction
This script trains both models, evaluates them, and provides detailed performance comparison.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score, 
                             precision_recall_fscore_support, roc_auc_score, roc_curve)
import xgboost as xgb
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def load_training_data():
    """
    Load the real training data for location suitability prediction
    """
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'cafe_location_training_dataset.csv')

    if not os.path.exists(data_path):
        print(f"Training data not found at {data_path}")
        raise FileNotFoundError(f"Data file not found: {data_path}")

    print(f"Loading training data from {data_path}")
    df = pd.read_csv(data_path)

    print(f"✓ Training data shape: {df.shape}")
    print(f"✓ Columns: {list(df.columns)}")
    print(f"✓ Suitability distribution:\n{df['suitability'].value_counts()}")

    return df

def prepare_data():
    """
    Load and prepare data for model training
    """
    df = load_training_data()

    feature_cols = [
        'competitors_within_500m', 'competitors_within_200m', 'competitors_min_distance', 'competitors_avg_distance',
        'roads_within_500m', 'roads_avg_distance',
        'schools_within_500m', 'schools_within_200m', 'schools_min_distance',
        'hospitals_within_500m', 'hospitals_min_distance',
        'bus_stops_within_500m', 'bus_stops_min_distance',
        'population_density_proxy', 'accessibility_score', 'foot_traffic_score', 'competition_pressure'
    ]

    X = df[feature_cols]
    y = df['suitability']

    # Encode target labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    print("\n✓ Label mapping:")
    for i, label in enumerate(label_encoder.classes_):
        print(f"  {i}: {label}")

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    print(f"\n✓ Data split: Train={X_train.shape[0]}, Test={X_test.shape[0]}")

    return X_train, X_test, y_train, y_test, label_encoder, scaler, feature_cols

def train_random_forest(X_train, y_train, X_test, y_test, label_encoder):
    """
    Train Random Forest model with optimized hyperparameters
    """
    print("\n" + "="*70)
    print("TRAINING RANDOM FOREST MODEL")
    print("="*70)

    # Train Random Forest with optimized parameters
    rf_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        bootstrap=True,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
        verbose=1
    )

    print("\n→ Training Random Forest...")
    rf_model.fit(X_train, y_train)

    # Cross-validation
    cv_scores = cross_val_score(rf_model, X_train, y_train, cv=5, n_jobs=-1)
    print(f"✓ 5-fold Cross-validation accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

    # Test set evaluation
    y_pred_rf = rf_model.predict(X_test)
    y_pred_proba_rf = rf_model.predict_proba(X_test)
    
    accuracy_rf = accuracy_score(y_test, y_pred_rf)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred_rf, average='weighted')

    print(f"\n✓ Random Forest Test Set Accuracy: {accuracy_rf:.4f}")
    print(f"✓ Precision: {precision:.4f}")
    print(f"✓ Recall: {recall:.4f}")
    print(f"✓ F1-Score: {f1:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_rf, target_names=label_encoder.classes_))

    print("\nConfusion Matrix:")
    cm_rf = confusion_matrix(y_test, y_pred_rf)
    print(cm_rf)

    return rf_model, y_pred_rf, y_pred_proba_rf, accuracy_rf, cm_rf

def train_xgboost(X_train, y_train, X_test, y_test, label_encoder):
    """
    Train XGBoost model with optimized hyperparameters
    """
    print("\n" + "="*70)
    print("TRAINING XGBOOST MODEL")
    print("="*70)

    # Train XGBoost with optimized parameters
    xgb_model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='multi:softprob',  # For multiclass classification
        num_class=len(np.unique(y_train)),
        random_state=42,
        n_jobs=-1,
        verbosity=1,
        eval_metric='mlogloss'  # Use multiclass log loss
    )

    print("\n→ Training XGBoost...")
    xgb_model.fit(X_train, y_train)

    # Cross-validation
    cv_scores = cross_val_score(xgb_model, X_train, y_train, cv=5, n_jobs=-1)
    print(f"✓ 5-fold Cross-validation accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

    # Test set evaluation
    y_pred_xgb = xgb_model.predict(X_test)
    y_pred_proba_xgb = xgb_model.predict_proba(X_test)
    
    accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred_xgb, average='weighted')

    print(f"\n✓ XGBoost Test Set Accuracy: {accuracy_xgb:.4f}")
    print(f"✓ Precision: {precision:.4f}")
    print(f"✓ Recall: {recall:.4f}")
    print(f"✓ F1-Score: {f1:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_xgb, target_names=label_encoder.classes_))

    print("\nConfusion Matrix:")
    cm_xgb = confusion_matrix(y_test, y_pred_xgb)
    print(cm_xgb)

    return xgb_model, y_pred_xgb, y_pred_proba_xgb, accuracy_xgb, cm_xgb

def compare_models(accuracy_rf, accuracy_xgb, y_test, y_pred_rf, y_pred_xgb, 
                   cm_rf, cm_xgb, label_encoder):
    """
    Compare the performance of both models
    """
    print("\n" + "="*70)
    print("MODEL COMPARISON RESULTS")
    print("="*70)

    improvement = ((accuracy_xgb - accuracy_rf) / accuracy_rf) * 100
    
    print(f"\n┌─ ACCURACY COMPARISON ─────────────────────────────────────┐")
    print(f"│ Random Forest Accuracy:  {accuracy_rf:.4f} ({accuracy_rf*100:.2f}%)")
    print(f"│ XGBoost Accuracy:        {accuracy_xgb:.4f} ({accuracy_xgb*100:.2f}%)")
    print(f"├────────────────────────────────────────────────────────────┤")
    print(f"│ Difference:              {abs(accuracy_xgb - accuracy_rf):.4f}")
    
    if accuracy_xgb > accuracy_rf:
        print(f"│ Winner:                  XGBoost (+{improvement:.2f}%)")
    else:
        print(f"│ Winner:                  Random Forest (+{abs(improvement):.2f}%)")
    print(f"└────────────────────────────────────────────────────────────┘")

    # Per-class metrics
    print(f"\n┌─ PER-CLASS ACCURACIES ────────────────────────────────────┐")
    
    for i, label in enumerate(label_encoder.classes_):
        rf_correct = np.sum((y_test == i) & (y_pred_rf == i))
        rf_total = np.sum(y_test == i)
        rf_accuracy = rf_correct / rf_total if rf_total > 0 else 0

        xgb_correct = np.sum((y_test == i) & (y_pred_xgb == i))
        xgb_total = np.sum(y_test == i)
        xgb_accuracy = xgb_correct / xgb_total if xgb_total > 0 else 0

        print(f"│ {label:20s} | RF: {rf_accuracy:.4f}  | XGB: {xgb_accuracy:.4f}")
    
    print(f"└────────────────────────────────────────────────────────────┘")

    # Error analysis
    print(f"\n┌─ ERROR ANALYSIS ──────────────────────────────────────────┐")
    rf_errors = np.sum(y_pred_rf != y_test)
    xgb_errors = np.sum(y_pred_xgb != y_test)
    
    print(f"│ Random Forest Errors:    {rf_errors}/{len(y_test)} ({rf_errors/len(y_test)*100:.2f}%)")
    print(f"│ XGBoost Errors:          {xgb_errors}/{len(y_test)} ({xgb_errors/len(y_test)*100:.2f}%)")
    print(f"│ Error Reduction:         {rf_errors - xgb_errors}")
    print(f"└────────────────────────────────────────────────────────────┘")

    return improvement

def save_models(rf_model, xgb_model, label_encoder, scaler, feature_cols):
    """
    Save trained models and preprocessing objects
    """
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    os.makedirs(models_dir, exist_ok=True)

    # Save Random Forest
    rf_path = os.path.join(models_dir, 'suitability_rf_model.pkl')
    joblib.dump(rf_model, rf_path)
    print(f"\n✓ Random Forest model saved to: {rf_path}")

    # Save XGBoost
    xgb_path = os.path.join(models_dir, 'suitability_xgb_model.pkl')
    joblib.dump(xgb_model, xgb_path)
    print(f"✓ XGBoost model saved to: {xgb_path}")

    # Save encoder and scaler
    encoder_path = os.path.join(models_dir, 'suitability_label_encoder.pkl')
    scaler_path = os.path.join(models_dir, 'suitability_scaler.pkl')
    features_path = os.path.join(models_dir, 'feature_names.pkl')

    joblib.dump(label_encoder, encoder_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(feature_cols, features_path)

    print(f"✓ Label encoder saved to: {encoder_path}")
    print(f"✓ Scaler saved to: {scaler_path}")
    print(f"✓ Feature names saved to: {features_path}")

def generate_comparison_report(accuracy_rf, accuracy_xgb, improvement):
    """
    Generate a detailed comparison report
    """
    report_file = os.path.join(os.path.dirname(__file__), 
                               f'models/comparison_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
    
    os.makedirs(os.path.dirname(report_file), exist_ok=True)

    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("XGBOOST vs RANDOM FOREST COMPARISON REPORT\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("ACCURACY RESULTS\n")
        f.write("-" * 70 + "\n")
        f.write(f"Random Forest Accuracy:  {accuracy_rf:.4f} ({accuracy_rf*100:.2f}%)\n")
        f.write(f"XGBoost Accuracy:        {accuracy_xgb:.4f} ({accuracy_xgb*100:.2f}%)\n")
        f.write(f"Difference:              {abs(accuracy_xgb - accuracy_rf):.4f}\n")
        f.write(f"Improvement:             {improvement:+.2f}%\n\n")
        
        if accuracy_xgb > accuracy_rf:
            f.write(f"WINNER: XGBoost (Better by {improvement:.2f}%)\n\n")
        else:
            f.write(f"WINNER: Random Forest (Better by {abs(improvement):.2f}%)\n\n")
        
        f.write("RECOMMENDATIONS\n")
        f.write("-" * 70 + "\n")
        if accuracy_xgb > accuracy_rf:
            f.write("✓ Deploy XGBoost model for production\n")
            f.write("✓ XGBoost provides better accuracy for cafe location suitability\n")
        else:
            f.write("✓ Continue using Random Forest model\n")
            f.write("✓ Random Forest provides better accuracy for this dataset\n")
        
        f.write("\nMODEL FILES CREATED\n")
        f.write("-" * 70 + "\n")
        f.write("✓ suitability_rf_model.pkl - Random Forest model\n")
        f.write("✓ suitability_xgb_model.pkl - XGBoost model\n")
        f.write("✓ suitability_label_encoder.pkl - Label encoder\n")
        f.write("✓ suitability_scaler.pkl - Feature scaler\n")
        f.write("✓ feature_names.pkl - Feature names\n")

    print(f"\n✓ Comparison report saved to: {report_file}")

def main():
    """
    Main training and comparison pipeline
    """
    print("\n" + "="*70)
    print("XGBOOST vs RANDOM FOREST COMPARISON")
    print("="*70)

    # Prepare data
    print("\nPREPARING DATA")
    print("-" * 70)
    X_train, X_test, y_train, y_test, label_encoder, scaler, feature_cols = prepare_data()

    # Train Random Forest
    rf_model, y_pred_rf, y_pred_proba_rf, accuracy_rf, cm_rf = train_random_forest(
        X_train, y_train, X_test, y_test, label_encoder
    )

    # Train XGBoost
    xgb_model, y_pred_xgb, y_pred_proba_xgb, accuracy_xgb, cm_xgb = train_xgboost(
        X_train, y_train, X_test, y_test, label_encoder
    )

    # Compare models
    improvement = compare_models(accuracy_rf, accuracy_xgb, y_test, y_pred_rf, y_pred_xgb, 
                                 cm_rf, cm_xgb, label_encoder)

    # Save models
    save_models(rf_model, xgb_model, label_encoder, scaler, feature_cols)

    # Generate report
    generate_comparison_report(accuracy_rf, accuracy_xgb, improvement)

    print("\n" + "="*70)
    print("✓ TRAINING AND COMPARISON COMPLETED SUCCESSFULLY")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
