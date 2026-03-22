"""
Compare the performance of the original vs optimized Random Forest models
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

def compare_models():
    """
    Compare original and optimized model performance
    """
    print("Comparing Original vs Optimized Random Forest Models")
    print("=" * 60)

    # Load the training data
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'cafe_location_training_dataset.csv')
    df = pd.read_csv(data_path)

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

    # Load encoders and scalers
    models_dir = os.path.join(os.path.dirname(__file__), 'models')

    # Original model
    try:
        original_model = joblib.load(os.path.join(models_dir, 'suitability_rf_model.pkl'))
        original_encoder = joblib.load(os.path.join(models_dir, 'suitability_label_encoder.pkl'))
        original_scaler = joblib.load(os.path.join(models_dir, 'suitability_scaler.pkl'))
        print("✓ Original model loaded")
    except:
        print("✗ Original model not found")
        return

    # Optimized model
    try:
        optimized_model = joblib.load(os.path.join(models_dir, 'suitability_rf_model_optimized.pkl'))
        optimized_encoder = joblib.load(os.path.join(models_dir, 'suitability_label_encoder.pkl'))
        optimized_scaler = joblib.load(os.path.join(models_dir, 'suitability_scaler.pkl'))
        features_info = joblib.load(os.path.join(models_dir, 'selected_features.pkl'))
        selected_features = features_info['selected_features']
        print("✓ Optimized model loaded")
    except:
        print("✗ Optimized model not found")
        return

    # Load best hyperparameters
    try:
        best_params = joblib.load(os.path.join(models_dir, 'best_hyperparameters.pkl'))
        print(f"✓ Best hyperparameters: {best_params}")
    except:
        best_params = {}
        print("✗ Best hyperparameters not found")

    # Split data (same as training)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Encode labels
    y_test_encoded = original_encoder.transform(y_test)

    # Test Original Model
    print("\n" + "="*60)
    print("ORIGINAL MODEL PERFORMANCE")
    print("="*60)

    X_test_scaled_original = original_scaler.transform(X_test)
    y_pred_original = original_model.predict(X_test_scaled_original)
    accuracy_original = accuracy_score(y_test_encoded, y_pred_original)

    print(".4f")
    print("\nClassification Report:")
    print(classification_report(y_test_encoded, y_pred_original, target_names=original_encoder.classes_))

    # Test Optimized Model
    print("\n" + "="*60)
    print("OPTIMIZED MODEL PERFORMANCE")
    print("="*60)

    # Scale all features first, then select the optimized features
    X_test_scaled_all = optimized_scaler.transform(X_test)
    selected_indices = [feature_cols.index(feat) for feat in selected_features]
    X_test_scaled_optimized = X_test_scaled_all[:, selected_indices]

    y_pred_optimized = optimized_model.predict(X_test_scaled_optimized)
    accuracy_optimized = accuracy_score(y_test_encoded, y_pred_optimized)

    print(".4f")
    print("\nClassification Report:")
    print(classification_report(y_test_encoded, y_pred_optimized, target_names=optimized_encoder.classes_))

    # Comparison Summary
    print("\n" + "="*60)
    print("MODEL COMPARISON SUMMARY")
    print("="*60)
    print(".4f")
    print(".4f")
    improvement = accuracy_optimized - accuracy_original
    print(".4f")

    if improvement > 0:
        print("🎉 OPTIMIZED MODEL SHOWS IMPROVEMENT!")
    elif improvement == 0:
        print("🤝 MODELS PERFORM SIMILARLY")
    else:
        print("⚠️  OPTIMIZED MODEL PERFORMED WORSE")

    print(f"\nOptimized model uses {len(selected_features)} features vs {len(feature_cols)} in original")
    print(f"Feature reduction: {len(feature_cols) - len(selected_features)} features removed")

    # Show top features
    print("\nTop 5 selected features:")
    feature_importance = pd.DataFrame({
        'feature': selected_features,
        'importance': optimized_model.feature_importances_
    }).sort_values('importance', ascending=False)

    for i, row in feature_importance.head().iterrows():
        print(".4f")

if __name__ == "__main__":
    compare_models()