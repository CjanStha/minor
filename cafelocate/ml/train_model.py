"""
Train ML model for cafe location suitability prediction
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_selection import SelectFromModel, RFE
import joblib
import os

def load_training_data():
    """
    Load the real training data for location suitability prediction
    """
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'cafe_location_training_dataset.csv')

    if not os.path.exists(data_path):
        print(f"Training data not found at {data_path}")
        print("Falling back to synthetic data generation...")
        return create_synthetic_training_data()

    print(f"Loading training data from {data_path}")
    df = pd.read_csv(data_path)

    print(f"Training data shape: {df.shape}")
    print("Columns:", list(df.columns))
    print("Suitability distribution:")
    print(df['suitability'].value_counts())

    return df

def create_synthetic_training_data():
    """
    Create synthetic training data for location suitability prediction
    Based on the features from the real training dataset
    """
    np.random.seed(42)

    # Create 1574 synthetic data points (same as real dataset)
    n_samples = 1574

    data = []

    for i in range(n_samples):
        # Generate features similar to real data
        competitors_within_500m = np.random.randint(0, 20)
        competitors_within_200m = np.random.randint(0, competitors_within_500m + 1)
        competitors_min_distance = np.random.uniform(25, 500)
        competitors_avg_distance = np.random.uniform(competitors_min_distance, 500)

        roads_within_500m = np.random.randint(0, 15)
        roads_avg_distance = np.random.uniform(50, 500)

        schools_within_500m = np.random.randint(0, 20)
        schools_within_200m = np.random.randint(0, schools_within_500m + 1)
        schools_min_distance = np.random.uniform(25, 500)

        hospitals_within_500m = np.random.randint(0, 5)
        hospitals_min_distance = np.random.uniform(100, 500)

        bus_stops_within_500m = np.random.randint(0, 10)
        bus_stops_min_distance = np.random.uniform(50, 500)

        population_density_proxy = np.random.uniform(0, 100)
        accessibility_score = np.random.uniform(0, 5)
        foot_traffic_score = np.random.uniform(0, 10)
        competition_pressure = np.random.uniform(0, 10)

        # Determine suitability based on features
        score = (
            (population_density_proxy * 0.2) +
            (accessibility_score * 0.15) +
            (foot_traffic_score * 0.15) +
            (schools_within_500m * 0.1) +
            (bus_stops_within_500m * 0.1) -
            (competition_pressure * 0.2) -
            (competitors_within_200m * 0.1)
        )

        if score > 15:
            suitability = 'High'
        elif score > 8:
            suitability = 'Medium'
        else:
            suitability = 'Low'

        data.append({
            'competitors_within_500m': competitors_within_500m,
            'competitors_within_200m': competitors_within_200m,
            'competitors_min_distance': competitors_min_distance,
            'competitors_avg_distance': competitors_avg_distance,
            'roads_within_500m': roads_within_500m,
            'roads_avg_distance': roads_avg_distance,
            'schools_within_500m': schools_within_500m,
            'schools_within_200m': schools_within_200m,
            'schools_min_distance': schools_min_distance,
            'hospitals_within_500m': hospitals_within_500m,
            'hospitals_min_distance': hospitals_min_distance,
            'bus_stops_within_500m': bus_stops_within_500m,
            'bus_stops_min_distance': bus_stops_min_distance,
            'population_density_proxy': population_density_proxy,
            'accessibility_score': accessibility_score,
            'foot_traffic_score': foot_traffic_score,
            'competition_pressure': competition_pressure,
            'suitability': suitability
        })

    return pd.DataFrame(data)

    return pd.DataFrame(data)

def hyperparameter_tuning(X_train, y_train):
    """
    Perform hyperparameter tuning for Random Forest using RandomizedSearchCV
    """
    print("Performing hyperparameter tuning...")

    # Define parameter grid for RandomizedSearchCV
    param_distributions = {
        'n_estimators': [100, 200, 300, 400, 500],
        'max_depth': [10, 15, 20, 25, None],
        'min_samples_split': [2, 5, 10, 15],
        'min_samples_leaf': [1, 2, 4, 6],
        'max_features': ['sqrt', 'log2', None],
        'bootstrap': [True, False],
        'class_weight': ['balanced', 'balanced_subsample', None]
    }

    # Initialize Random Forest
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)

    # Use RandomizedSearchCV for efficiency
    random_search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_distributions,
        n_iter=50,  # Number of parameter settings sampled
        cv=5,       # 5-fold cross-validation
        scoring='accuracy',
        n_jobs=-1,
        random_state=42,
        verbose=1
    )

    # Fit the random search
    random_search.fit(X_train, y_train)

    print(f"Best parameters found: {random_search.best_params_}")
    print(f"Best cross-validation score: {random_search.best_score_:.4f}")

    return random_search.best_estimator_, random_search.best_params_

def feature_selection(X_train, y_train, X_test, feature_cols):
    """
    Perform feature selection to improve model performance
    """
    print("Performing feature selection...")

    # Use Random Forest for feature importance
    rf_temp = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_temp.fit(X_train, y_train)

    # Get feature importances
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': rf_temp.feature_importances_
    }).sort_values('importance', ascending=False)

    print("Feature importances:")
    for idx, row in feature_importance.iterrows():
        print(".4f")

    # Select features with importance > threshold (e.g., > 0.01)
    threshold = 0.01
    selected_features = feature_importance[feature_importance['importance'] > threshold]['feature'].tolist()

    print(f"\nSelected {len(selected_features)} out of {len(feature_cols)} features")
    print("Selected features:", selected_features)

    # Get indices of selected features
    selected_indices = [feature_cols.index(feat) for feat in selected_features]

    return selected_indices, selected_features

def train_suitability_model():
    """
    Train and save the location suitability ML model with hyperparameter tuning
    """
    print("Loading training data...")
    df = load_training_data()

    # Prepare features and target
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

    print("\nLabel mapping:")
    for i, label in enumerate(label_encoder.classes_):
        print(f"  {i}: {label}")

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

    # Feature selection
    selected_indices, selected_features = feature_selection(X_train, y_train, X_test, feature_cols)

    # Apply feature selection
    X_train_selected = X_train[:, selected_indices]
    X_test_selected = X_test[:, selected_indices]

    # Hyperparameter tuning
    best_model, best_params = hyperparameter_tuning(X_train_selected, y_train)

    # Train final model with best parameters
    print("\nTraining final Random Forest model with optimized parameters...")
    model = RandomForestClassifier(**best_params, random_state=42, n_jobs=-1)
    model.fit(X_train_selected, y_train)

    # Cross-validation with best model
    cv_scores = cross_val_score(model, X_train_selected, y_train, cv=10)
    print(f"10-fold Cross-validation accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

    # Evaluate on test set
    y_pred = model.predict(X_test_selected)
    accuracy = accuracy_score(y_test, y_pred)
    print(".4f")

    print("\nModel Performance on Test Set:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    # Save model, encoder, scaler, and feature info
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    os.makedirs(models_dir, exist_ok=True)

    model_path = os.path.join(models_dir, 'suitability_rf_model_optimized.pkl')
    encoder_path = os.path.join(models_dir, 'suitability_label_encoder.pkl')
    scaler_path = os.path.join(models_dir, 'suitability_scaler.pkl')
    features_path = os.path.join(models_dir, 'selected_features.pkl')

    joblib.dump(model, model_path)
    joblib.dump(label_encoder, encoder_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump({'selected_features': selected_features, 'selected_indices': selected_indices}, features_path)

    print(f"\nOptimized model saved to: {model_path}")
    print(f"Encoder saved to: {encoder_path}")
    print(f"Scaler saved to: {scaler_path}")
    print(f"Feature info saved to: {features_path}")

    # Save best parameters
    params_path = os.path.join(models_dir, 'best_hyperparameters.pkl')
    joblib.dump(best_params, params_path)
    print(f"Best hyperparameters saved to: {params_path}")

    return model, label_encoder, scaler

if __name__ == "__main__":
    train_suitability_model()