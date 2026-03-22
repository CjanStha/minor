import joblib
import numpy as np
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Path to the trained suitability model files ─────────────────────────────────
# These .pkl files are created by ml/train_model.py
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / 'models' / 'suitability_rf_model_optimized.pkl'
ENCODER_PATH = BASE_DIR / 'models' / 'suitability_label_encoder.pkl'
SCALER_PATH = BASE_DIR / 'models' / 'suitability_scaler.pkl'
FEATURES_PATH = BASE_DIR / 'models' / 'selected_features.pkl'

# ── Load model once at import time ────────────────────────────────────────────
_model = None
_encoder = None
_scaler = None
_selected_features = None

def _load_model():
    """Load optimized suitability model from disk if not already loaded."""
    global _model, _encoder, _scaler, _selected_features
    if _model is None:
        try:
            _model = joblib.load(MODEL_PATH)
            _encoder = joblib.load(ENCODER_PATH)
            _scaler = joblib.load(SCALER_PATH)
            features_info = joblib.load(FEATURES_PATH)
            _selected_features = features_info['selected_features']
            logger.info("Optimized suitability ML model loaded successfully.")
        except FileNotFoundError:
            logger.warning("Optimized suitability models not found. Falling back to basic model.")
            # Try loading basic model
            try:
                _model = joblib.load(BASE_DIR / 'models' / 'suitability_rf_model.pkl')
                _encoder = joblib.load(BASE_DIR / 'models' / 'suitability_label_encoder.pkl')
                _scaler = joblib.load(BASE_DIR / 'models' / 'suitability_scaler.pkl')
                # Use all features if no feature selection
                _selected_features = [
                    'competitors_within_500m', 'competitors_within_200m', 'competitors_min_distance', 'competitors_avg_distance',
                    'roads_within_500m', 'roads_avg_distance',
                    'schools_within_500m', 'schools_within_200m', 'schools_min_distance',
                    'hospitals_within_500m', 'hospitals_min_distance',
                    'bus_stops_within_500m', 'bus_stops_min_distance',
                    'population_density_proxy', 'accessibility_score', 'foot_traffic_score', 'competition_pressure'
                ]
                logger.info("Basic suitability ML model loaded as fallback.")
            except FileNotFoundError:
                logger.warning("No suitability models found. Using rule-based predictions.")
                _model = None
                _encoder = None
                _scaler = None
                _selected_features = None

# ── Feature mapping for suitability prediction ────────────────────────────────
SUITABILITY_FEATURES = [
    'competitors_within_500m', 'competitors_within_200m', 'competitors_min_distance', 'competitors_avg_distance',
    'roads_within_500m', 'roads_avg_distance',
    'schools_within_500m', 'schools_within_200m', 'schools_min_distance',
    'hospitals_within_500m', 'hospitals_min_distance',
    'bus_stops_within_500m', 'bus_stops_min_distance',
    'population_density_proxy', 'accessibility_score', 'foot_traffic_score', 'competition_pressure'
]

def get_suitability_prediction(features_dict: dict) -> dict:
    """
    Get location suitability prediction using the optimized Random Forest model.

    Args:
        features_dict: Dictionary containing all required features

    Returns:
        Dictionary with prediction results
    """
    _load_model()

    if _model is None:
        return {
            'predicted_suitability': 'Model not trained yet',
            'confidence': 0.0,
            'all_probabilities': {},
            'model_type': 'rule_based'
        }

    try:
        # Extract features in correct order
        if _selected_features:
            # Use only selected features
            features = [features_dict[feat] for feat in _selected_features if feat in features_dict]
        else:
            # Use all features
            features = [features_dict[feat] for feat in SUITABILITY_FEATURES if feat in features_dict]

        if len(features) == 0:
            return {
                'predicted_suitability': 'Insufficient features',
                'confidence': 0.0,
                'all_probabilities': {},
                'model_type': 'error'
            }

        # Convert to numpy array and scale
        features_array = np.array(features).reshape(1, -1)
        features_scaled = _scaler.transform(features_array)

        # Get prediction and probabilities
        prediction_encoded = _model.predict(features_scaled)[0]
        probabilities = _model.predict_proba(features_scaled)[0]

        # Decode prediction
        predicted_suitability = _encoder.inverse_transform([prediction_encoded])[0]

        # Get confidence (highest probability)
        confidence = float(max(probabilities))

        # Create probabilities dictionary
        all_probabilities = {}
        for i, prob in enumerate(probabilities):
            class_name = _encoder.inverse_transform([i])[0]
            all_probabilities[class_name] = float(prob)

        return {
            'predicted_suitability': predicted_suitability,
            'confidence': confidence,
            'all_probabilities': all_probabilities,
            'model_type': 'optimized_random_forest',
            'features_used': len(features)
        }

    except Exception as e:
        logger.error(f"Error in suitability prediction: {str(e)}")
        return {
            'predicted_suitability': 'Prediction error',
            'confidence': 0.0,
            'all_probabilities': {},
            'model_type': 'error',
            'error': str(e)
        }

def get_feature_importance():
    """
    Get feature importance from the trained model.
    """
    _load_model()

    if _model is None or not hasattr(_model, 'feature_importances_'):
        return {}

    try:
        importance_dict = {}
        for feat, imp in zip(_selected_features or SUITABILITY_FEATURES, _model.feature_importances_):
            importance_dict[feat] = float(imp)

        return importance_dict
    except:
        return {}