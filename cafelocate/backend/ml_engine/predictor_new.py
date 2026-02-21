import joblib
import numpy as np
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Path to the trained model files ──────────────────────────────────────────
# These .pkl files are created by ml/train_model.py
# They live in ml/models/ folder and are copied here after training
BASE_DIR      = Path(__file__).resolve().parent
MODEL_PATH    = BASE_DIR / 'models' / 'rf_model.pkl'
ENCODER_PATH  = BASE_DIR / 'models' / 'label_encoder.pkl'

# ── Load model once at import time ────────────────────────────────────────────
# Module-level variables = loaded ONCE when Django starts, not on each request.
# This is critical for performance — loading a model takes ~1 second.
_model   = None
_encoder = None

def _load_model():
    """Load model from disk if not already loaded (lazy loading)."""
    global _model, _encoder
    if _model is None:
        try:
            _model   = joblib.load(MODEL_PATH)
            _encoder = joblib.load(ENCODER_PATH)
            logger.info("ML model loaded successfully.")
        except FileNotFoundError:
            logger.warning("ML models not found. Using fallback predictions.")
            _model = None
            _encoder = None


# ── Feature names — must match what the model was trained on ──────────────────
FEATURE_NAMES = [
    'competitor_count',     # number of cafes within 500m
    'avg_competitor_rating', # average rating of those cafes
    'road_length_m',         # total road length in buffer (meters)
    'population_density',    # people per sq km in the ward
]

# ── Human-readable labels ─────────────────────────────────────────────────────
CAFE_TYPE_LABELS = {
    'coffee_shop':   'Coffee Shop',
    'bakery':        'Bakery Café',
    'dessert_shop':  'Dessert Shop',
    'restaurant':    'Restaurant Café',
    'juice_bar':     'Juice Bar',
    'ice_cream':     'Ice Cream Parlor',
    'cafe_bar':      'Café Bar',
    'internet_cafe': 'Internet Café',
}

# Map café types to suitability levels
TYPE_TO_SUITABILITY = {
    'coffee_shop':   'High Suitability',
    'bakery':        'High Suitability',
    'dessert_shop':  'Medium Suitability',
    'restaurant':    'Medium Suitability',
    'juice_bar':     'Medium Suitability',
    'ice_cream':     'Low Suitability',
    'cafe_bar':      'Low Suitability',
    'internet_cafe': 'Low Suitability',
}


def get_suitability_prediction(features: list) -> dict:
    """
    Get location suitability prediction based on café type prediction.
    This maps café type predictions to suitability levels.
    """
    # Get café type prediction
    type_prediction = get_prediction(features)

    if type_prediction.get('predicted_type') == 'Model not trained yet':
        return {
            'predicted_suitability': 'Model not trained yet',
            'confidence': 0,
            'all_probabilities': {},
        }

    predicted_type = type_prediction['predicted_type']
    confidence = type_prediction['confidence']

    # Map to suitability
    suitability = TYPE_TO_SUITABILITY.get(predicted_type, 'Medium Suitability')

    # Create suitability probabilities based on type probabilities
    type_probs = type_prediction.get('all_probabilities', {})
    suitability_probs = {'Low Suitability': 0, 'Medium Suitability': 0, 'High Suitability': 0}

    for type_name, prob in type_probs.items():
        if type_name in ['Coffee Shop', 'Bakery Café']:
            suitability_probs['High Suitability'] += prob
        elif type_name in ['Dessert Shop', 'Restaurant Café', 'Juice Bar']:
            suitability_probs['Medium Suitability'] += prob
        else:
            suitability_probs['Low Suitability'] += prob

    return {
        'predicted_suitability': suitability,
        'confidence': confidence,
        'all_probabilities': {k: round(v, 3) for k, v in suitability_probs.items()},
    }


def get_prediction(features: list) -> dict:
    """
    Run prediction from the trained Random Forest model.

    Args:
        features: [competitor_count, avg_rating, road_length_m, pop_density]

    Returns:
        { 'predicted_type': 'Bakery Café', 'confidence': 0.87,
          'all_probabilities': {'Coffee Shop': 0.08, 'Bakery Café': 0.87, ...} }
    """
    _load_model()

    if _model is None:
        # Model not trained yet — return a safe fallback
        return {
            'predicted_type':    'Model not trained yet',
            'confidence':        0,
            'all_probabilities': {},
        }

    # Convert feature list to 2D array (sklearn expects shape: [n_samples, n_features])
    X = np.array([features])    # shape: (1, 4)

    # Get predicted class (integer) and convert to label string
    predicted_int   = _model.predict(X)[0]
    predicted_label = _encoder.inverse_transform([predicted_int])[0]

    # Get probabilities for all classes
    probabilities = _model.predict_proba(X)[0]
    class_labels  = _encoder.inverse_transform(_model.classes_)

    all_probs = {
        CAFE_TYPE_LABELS.get(label, label): round(float(prob), 3)
        for label, prob in zip(class_labels, probabilities)
    }

    return {
        'predicted_type':    CAFE_TYPE_LABELS.get(predicted_label, predicted_label),
        'confidence':        round(float(max(probabilities)), 3),
        'all_probabilities':  all_probs,
    }