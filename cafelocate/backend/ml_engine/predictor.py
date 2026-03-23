import joblib
import numpy as np
import pandas as pd
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / 'models' / 'rf_model.pkl'
ENCODER_PATH = BASE_DIR / 'models' / 'label_encoder.pkl'

_model = None
_encoder = None

CAFE_TYPE_LABELS = {
    'coffee_shop': 'Coffee Shop',
    'bakery': 'Bakery Cafe',
    'dessert_shop': 'Dessert Shop',
    'restaurant': 'Restaurant Cafe',
    'juice_bar': 'Juice Bar',
    'ice_cream': 'Ice Cream Parlor',
    'cafe_bar': 'Cafe Bar',
    'internet_cafe': 'Internet Cafe',
}

FEATURE_NAMES = [
    'competitor_count',
    'avg_competitor_rating',
    'road_length_m',
    'population_density',
]


def _load_model():
    global _model, _encoder
    if _model is not None and _encoder is not None:
        return

    try:
        _model = joblib.load(MODEL_PATH)
        _encoder = joblib.load(ENCODER_PATH)
        if hasattr(_model, 'n_jobs'):
            _model.n_jobs = 1
        logger.info('Cafe type recommendation model loaded successfully.')
    except FileNotFoundError:
        logger.warning('Cafe type recommendation model not found.')
        _model = None
        _encoder = None


def get_prediction(features):
    """
    Predict the best cafe type for a location using the existing classifier.

    Args:
        features: [competitor_count, avg_rating, road_length_m, population_density]
    """
    _load_model()

    if _model is None or _encoder is None:
        return {
            'predicted_type': 'Model not trained yet',
            'confidence': 0.0,
            'all_probabilities': {},
        }

    if not features or len(features) != 4:
        return {
            'predicted_type': 'Insufficient features',
            'confidence': 0.0,
            'all_probabilities': {},
        }

    X = pd.DataFrame([np.array(features, dtype=float)], columns=FEATURE_NAMES)
    predicted_int = _model.predict(X)[0]
    predicted_label = _encoder.inverse_transform([predicted_int])[0]

    probabilities = _model.predict_proba(X)[0] if hasattr(_model, 'predict_proba') else []
    class_labels = _encoder.inverse_transform(_model.classes_) if hasattr(_model, 'classes_') else []

    all_probs = {
        CAFE_TYPE_LABELS.get(label, label): round(float(prob), 3)
        for label, prob in zip(class_labels, probabilities)
    }

    return {
        'predicted_type': CAFE_TYPE_LABELS.get(predicted_label, predicted_label),
        'confidence': round(float(max(probabilities)), 3) if len(probabilities) else 0.0,
        'all_probabilities': all_probs,
    }
