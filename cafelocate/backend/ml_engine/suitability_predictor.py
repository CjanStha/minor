import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[2]
MODELS_DIR = BASE_DIR / 'ml' / 'models'

RF_MODEL_PATH = MODELS_DIR / 'rf_regressor_v3_85_15.pkl'
XGB_MODEL_PATH = MODELS_DIR / 'xgb_regressor_v3_85_15.pkl'
SCALER_PATH = MODELS_DIR / 'scaler_regression.pkl'
FEATURES_PATH = MODELS_DIR / 'feature_columns_regression.pkl'

DEFAULT_FEATURES = [
    'competitors_within_500m', 'competitors_within_200m',
    'competitors_min_distance', 'competitors_avg_distance',
    'roads_within_500m', 'roads_avg_distance',
    'schools_within_500m', 'schools_within_200m', 'schools_min_distance',
    'hospitals_within_500m', 'hospitals_min_distance',
    'bus_stops_within_500m', 'bus_stops_min_distance',
    'population_density_proxy', 'accessibility_score',
    'foot_traffic_score', 'competition_pressure',
]

_rf_model = None
_xgb_model = None
_scaler = None
_feature_columns = None


def _load_models():
    global _rf_model, _xgb_model, _scaler, _feature_columns
    if _scaler is not None and _feature_columns is not None and (_rf_model is not None or _xgb_model is not None):
        return

    try:
        _scaler = joblib.load(SCALER_PATH)
        _feature_columns = joblib.load(FEATURES_PATH)
    except FileNotFoundError:
        logger.warning('Regression preprocessing artifacts not found. Using fallback scoring.')
        _rf_model = None
        _xgb_model = None
        _scaler = None
        _feature_columns = DEFAULT_FEATURES
        return

    try:
        _rf_model = joblib.load(RF_MODEL_PATH)
        if hasattr(_rf_model, 'n_jobs'):
            _rf_model.n_jobs = 1
    except Exception as exc:
        logger.warning(f'Random Forest regressor could not be loaded: {exc}')
        _rf_model = None

    try:
        _xgb_model = joblib.load(XGB_MODEL_PATH)
        if hasattr(_xgb_model, 'n_jobs'):
            _xgb_model.n_jobs = 1
    except Exception as exc:
        logger.warning(f'XGBoost regressor could not be loaded: {exc}')
        _xgb_model = None

    if _rf_model is None and _xgb_model is None:
        logger.warning('No regression models could be loaded. Using fallback scoring.')
    elif _rf_model is not None and _xgb_model is not None:
        logger.info('Regression suitability ensemble loaded successfully.')
    elif _rf_model is not None:
        logger.info('Regression suitability Random Forest model loaded successfully.')
    else:
        logger.info('Regression suitability XGBoost model loaded successfully.')


def _score_to_level(score):
    if score >= 70:
        return 'High Suitability'
    if score >= 40:
        return 'Medium Suitability'
    return 'Low Suitability'


def _build_feature_array(features_dict):
    feature_columns = _feature_columns or DEFAULT_FEATURES
    values = [float(features_dict.get(feature, 0.0)) for feature in feature_columns]
    return pd.DataFrame([values], columns=feature_columns, dtype=float), feature_columns


def _fallback_score(features_dict):
    population_component = min(100.0, float(features_dict.get('population_density_proxy', 0.0)) * 10.0) * 0.20
    accessibility_component = min(10.0, float(features_dict.get('accessibility_score', 0.0))) * 10.0 * 0.15
    foot_traffic_component = min(10.0, float(features_dict.get('foot_traffic_score', 0.0))) * 10.0 * 0.15
    schools_component = min(20.0, float(features_dict.get('schools_within_500m', 0.0)) * 5.0) * 0.10
    bus_component = min(20.0, float(features_dict.get('bus_stops_within_500m', 0.0)) * 5.0) * 0.10
    competition_penalty = min(100.0, float(features_dict.get('competition_pressure', 0.0)) * 10.0) * 0.20
    competitor_penalty = min(100.0, float(features_dict.get('competitors_within_200m', 0.0)) * 10.0) * 0.10

    score = (
        population_component +
        accessibility_component +
        foot_traffic_component +
        schools_component +
        bus_component -
        competition_penalty -
        competitor_penalty
    )
    return float(max(0.0, min(100.0, score)))


def get_suitability_prediction(features_dict):
    """
    Predict a continuous suitability score using the v3 regression models.
    """
    _load_models()

    try:
        features_array, feature_columns = _build_feature_array(features_dict)

        if (_rf_model is None and _xgb_model is None) or _scaler is None:
            score = _fallback_score(features_dict)
            return {
                'predicted_score': score,
                'predicted_suitability': _score_to_level(score),
                'confidence': 0.0,
                'model_type': 'regression_fallback',
                'features_used': len(feature_columns),
                'model_breakdown': {},
            }

        features_scaled = pd.DataFrame(
            _scaler.transform(features_array),
            columns=feature_columns,
        )
        model_scores = {}
        if _rf_model is not None:
            model_scores['random_forest_v3_score'] = float(_rf_model.predict(features_scaled)[0])
        if _xgb_model is not None:
            model_scores['xgboost_v3_score'] = float(_xgb_model.predict(features_scaled)[0])

        ensemble_score = float(np.clip(sum(model_scores.values()) / len(model_scores), 0.0, 100.0))

        if len(model_scores) > 1:
            score_values = list(model_scores.values())
            confidence = float(max(0.0, min(1.0, 1.0 - abs(score_values[0] - score_values[1]) / 100.0)))
            model_type = 'regression_ensemble_v3'
        else:
            confidence = 0.75
            model_type = 'regression_single_model_v3'

        return {
            'predicted_score': round(ensemble_score, 2),
            'predicted_suitability': _score_to_level(ensemble_score),
            'confidence': round(confidence, 3),
            'model_type': model_type,
            'features_used': len(feature_columns),
            'model_breakdown': {name: round(value, 2) for name, value in model_scores.items()},
        }
    except Exception as exc:
        logger.error(f'Error in regression suitability prediction: {exc}')
        score = _fallback_score(features_dict)
        return {
            'predicted_score': round(score, 2),
            'predicted_suitability': _score_to_level(score),
            'confidence': 0.0,
            'model_type': 'regression_error_fallback',
            'features_used': len(_feature_columns or DEFAULT_FEATURES),
            'model_breakdown': {},
            'error': str(exc),
        }
