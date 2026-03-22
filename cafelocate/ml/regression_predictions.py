"""
Regression Model Prediction & Interpretation Tool
Demonstrates how to use trained regression models to predict continuous suitability scores.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import json

class RegressionPredictor:
    """Load and use trained regression models for predictions"""
    
    def __init__(self, models_dir='./models'):
        self.models_dir = Path(models_dir)
        
        # Load models
        self.rf_v2 = joblib.load(self.models_dir / 'rf_regressor_v2_80_20.pkl')
        self.xgb_v2 = joblib.load(self.models_dir / 'xgb_regressor_v2_80_20.pkl')
        self.rf_v3 = joblib.load(self.models_dir / 'rf_regressor_v3_85_15.pkl')
        self.xgb_v3 = joblib.load(self.models_dir / 'xgb_regressor_v3_85_15.pkl')
        
        self.scaler = joblib.load(self.models_dir / 'scaler_regression.pkl')
        self.feature_cols = joblib.load(self.models_dir / 'feature_columns_regression.pkl')
        
        print("✓ All regression models loaded successfully")
        print(f"  Features: {len(self.feature_cols)}")
    
    def interpret_score(self, score):
        """Interpret continuous suitability score"""
        if score >= 75:
            return "🟢 Excellent Location - Highly suitable for cafe"
        elif score >= 60:
            return "🔵 Good Location - Suitable for cafe"
        elif score >= 45:
            return "🟡 Average Location - Moderate suitability"
        elif score >= 30:
            return "🟠 Poor Location - Below average suitability"
        else:
            return "🔴 Very Poor Location - Not recommended"
    
    def predict_single(self, features_dict, model_type='ensemble'):
        """
        Predict suitability score for a single cafe
        
        Args:
            features_dict: Dictionary with feature values
            model_type: 'rf_v2', 'xgb_v2', 'rf_v3', 'xgb_v3', or 'ensemble'
        """
        # Convert to DataFrame and extract feature values
        X = pd.DataFrame([features_dict])[self.feature_cols]
        X_scaled = self.scaler.transform(X)
        
        # Predict based on model type
        if model_type == 'rf_v2':
            score = self.rf_v2.predict(X_scaled)[0]
        elif model_type == 'xgb_v2':
            score = self.xgb_v2.predict(X_scaled)[0]
        elif model_type == 'rf_v3':
            score = self.rf_v3.predict(X_scaled)[0]
        elif model_type == 'xgb_v3':
            score = self.xgb_v3.predict(X_scaled)[0]
        elif model_type == 'ensemble':
            # Ensemble: average all 4 models
            score = (
                self.rf_v2.predict(X_scaled)[0] +
                self.xgb_v2.predict(X_scaled)[0] +
                self.rf_v3.predict(X_scaled)[0] +
                self.xgb_v3.predict(X_scaled)[0]
            ) / 4
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Clip to valid range
        score = np.clip(score, 0, 100)
        
        return score
    
    def batch_predict(self, df, model_type='ensemble'):
        """Predict for multiple cafes"""
        
        X = df[self.feature_cols].copy()
        X_scaled = self.scaler.transform(X)
        
        if model_type == 'rf_v2':
            predictions = self.rf_v2.predict(X_scaled)
        elif model_type == 'xgb_v2':
            predictions = self.xgb_v2.predict(X_scaled)
        elif model_type == 'rf_v3':
            predictions = self.rf_v3.predict(X_scaled)
        elif model_type == 'xgb_v3':
            predictions = self.xgb_v3.predict(X_scaled)
        elif model_type == 'ensemble':
            predictions = (
                self.rf_v2.predict(X_scaled) +
                self.xgb_v2.predict(X_scaled) +
                self.rf_v3.predict(X_scaled) +
                self.xgb_v3.predict(X_scaled)
            ) / 4
        
        predictions = np.clip(predictions, 0, 100)
        
        return predictions


def demo_predictions():
    """Demonstrate prediction on sample cafes"""
    
    print("\n" + "="*70)
    print("REGRESSION MODEL PREDICTION DEMONSTRATION")
    print("="*70)
    
    # Load predictor
    predictor = RegressionPredictor()
    
    # Load full dataset for examples
    df = pd.read_csv('../data/combined_comprehensive_dataset_ft_enriched.csv')
    
    # Get feature columns
    feature_cols = predictor.feature_cols
    
    print(f"\n{'-'*70}")
    print(f"SAMPLE PREDICTIONS (Ensemble of all 4 models)")
    print(f"{'-'*70}")
    
    # Pick 10 random samples
    sample_indices = np.random.choice(len(df), 10, replace=False)
    
    print(f"\n{'#':>2} {'Cafe Name':<40} {'Score':>8} {'Interpretation':<30}")
    print(f"{'-'*100}")
    
    results = []
    for i, idx in enumerate(sample_indices):
        cafe_name = df.iloc[idx]['name']
        features = df.iloc[idx][feature_cols].to_dict()
        
        score = predictor.predict_single(features, model_type='ensemble')
        interpretation = predictor.interpret_score(score)
        
        print(f"{i+1:>2} {cafe_name:<40} {score:>8.2f} {interpretation:<30}")
        
        results.append({
            'name': cafe_name,
            'score': float(score),
            'interpretation': interpretation
        })
    
    print(f"{'-'*100}")
    
    # Statistics
    print(f"\n{'-'*70}")
    print(f"BATCH PREDICTION STATISTICS")
    print(f"{'-'*70}")
    
    batch_scores = predictor.batch_predict(df[feature_cols], model_type='ensemble')
    
    print(f"\nEnsemble Predictions on Full Dataset (1,072 cafes):")
    print(f"  Mean Score: {batch_scores.mean():.2f}")
    print(f"  Median Score: {np.median(batch_scores):.2f}")
    print(f"  Std Dev: {batch_scores.std():.2f}")
    print(f"  Min Score: {batch_scores.min():.2f}")
    print(f"  Max Score: {batch_scores.max():.2f}")
    
    # Distribution
    print(f"\nScore Distribution:")
    ranges = [(0, 30), (30, 45), (45, 60), (60, 75), (75, 100)]
    for min_s, max_s in ranges:
        count = ((batch_scores >= min_s) & (batch_scores < max_s)).sum()
        pct = count / len(batch_scores) * 100
        bar = "█" * int(pct / 2)
        print(f"  {min_s:>2}-{max_s:<2}: {count:>4} ({pct:>5.1f}%) {bar}")
    
    # Add predictions to dataset
    df['suitability_score_predicted'] = batch_scores
    
    # Show top 10 locations
    print(f"\n{'-'*70}")
    print(f"TOP 10 MOST SUITABLE LOCATIONS")
    print(f"{'-'*70}")
    
    top_10 = df.nlargest(10, 'suitability_score_predicted')[['name', 'suitability_score_predicted', 'lat', 'lng']]
    print(f"\n{'#':<3} {'Cafe Name':<45} {'Score':>8} {'Coordinates':<20}")
    print(f"{'-'*78}")
    
    for i, (idx, row) in enumerate(top_10.iterrows(), 1):
        coords = f"{row['lat']:.4f}, {row['lng']:.4f}"
        print(f"{i:<3} {row['name']:<45} {row['suitability_score_predicted']:>8.2f} {coords:<20}")
    
    # Save predictions to CSV
    df_with_predictions = df[['name', 'lat', 'lng', 'suitability_score_predicted']].copy()
    df_with_predictions = df_with_predictions.sort_values('suitability_score_predicted', ascending=False)
    
    output_path = './regression_predictions.csv'
    df_with_predictions.to_csv(output_path, index=False)
    print(f"\n✓ Predictions saved: {output_path}")
    
    # Save summary report
    summary = {
        'total_predictions': len(batch_scores),
        'mean_score': float(batch_scores.mean()),
        'median_score': float(np.median(batch_scores)),
        'std_score': float(batch_scores.std()),
        'min_score': float(batch_scores.min()),
        'max_score': float(batch_scores.max()),
        'distribution': {
            'very_poor_0_30': int(((batch_scores >= 0) & (batch_scores < 30)).sum()),
            'poor_30_45': int(((batch_scores >= 30) & (batch_scores < 45)).sum()),
            'average_45_60': int(((batch_scores >= 45) & (batch_scores < 60)).sum()),
            'good_60_75': int(((batch_scores >= 60) & (batch_scores < 75)).sum()),
            'excellent_75_100': int(((batch_scores >= 75) & (batch_scores <= 100)).sum()),
        }
    }
    
    summary_path = './models/regression_predictions_summary.json'
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Summary saved: {summary_path}")


if __name__ == '__main__':
    demo_predictions()
