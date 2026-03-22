"""
Regression-Based Model Training for Cafe Location Suitability
Uses continuous suitability scores instead of categorical labels.

Models:
  - Random Forest Regressor
  - XGBoost Regressor

Dataset: combined_comprehensive_dataset_ft_enriched.csv (with foot traffic enriched)
Target: Continuous suitability score (0-100)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score, 
    mean_absolute_percentage_error
)
import joblib
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================
DATA_PATH = '../data/combined_comprehensive_dataset_ft_enriched.csv'
MODELS_DIR = './models'
RANDOM_STATE = 42
TEST_SIZE_V2 = 0.20  # 80-20 split
TEST_SIZE_V3 = 0.15  # 85-15 split
CV_FOLDS = 5

# Feature columns to use (18 features)
FEATURE_COLS = [
    'competitors_within_500m', 'competitors_within_200m', 
    'competitors_min_distance', 'competitors_avg_distance',
    'roads_within_500m', 'roads_avg_distance',
    'schools_within_500m', 'schools_within_200m', 'schools_min_distance',
    'hospitals_within_500m', 'hospitals_min_distance',
    'bus_stops_within_500m', 'bus_stops_min_distance',
    'population_density_proxy', 'accessibility_score',
    'foot_traffic_score', 'competition_pressure'
]

class RegressionModelTrainer:
    """Train regression models for continuous suitability scoring"""
    
    def __init__(self, data_path, models_dir=MODELS_DIR):
        self.df = pd.read_csv(data_path)
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        
    def generate_continuous_target(self):
        """
        Generate continuous suitability score (0-100) based on feature engineering.
        
        Formula (from synthetic suitability calculation):
        score = (0.20*density + 0.15*accessibility + 0.15*foot_traffic
                 + 0.10*schools + 0.10*transit - 0.20*competition - 0.10*competitors)
        
        Normalized to 0-100 range for interpretability.
        """
        print("\n" + "="*70)
        print("GENERATING CONTINUOUS REGRESSION TARGET")
        print("="*70)
        
        df = self.df.copy()
        
        # Normalize features to 0-1 range first
        normalized_features = {}
        for col in ['population_density_proxy', 'accessibility_score', 'foot_traffic_score', 
                    'schools_within_500m', 'bus_stops_within_500m', 'competition_pressure', 
                    'competitors_within_200m']:
            max_val = df[col].max()
            if max_val > 0:
                normalized_features[col] = df[col] / max_val
            else:
                normalized_features[col] = df[col]
        
        # Generate continuous score with weighted formula
        continuous_score = (
            0.20 * normalized_features['population_density_proxy'] +
            0.15 * normalized_features['accessibility_score'] +
            0.15 * normalized_features['foot_traffic_score'] +
            0.10 * normalized_features['schools_within_500m'] +
            0.10 * normalized_features['bus_stops_within_500m'] -
            0.20 * normalized_features['competition_pressure'] -
            0.10 * normalized_features['competitors_within_200m']
        )
        
        # Normalize to 0-100 range
        min_score = continuous_score.min()
        max_score = continuous_score.max()
        continuous_score_normalized = ((continuous_score - min_score) / 
                                       (max_score - min_score) * 100)
        
        df['suitability_score_continuous'] = continuous_score_normalized
        
        print(f"\n✓ Continuous suitability scores generated")
        print(f"  Range: {continuous_score_normalized.min():.2f} - {continuous_score_normalized.max():.2f}")
        print(f"  Mean: {continuous_score_normalized.mean():.2f}")
        print(f"  Std: {continuous_score_normalized.std():.2f}")
        print(f"  Median: {continuous_score_normalized.median():.2f}")
        
        return df
    
    def prepare_data(self, df):
        """
        Prepare features and targets for modeling.
        Handle missing values and scale features.
        """
        print("\n" + "="*70)
        print("PREPARING DATA FOR REGRESSION")
        print("="*70)
        
        # Remove rows without continuous score
        df_clean = df.dropna(subset=['suitability_score_continuous']).copy()
        
        print(f"\nDataset shape: {df_clean.shape}")
        print(f"Samples with targets: {len(df_clean)}")
        
        # Extract features
        X = df_clean[FEATURE_COLS].copy()
        y = df_clean['suitability_score_continuous'].copy()
        
        # Handle missing values in features
        missing_summary = X.isna().sum()
        if missing_summary.sum() > 0:
            print(f"\nMissing values in features:")
            for col in missing_summary[missing_summary > 0].index:
                print(f"  {col}: {missing_summary[col]}")
            
            # Use mean imputation
            X = X.fillna(X.mean())
            print(f"✓ Applied mean imputation")
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        
        print(f"\n✓ Features scaled (StandardScaler)")
        print(f"  Features: {len(FEATURE_COLS)}")
        print(f"  Samples: {len(X_scaled)}")
        print(f"  Target range: {y.min():.2f} - {y.max():.2f}")
        
        return X_scaled, y, scaler
    
    def train_regression_models(self, X, y, test_size, version_name):
        """Train Random Forest and XGBoost regressors"""
        
        print(f"\n{'='*70}")
        print(f"TRAINING REGRESSION MODELS - {version_name}")
        print(f"{'='*70}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=RANDOM_STATE
        )
        
        print(f"\nTrain-Test Split ({(1-test_size)*100:.0f}-{test_size*100:.0f}):")
        print(f"  Train: {len(X_train)} samples")
        print(f"  Test: {len(X_test)} samples")
        print(f"  Train target range: {y_train.min():.2f} - {y_train.max():.2f}")
        print(f"  Test target range: {y_test.min():.2f} - {y_test.max():.2f}")
        
        # ===== Random Forest Regressor =====
        print(f"\n{'-'*70}")
        print(f"Training Random Forest Regressor...")
        print(f"{'-'*70}")
        
        rf_regressor = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=0
        )
        
        rf_regressor.fit(X_train, y_train)
        rf_pred_train = rf_regressor.predict(X_train)
        rf_pred_test = rf_regressor.predict(X_test)
        
        # RF Metrics
        rf_metrics = {
            'train': self.calculate_regression_metrics(y_train, rf_pred_train, 'Train'),
            'test': self.calculate_regression_metrics(y_test, rf_pred_test, 'Test'),
            'cv': self.cross_validate_regressor(rf_regressor, X_train, y_train, 'RF')
        }
        
        # ===== XGBoost Regressor =====
        print(f"\n{'-'*70}")
        print(f"Training XGBoost Regressor...")
        print(f"{'-'*70}")
        
        xgb_regressor = XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            verbosity=0
        )
        
        xgb_regressor.fit(X_train, y_train)
        xgb_pred_train = xgb_regressor.predict(X_train)
        xgb_pred_test = xgb_regressor.predict(X_test)
        
        # XGB Metrics
        xgb_metrics = {
            'train': self.calculate_regression_metrics(y_train, xgb_pred_train, 'Train'),
            'test': self.calculate_regression_metrics(y_test, xgb_pred_test, 'Test'),
            'cv': self.cross_validate_regressor(xgb_regressor, X_train, y_train, 'XGB')
        }
        
        # Save models
        print(f"\n{'-'*70}")
        print(f"Saving models...")
        print(f"{'-'*70}")
        
        rf_path = self.models_dir / f'rf_regressor_{version_name}.pkl'
        xgb_path = self.models_dir / f'xgb_regressor_{version_name}.pkl'
        
        joblib.dump(rf_regressor, rf_path)
        joblib.dump(xgb_regressor, xgb_path)
        
        print(f"✓ Random Forest saved: {rf_path}")
        print(f"✓ XGBoost saved: {xgb_path}")
        
        return {
            'version': version_name,
            'rf_model': rf_regressor,
            'xgb_model': xgb_regressor,
            'rf_metrics': rf_metrics,
            'xgb_metrics': xgb_metrics,
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test
        }
    
    def calculate_regression_metrics(self, y_true, y_pred, set_name):
        """Calculate regression evaluation metrics"""
        
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        mape = mean_absolute_percentage_error(y_true, y_pred)
        
        # Median Absolute Error
        median_ae = np.median(np.abs(y_true - y_pred))
        
        print(f"\n{set_name} Set Metrics:")
        print(f"  R² Score: {r2:.4f}")
        print(f"  RMSE: {rmse:.4f} (lower is better)")
        print(f"  MAE: {mae:.4f}")
        print(f"  Median AE: {median_ae:.4f}")
        print(f"  MAPE: {mape:.4f}%")
        
        return {
            'r2': float(r2),
            'rmse': float(rmse),
            'mae': float(mae),
            'median_ae': float(median_ae),
            'mape': float(mape),
            'mse': float(mse)
        }
    
    def cross_validate_regressor(self, model, X, y, model_name):
        """Perform k-fold cross-validation"""
        
        scoring = {
            'r2': 'r2',
            'neg_mse': 'neg_mean_squared_error',
            'neg_mae': 'neg_mean_absolute_error'
        }
        
        cv_results = cross_validate(model, X, y, cv=CV_FOLDS, scoring=scoring)
        
        cv_summary = {
            'r2_mean': float(cv_results['test_r2'].mean()),
            'r2_std': float(cv_results['test_r2'].std()),
            'rmse_mean': float(np.sqrt(-cv_results['test_neg_mse'].mean())),
            'rmse_std': float(np.sqrt(cv_results['test_neg_mse'].std())),
            'mae_mean': float(-cv_results['test_neg_mae'].mean()),
            'mae_std': float(-cv_results['test_neg_mae'].std())
        }
        
        print(f"\n{model_name} Cross-Validation ({CV_FOLDS}-Fold):")
        print(f"  R² Mean: {cv_summary['r2_mean']:.4f} ± {cv_summary['r2_std']:.4f}")
        print(f"  RMSE Mean: {cv_summary['rmse_mean']:.4f} ± {cv_summary['rmse_std']:.4f}")
        print(f"  MAE Mean: {cv_summary['mae_mean']:.4f} ± {cv_summary['mae_std']:.4f}")
        
        return cv_summary
    
    def generate_comparison_report(self, results_v2, results_v3):
        """Generate comprehensive comparison report"""
        
        print(f"\n{'='*70}")
        print(f"REGRESSION MODEL COMPARISON REPORT")
        print(f"{'='*70}")
        
        comparison_data = {
            'v2_80_20': {
                'rf_test_r2': results_v2['rf_metrics']['test']['r2'],
                'rf_test_rmse': results_v2['rf_metrics']['test']['rmse'],
                'rf_test_mae': results_v2['rf_metrics']['test']['mae'],
                'xgb_test_r2': results_v2['xgb_metrics']['test']['r2'],
                'xgb_test_rmse': results_v2['xgb_metrics']['test']['rmse'],
                'xgb_test_mae': results_v2['xgb_metrics']['test']['mae'],
            },
            'v3_85_15': {
                'rf_test_r2': results_v3['rf_metrics']['test']['r2'],
                'rf_test_rmse': results_v3['rf_metrics']['test']['rmse'],
                'rf_test_mae': results_v3['rf_metrics']['test']['mae'],
                'xgb_test_r2': results_v3['xgb_metrics']['test']['r2'],
                'xgb_test_rmse': results_v3['xgb_metrics']['test']['rmse'],
                'xgb_test_mae': results_v3['xgb_metrics']['test']['mae'],
            }
        }
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(comparison_data).T
        print(f"\nTest Set Performance Comparison:")
        print(comparison_df.to_string())
        
        return comparison_data, comparison_df
    
    def save_reports(self, results_v2, results_v3, comparison_data, comparison_df):
        """Save all reports and metadata"""
        
        # Regression training report
        report = {
            'model_type': 'Regression (Random Forest & XGBoost)',
            'target_type': 'Continuous Suitability Score (0-100)',
            'dataset': 'combined_comprehensive_dataset_ft_enriched.csv',
            'features': FEATURE_COLS,
            'feature_count': len(FEATURE_COLS),
            'v2_80_20': results_v2,
            'v3_85_15': results_v3,
            'comparison': comparison_data
        }
        
        # Clean for JSON serialization
        def make_serializable(obj):
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, pd.Series):
                return obj.to_dict()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        report_clean = {}
        for k, v in report.items():
            if k not in ['v2_80_20', 'v3_85_15']:
                report_clean[k] = v
        
        # Save as JSON
        report_path = self.models_dir / 'regression_training_report.json'
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, default=str)
        print(f"✓ Report saved: {report_path}")
        
        # Save comparison as CSV
        comparison_csv_path = self.models_dir / 'regression_model_comparison.csv'
        comparison_df.to_csv(comparison_csv_path)
        print(f"✓ Comparison CSV saved: {comparison_csv_path}")
        
        # Save detailed metrics
        detailed_path = self.models_dir / 'regression_detailed_metrics.json'
        detailed_metrics = {
            'v2_80_20': {
                'rf_metrics': results_v2['rf_metrics'],
                'xgb_metrics': results_v2['xgb_metrics']
            },
            'v3_85_15': {
                'rf_metrics': results_v3['rf_metrics'],
                'xgb_metrics': results_v3['xgb_metrics']
            }
        }
        with open(detailed_path, 'w', encoding='utf-8') as f:
            json.dump(detailed_metrics, f, indent=2, default=str)
        print(f"✓ Detailed metrics saved: {detailed_path}")

def main():
    print("\n" + "="*70)
    print("REGRESSION-BASED CAFE LOCATION SUITABILITY PREDICTION")
    print("="*70)
    print("\nApproach: Continuous Regression (Random Forest & XGBoost)")
    print("Target: Suitability Score (0-100)")
    print("Dataset: combined_comprehensive_dataset_ft_enriched.csv")
    
    # Initialize trainer
    trainer = RegressionModelTrainer(DATA_PATH)
    
    # Step 1: Generate continuous targets
    df = trainer.generate_continuous_target()
    
    # Step 2: Prepare data
    X, y, scaler = trainer.prepare_data(df)
    
    # Save scaler
    scaler_path = trainer.models_dir / 'scaler_regression.pkl'
    joblib.dump(scaler, scaler_path)
    print(f"\n✓ Scaler saved: {scaler_path}")
    
    # Save feature columns
    features_path = trainer.models_dir / 'feature_columns_regression.pkl'
    joblib.dump(FEATURE_COLS, features_path)
    print(f"✓ Feature columns saved: {features_path}")
    
    # Step 3: Train models with different splits
    results_v2 = trainer.train_regression_models(X, y, TEST_SIZE_V2, 'v2_80_20')
    results_v3 = trainer.train_regression_models(X, y, TEST_SIZE_V3, 'v3_85_15')
    
    # Step 4: Generate comparison report
    comparison_data, comparison_df = trainer.generate_comparison_report(results_v2, results_v3)
    
    # Step 5: Save all reports
    trainer.save_reports(results_v2, results_v3, comparison_data, comparison_df)
    
    print(f"\n{'='*70}")
    print(f"✓ REGRESSION TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"\nModels ready for prediction:")
    print(f"  - rf_regressor_v2_80_20.pkl")
    print(f"  - xgb_regressor_v2_80_20.pkl")
    print(f"  - rf_regressor_v3_85_15.pkl")
    print(f"  - xgb_regressor_v3_85_15.pkl")
    print(f"\nReports saved:")
    print(f"  - regression_training_report.json")
    print(f"  - regression_model_comparison.csv")
    print(f"  - regression_detailed_metrics.json")

if __name__ == '__main__':
    main()
