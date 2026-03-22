"""
Generate confusion matrices for all 4 models
Converts regression predictions to classification categories
"""

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


class ConfusionMatrixGenerator:
    def __init__(self):
        # Use absolute paths
        base_dir = Path(__file__).parent.parent
        self.models_dir = base_dir / 'ml' / 'models'
        self.data_dir = base_dir / 'data'
        
        # Load all models
        self.rf_v2 = joblib.load(self.models_dir / 'rf_regressor_v2_80_20.pkl')
        self.xgb_v2 = joblib.load(self.models_dir / 'xgb_regressor_v2_80_20.pkl')
        self.rf_v3 = joblib.load(self.models_dir / 'rf_regressor_v3_85_15.pkl')
        self.xgb_v3 = joblib.load(self.models_dir / 'xgb_regressor_v3_85_15.pkl')
        
        # Load scaler and features
        self.scaler = joblib.load(self.models_dir / 'scaler_regression.pkl')
        self.feature_cols = joblib.load(self.models_dir / 'feature_columns_regression.pkl')
        
        # Load dataset
        self.df = pd.read_csv(self.data_dir / 'combined_comprehensive_dataset_ft_enriched.csv')
        
        print("[OK] All models and data loaded for confusion matrix generation")
    
    def categorize_scores(self, scores, n_bins=4):
        """Convert continuous scores to categorical labels"""
        if n_bins == 4:
            # Very Low, Low, High, Very High
            bins = [0, 25, 50, 75, 100]
            labels = ['Very Low', 'Low', 'High', 'Very High']
        elif n_bins == 3:
            # Low, Medium, High
            bins = [0, 33.33, 66.67, 100]
            labels = ['Low', 'Medium', 'High']
        else:  # 5 bins
            # Very Low, Low, Medium, High, Very High
            bins = [0, 20, 40, 60, 80, 100]
            labels = ['Very Low', 'Low', 'Medium', 'High', 'Very High']
        
        return pd.cut(scores, bins=bins, labels=labels, include_lowest=True)
    
    def generate_confusion_matrices(self):
        """Generate confusion matrices for all models"""
        
        # Generate continuous target scores for all records
        normalized_features = {}
        for col in ['population_density_proxy', 'accessibility_score', 'foot_traffic_score', 
                    'schools_within_500m', 'bus_stops_within_500m', 'competition_pressure', 
                    'competitors_within_200m']:
            max_val = self.df[col].max()
            if max_val > 0:
                normalized_features[col] = self.df[col] / max_val
            else:
                normalized_features[col] = self.df[col]
        
        continuous_score = (
            0.20 * normalized_features['population_density_proxy'] +
            0.15 * normalized_features['accessibility_score'] +
            0.15 * normalized_features['foot_traffic_score'] +
            0.10 * normalized_features['schools_within_500m'] +
            0.10 * normalized_features['bus_stops_within_500m'] -
            0.20 * normalized_features['competition_pressure'] -
            0.10 * normalized_features['competitors_within_200m']
        )
        min_score = continuous_score.min()
        max_score = continuous_score.max()
        continuous_score_normalized = ((continuous_score - min_score) / (max_score - min_score) * 100)
        
        # Prepare data
        df_clean = self.df.copy()
        X = df_clean[self.feature_cols].copy()
        
        # Fill missing values
        for col in X.columns:
            X[col] = X[col].fillna(X[col].mean())
        
        # Get target
        y_true = continuous_score_normalized.fillna(continuous_score_normalized.mean()).values
        y_true = np.nan_to_num(y_true, nan=50.0)  # Replace any remaining NaN with median
        
        X_scaled = self.scaler.transform(X)
        y_true_cat = self.categorize_scores(y_true, n_bins=4)
        
        # Make predictions with all models
        models = {
            'RF v2 (80-20)': self.rf_v2,
            'XGB v2 (80-20)': self.xgb_v2,
            'RF v3 (85-15)': self.rf_v3,
            'XGB v3 (85-15)': self.xgb_v3,
        }
        
        print("\n" + "="*80)
        print("CONFUSION MATRICES - ALL MODELS")
        print("="*80)
        
        results = {}
        
        for model_name, model in models.items():
            print(f"\n{'-'*80}")
            print(f"MODEL: {model_name}")
            print(f"{'-'*80}")
            
            # Get predictions
            y_pred = model.predict(X_scaled)
            y_pred_cat = self.categorize_scores(y_pred, n_bins=4)
            
            # Generate confusion matrix
            cm = confusion_matrix(y_true_cat, y_pred_cat, 
                                labels=['Very Low', 'Low', 'High', 'Very High'])
            
            # Display confusion matrix
            print("\nConfusion Matrix:")
            print("               Pred: Very Low  Pred: Low  Pred: High  Pred: Very High")
            labels_list = ['Very Low', 'Low', 'High', 'Very High']
            for i, label in enumerate(labels_list):
                print(f"Actual: {label:9s}  {cm[i,0]:6d}        {cm[i,1]:6d}       {cm[i,2]:6d}        {cm[i,3]:6d}")
            
            # Calculate metrics
            diag_sum = np.trace(cm)
            total = np.sum(cm)
            accuracy = diag_sum / total if total > 0 else 0
            
            print(f"\nAccuracy: {accuracy:.4f} ({diag_sum}/{total})")
            
            # Category-wise accuracy
            print("\nCategory-wise Accuracy:")
            for i, label in enumerate(labels_list):
                row_sum = np.sum(cm[i, :])
                if row_sum > 0:
                    cat_acc = cm[i, i] / row_sum
                    print(f"  {label:9s}: {cat_acc:.4f} ({cm[i,i]}/{row_sum})")
            
            # Classification report
            print("\nClassification Report:")
            print(classification_report(y_true_cat, y_pred_cat, 
                                        labels=['Very Low', 'Low', 'High', 'Very High'],
                                        zero_division=0))
            
            results[model_name] = {
                'confusion_matrix': cm,
                'accuracy': accuracy,
                'y_pred': y_pred,
                'y_true': y_true
            }
        
        return results
    
    def create_visualizations(self, results):
        """Create confusion matrix visualizations"""
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        model_names = list(results.keys())
        
        for idx, (model_name, data) in enumerate(results.items()):
            ax = axes[idx // 2, idx % 2]
            cm = data['confusion_matrix']
            
            # Plot confusion matrix heatmap
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       xticklabels=['Very Low', 'Low', 'High', 'Very High'],
                       yticklabels=['Very Low', 'Low', 'High', 'Very High'],
                       cbar_kws={'label': 'Count'})
            
            ax.set_title(f'{model_name}\nAccuracy: {data["accuracy"]:.4f}', fontsize=12, fontweight='bold')
            ax.set_ylabel('Actual', fontsize=10)
            ax.set_xlabel('Predicted', fontsize=10)
        
        plt.tight_layout()
        plt.savefig('./models/confusion_matrices_all_models.png', dpi=300, bbox_inches='tight')
        print("\n[OK] Confusion matrices visualization saved: ./models/confusion_matrices_all_models.png")
        
        return fig
    
    def create_summary_report(self, results):
        """Create a summary report of confusion matrices"""
        
        summary_data = []
        for model_name, data in results.items():
            summary_data.append({
                'Model': model_name,
                'Accuracy': f"{data['accuracy']:.4f}",
                'Correct Predictions': f"{np.trace(data['confusion_matrix'])}"
            })
        
        df_summary = pd.DataFrame(summary_data)
        
        print("\n" + "="*80)
        print("CONFUSION MATRIX SUMMARY")
        print("="*80)
        print(df_summary.to_string(index=False))
        
        # Save to CSV
        df_summary.to_csv('./models/confusion_matrix_summary.csv', index=False)
        print("\n[OK] Summary saved: ./models/confusion_matrix_summary.csv")


def main():
    generator = ConfusionMatrixGenerator()
    results = generator.generate_confusion_matrices()
    generator.create_visualizations(results)
    generator.create_summary_report(results)
    
    print("\n" + "="*80)
    print("CONFUSION MATRIX GENERATION COMPLETE")
    print("="*80)
    print("\nGenerated Files:")
    print("  1. ./models/confusion_matrices_all_models.png - Visual comparison")
    print("  2. ./models/confusion_matrix_summary.csv - Summary statistics")


if __name__ == '__main__':
    main()
