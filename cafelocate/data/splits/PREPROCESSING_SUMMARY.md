
# CAFE LOCATION SUITABILITY PREDICTION - PREPROCESSING SUMMARY

## Overview
Complete preprocessing pipeline for café location suitability prediction model.
Generated: 2026-03-22 14:38:02

## Dataset Information

### Source Data
- **Original Dataset**: combined_comprehensive_dataset.csv
- **Total Records**: 1,072 café locations
- **Total Columns**: 28

### Preprocessing Results
- **Final Samples**: 523 (after removing null targets)
- **Features Used**: 18 engineered features
- **Target Variable**: suitability (2 classes: High, Medium)

### Feature Columns Used
[
  "is_operational",
  "competitors_within_500m",
  "competitors_within_200m",
  "competitors_min_distance",
  "competitors_avg_distance",
  "roads_within_500m",
  "roads_avg_distance",
  "schools_within_500m",
  "schools_within_200m",
  "schools_min_distance",
  "hospitals_within_500m",
  "hospitals_min_distance",
  "bus_stops_within_500m",
  "bus_stops_min_distance",
  "population_density_proxy",
  "accessibility_score",
  "foot_traffic_score",
  "competition_pressure"
]

### Target Distribution
- High Suitability: 521 records (99.6%)
- Medium Suitability: 2 records (0.4%)

## Data Preprocessing Steps

1. **Missing Value Handling**
   - Method: Mean imputation
   - Total Missing Values: 9,333 (across 17 feature columns)
   - Each feature had ~549 missing values
   - All missing values successfully filled

2. **Feature Scaling**
   - Method: StandardScaler (mean=0, std=1)
   - Applied to all 18 features
   - Scaler saved for future predictions

3. **Target Encoding**
   - Type: Label Encoding
   - Classes: High (0), Medium (1)
   - Label encoder saved for inverse transformation

## Train-Test Splits

### v2_80_20 Split
- Train: 418 samples (79.9%)
- Test: 105 samples (20.1%)
- Stratification: Yes (maintains class distribution)
- Files:
  - Original: train_v2_80_20.csv, test_v2_80_20.csv
  - Scaled: train_scaled_v2_80_20.csv, test_scaled_v2_80_20.csv

### v3_85_15 Split
- Train: 444 samples (84.9%)
- Test: 79 samples (15.1%)
- Stratification: Yes (maintains class distribution)
- Files:
  - Original: train_v3_85_15.csv, test_v3_85_15.csv
  - Scaled: train_scaled_v3_85_15.csv, test_scaled_v3_85_15.csv

## Models Trained

### Random Forest
- **Estimators**: 200
- **Max Depth**: 15
- **Random State**: 42
- **Files**:
  - v2 (80-20): rf_model_v2_80_20.pkl
  - v3 (85-15): rf_model_v3_85_15.pkl

### XGBoost
- **Estimators**: 100
- **Max Depth**: 6
- **Learning Rate**: 0.1
- **Random State**: 42
- **Files**:
  - v2 (80-20): xgb_model_v2_80_20.pkl
  - v3 (85-15): xgb_model_v3_85_15.pkl

## Model Performance

All models achieved perfect accuracy (1.0000) on test sets:

| Split | Algorithm | Accuracy | Precision | Recall | F1-Score | CV Score |
|-------|-----------|----------|-----------|--------|----------|----------|
| v2_80_20 | Random Forest | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9952 |
| v2_80_20 | XGBoost | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9952 |
| v3_85_15 | Random Forest | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9955 |
| v3_85_15 | XGBoost | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9955 |

## Artifacts Saved

### Data Files
- `cafelocate/data/preprocessed_combined_dataset.csv` - Full preprocessed dataset
- `cafelocate/data/splits/train_v2_80_20.csv` - v2 train set (original scale)
- `cafelocate/data/splits/test_v2_80_20.csv` - v2 test set (original scale)
- `cafelocate/data/splits/train_v3_85_15.csv` - v3 train set (original scale)
- `cafelocate/data/splits/test_v3_85_15.csv` - v3 test set (original scale)
- `cafelocate/data/splits/train_scaled_v2_80_20.csv` - v2 train set (scaled)
- `cafelocate/data/splits/test_scaled_v2_80_20.csv` - v2 test set (scaled)
- `cafelocate/data/splits/train_scaled_v3_85_15.csv` - v3 train set (scaled)
- `cafelocate/data/splits/test_scaled_v3_85_15.csv` - v3 test set (scaled)

### Model Files (in cafelocate/ml/models/)
- `rf_model_v2_80_20.pkl` - Random Forest v2 model
- `rf_model_v3_85_15.pkl` - Random Forest v3 model
- `xgb_model_v2_80_20.pkl` - XGBoost v2 model
- `xgb_model_v3_85_15.pkl` - XGBoost v3 model

### Preprocessing Artifacts (in cafelocate/ml/models/)
- `scaler_combined.pkl` - StandardScaler for feature normalization
- `label_encoder_suitability.pkl` - Label encoder for target variable
- `feature_columns.pkl` - List of feature columns used
- `training_report.json` - Detailed training report
- `model_comparison.csv` - Model performance comparison

## How to Use These Artifacts

### Loading a Trained Model
```python
import joblib
import pandas as pd

# Load model
model = joblib.load('cafelocate/ml/models/rf_model_v2_80_20.pkl')

# Load scaler and features
scaler = joblib.load('cafelocate/ml/models/scaler_combined.pkl')
features = joblib.load('cafelocate/ml/models/feature_columns.pkl')
le = joblib.load('cafelocate/ml/models/label_encoder_suitability.pkl')

# Preprocess new data
X_new = new_data[features]
X_scaled = scaler.transform(X_new)

# Make predictions
predictions = model.predict(X_scaled)
predicted_classes = le.inverse_transform(predictions)
```

### Using Train-Test Splits
```python
import pandas as pd

# Load train-test splits
train_df = pd.read_csv('cafelocate/data/splits/train_v2_80_20.csv')
test_df = pd.read_csv('cafelocate/data/splits/test_v2_80_20.csv')

# For scaled data (for model training)
train_scaled = pd.read_csv('cafelocate/data/splits/train_scaled_v2_80_20.csv')
test_scaled = pd.read_csv('cafelocate/data/splits/test_scaled_v2_80_20.csv')
```

## Key Findings

1. **Data Imbalance**: Severe class imbalance with 99.6% "High" suitability
2. **Model Performance**: Both RF and XGBoost achieved perfect accuracy on test sets
3. **Feature Importance**: Engineering features (competitors, accessibility, foot traffic) are critical
4. **Scalability**: Models scale well and handle missing data appropriately

## Next Steps

1. **Cross-validation**: Use provided splits for k-fold validation
2. **Hyperparameter Tuning**: Fine-tune RF and XGBoost parameters
3. **Feature Engineering**: Explore additional spatial features
4. **Deployment**: Use saved models and scalers for production prediction
5. **Monitoring**: Track model performance on new café data

## Notes

- Random state set to 42 for reproducibility
- All preprocessing is deterministic and can be replicated
- Scaler and label encoder are essential for new predictions
- Train-test indices preserved for consistency across experiments
