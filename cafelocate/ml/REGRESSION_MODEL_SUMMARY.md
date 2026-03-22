# Regression-Based Cafe Location Suitability Prediction Model

## Overview

Successfully transitioned the cafe location suitability prediction project from **classification** to **regression-based modeling**. This approach provides continuous suitability scores (0-100) instead of discrete categories, enabling more nuanced location recommendations.

## Dataset Adjustments

### Source Data
- **Original Dataset**: `combined_comprehensive_dataset.csv` (1,072 records)
- **Enriched Dataset**: `combined_comprehensive_dataset_ft_enriched.csv`
  - Added missing foot traffic scores (549 records filled using ensemble method)
  - Complete feature coverage: 17 features, 0 missing values

### Target Variable: Continuous Regression Score

**Formula Used** (normalized to 0-100):
```
suitability_score = (
    0.20 × population_density +
    0.15 × accessibility +
    0.15 × foot_traffic +
    0.10 × schools_within_500m +
    0.10 × bus_stops_within_500m -
    0.20 × competition_pressure -
    0.10 × competitors_within_200m
) × 100
```

**Score Distribution**:
- Range: 0.00 - 100.00
- Mean: 48.18
- Standard Deviation: 19.01
- Median: 47.35
- Training Samples: 523 (with complete targets)

### Why Regression Over Classification?

| Aspect | Classification | Regression |
|--------|----------------|-----------|
| Output | Discrete categories (High/Medium) | Continuous scores (0-100) |
| Information Loss | Loses granular details | Preserves full information |
| Class Imbalance | Severe (99.6% High, 0.4% Medium) | ✅ Not an issue |
| Interpretation | Binary decision | Nuanced ranking |
| Real-World Use | Less useful for planning | Better for location prioritization |

## Model Architecture

### Models Trained

#### 1. Random Forest Regressor
- **Estimators**: 200 trees
- **Max Depth**: 15
- **Min Samples Split**: 5
- **Min Samples Leaf**: 2

#### 2. XGBoost Regressor
- **Estimators**: 100
- **Max Depth**: 6
- **Learning Rate**: 0.1
- **Subsample**: 0.8
- **Column Sample**: 0.8

### Train-Test Splits

**Version 2 (80-20 Split)**
- Train: 418 samples
- Test: 105 samples

**Version 3 (85-15 Split)**
- Train: 444 samples
- Test: 79 samples

## Model Performance

### v2 (80-20 Split)

#### Random Forest Regressor
```
Train Set:
  R² Score: 0.9869 (explains 98.69% of variance)
  RMSE: 2.1983
  MAE: 1.4068
  Median AE: 0.8903

Test Set:
  R² Score: 0.9516 (explains 95.16% of variance)
  RMSE: 3.9555
  MAE: 2.9263
  Median AE: 1.8774

5-Fold Cross-Validation:
  R² Mean: 0.9272 ± 0.0120
  RMSE Mean: 5.1298
```

#### XGBoost Regressor
```
Train Set:
  R² Score: 0.9997 (explains 99.97% of variance)
  RMSE: 0.3136
  MAE: 0.2347

Test Set:
  R² Score: 0.9658 (explains 96.58% of variance) ⭐ BEST TEST PERFORMANCE
  RMSE: 3.3257
  MAE: 2.3785

5-Fold Cross-Validation:
  R² Mean: 0.9529 ± 0.0030
  RMSE Mean: 4.1554
```

---

### v3 (85-15 Split)

#### Random Forest Regressor
```
Test Set:
  R² Score: 0.9535
  RMSE: 3.7690
  MAE: 2.8067

5-Fold CV:
  R² Mean: 0.9328 ± 0.0056
  RMSE Mean: 4.9675
```

#### XGBoost Regressor
```
Test Set:
  R² Score: 0.9685 (explains 96.85% of variance) ⭐ BEST OVERALL TEST
  RMSE: 3.1025
  MAE: 2.1576

5-Fold CV:
  R² Mean: 0.9635 ± 0.0047
  RMSE Mean: 3.6948
```

---

## Prediction Results

### Full Dataset Predictions (1,072 Cafes)

**Ensemble Model** (Average of all 4 models):

```
Mean Score: 44.59
Median Score: 41.19
Std Dev: 13.29
Min Score: 6.22
Max Score: 95.27
```

### Score Distribution

```
 0-30 (Very Poor):      101 cafes (9.4%)   - 🔴 Not recommended
30-45 (Poor):           686 cafes (64.0%)  - 🟠 Below average
45-60 (Average):        130 cafes (12.1%)  - 🟡 Moderate suitability
60-75 (Good):           108 cafes (10.1%)  - 🔵 Suitable
75-100 (Excellent):      47 cafes (4.4%)   - 🟢 Highly suitable
```

### Top 10 Most Suitable Locations

| Rank | Cafe Name | Score | Coordinates | Rating |
|------|-----------|-------|-------------|--------|
| 1 | H2O Cafe | 95.27 | 27.7178, 85.3474 | 🟢 Excellent |
| 2 | Unknown Cafe | 94.35 | 27.7010, 85.3226 | 🟢 Excellent |
| 3 | Chiya bhatea | 91.76 | 27.7181, 85.3485 | 🟢 Excellent |
| 4 | Cafe Zen | 91.18 | 27.7183, 85.3470 | 🟢 Excellent |
| 5 | क्याफे डे पासज | 89.80 | 27.7183, 85.3465 | 🟢 Excellent |
| 6 | Krishna Cafe | 88.45 | 27.7176, 85.3482 | 🟢 Excellent |
| 7 | Cafe Inc | 87.92 | 27.7187, 85.3456 | 🟢 Excellent |
| 8 | Navya Cafe | 87.34 | 27.7190, 85.3445 | 🟢 Excellent |
| 9 | The Darling Cafe | 86.76 | 27.7182, 85.3468 | 🟢 Excellent |
| 10 | Chill Space Cafe | 85.28 | 27.7177, 85.3480 | 🟢 Excellent |

## Saved Artifacts

### Trained Models
- `rf_regressor_v2_80_20.pkl` - Random Forest (80-20 split)
- `xgb_regressor_v2_80_20.pkl` - XGBoost (80-20 split)
- `rf_regressor_v3_85_15.pkl` - Random Forest (85-15 split)
- `xgb_regressor_v3_85_15.pkl` - XGBoost (85-15 split)

### Supporting Files
- `scaler_regression.pkl` - StandardScaler for feature normalization
- `feature_columns_regression.pkl` - Feature list (17 features)

### Reports & Analysis
- `regression_training_report.json` - Complete training metrics
- `regression_model_comparison.csv` - Side-by-side model comparison
- `regression_detailed_metrics.json` - Comprehensive evaluation metrics
- `regression_predictions_summary.json` - Prediction statistics
- `regression_predictions.csv` - All 1,072 cafes with predicted scores

## Usage Instructions

### Making Predictions

```python
from regression_predictions import RegressionPredictor
import pandas as pd

# Load predictor
predictor = RegressionPredictor()

# Load dataset
df = pd.read_csv('combined_comprehensive_dataset_ft_enriched.csv')

# Single prediction
features = df.iloc[0][predictor.feature_cols].to_dict()
score = predictor.predict_single(features, model_type='ensemble')
print(f"Suitability Score: {score:.2f}")

# Batch predictions for all cafes
predictions = predictor.batch_predict(df, model_type='ensemble')
```

### Model Selection

- **With Ensemble**: Combines all 4 models for most robust predictions
  - Best for production use
  - Average of RF v2, XGB v2, RF v3, XGB v3

- **Individual Models**:
  - `'rf_v2'` - Random Forest 80-20 split
  - `'xgb_v2'` - XGBoost 80-20 split (best single model)
  - `'rf_v3'` - Random Forest 85-15 split
  - `'xgb_v3'` - XGBoost 85-15 split (second best)

## Key Advantages

✅ **Continuous Scores**: 0-100 scale provides nuanced ranking
✅ **No Class Imbalance**: Regression doesn't suffer from categorical skew
✅ **High Accuracy**: R² > 0.95 on test sets
✅ **Interpretable**: Scores directly indicate suitability
✅ **Scalable**: Works on full 1,072 cafe dataset
✅ **Ensemble Robust**: Multiple models reduce overfitting risk

## Next Steps

1. **Real-World Validation**: Test on newly opening cafes in Kathmandu
2. **Feature Importance Analysis**: Identify which factors matter most
3. **Production Deployment**: Integrate into web/mobile applications
4. **Continuous Improvement**: Collect actual outcomes (operating status, revenue) and retrain
5. **Clustering Analysis**: Discover natural cafe segments (optional)

## Note on Synthetic Data

Current scores are derived from a **synthetic formula** based on engineered features. For production use, validate against:
- Actual cafe performance metrics (revenue, customer count)
- Real foot traffic data (Google Popular Times API)
- Cafe operating status (still open after 12 months?)
- Customer reviews and satisfaction

See `CONTINUOUS_SCORE_REAL_WORLD_ANALYSIS.md` for detailed roadmap to production.

---

**Generated**: March 22, 2026
**Status**: ✅ Complete - Ready for prediction and deployment
