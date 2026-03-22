
# CLASS IMBALANCE HANDLING - RE-EVALUATION REPORT

**Generated:** 2026-03-22 14:45:06

## Problem Identified

The original evaluation achieved perfect accuracy (1.0) across all metrics, which was unrealistic.

### Root Cause: Extreme Class Imbalance

- **High Suitability:** 521 samples (99.6%)
- **Medium Suitability:** 2 samples (0.4%)

### Why Perfect Scores Were Misleading

1. Random stratified split did not guarantee minority class in test set
2. v2 (80-20) split: Test set had 0 "Medium" samples
3. Models learned to predict "High" for everything (99.6% accuracy by default)
4. Could not evaluate true minority class performance

---

## Solutions Implemented

### 1. Manual Stratified Splitting
- Ensured at least 1 minority sample in test set
- Preserved class distribution in both train and test sets
- Random state = 42 for reproducibility

### 2. Class Weights
- **Random Forest:** Applied sample weights inversely proportional to class frequency
- **XGBoost:** Applied scale_pos_weight = (negative_class / positive_class)
- Effect: Penalizes misclassification of minority class

### 3. Comprehensive Metrics

#### Standard Metrics
- Accuracy, Precision (weighted), Recall (weighted), F1 (weighted)

#### Balanced Metrics (for imbalanced data)
- **Macro-averaged:** Precision, Recall, F1 (equal weight to each class)
- **Matthews Correlation Coefficient:** -1 to +1 scale, 0 = random, 1 = perfect
- **ROC-AUC:** Shows trade-off between TPR and FPR

#### Per-Class Metrics
- Separate precision, recall, F1 for each class
- Shows how well model handles minority class specifically

---

## Re-Evaluation Results

### Dataset Split

**v2_stratified_80_20:**
- Train: 1 Medium + 417 High
- Test: 1 Medium + 104 High

**v3_stratified_85_15:**
- Train: 1 Medium + 443 High
- Test: 1 Medium + 78 High

### Model Performance Summary

#### Random Forest (with Class Weights)

**v2 Split:**
- Accuracy: 0.9905
- F1 (Macro): 0.4976
- Matthews Corr: 0.0000

**v3 Split:**
- Accuracy: 0.9873
- F1 (Macro): 0.4968
- Matthews Corr: 0.0000

#### XGBoost (with Class Weights)

**v2 Split:**
- Accuracy: 0.9905
- F1 (Macro): 0.4976
- Matthews Corr: 0.0000

**v3 Split:**
- Accuracy: 0.9873
- F1 (Macro): 0.4968
- Matthews Corr: 0.0000

---

## Key Insights

### Minority Class Performance

Looking at the "Medium" suitability class specifically:

| Split | Algorithm | Precision | Recall | F1 |
|-------|-----------|-----------|--------|-----|
| v2 | RF | 0.0000 | 0.0000 | 0.0000 |
| v2 | XGB | 0.0000 | 0.0000 | 0.0000 |
| v3 | RF | 0.0000 | 0.0000 | 0.0000 |
| v3 | XGB | 0.0000 | 0.0000 | 0.0000 |

**Interpretation:**
- A score of 0.0 means the model failed to detect the minority class
- Low recall (< 0.5) means missing many minority samples
- High precision with low recall means few false positives but many false negatives

### Recommendations

1. **Collect More Data:** With only 2 "Medium" samples, evaluation is unreliable
   - Target: Minimum 50-100 samples per class
   - Goal: Achieve at least 5% minority class representation

2. **Alternative Approaches:**
   - Undersample majority class to balance classes
   - Oversample minority class (with caution)
   - Adjust decision threshold for minority class preference
   - Use anomaly detection instead of classification

3. **Better Metrics for Business:**
   - Cost-sensitive evaluation (assign cost to each error type)
   - Precision-Recall curve instead of ROC-AUC
   - Expected value framework

4. **Next Steps:**
   - Collect more labeled "Medium" suitability locations
   - Define business requirements (precision vs recall priority)
   - Implement threshold adjustment based on business needs

---

## Files Generated

- `imbalanced_evaluation_report.json` - Full JSON report
- `imbalanced_evaluation_comparison.csv` - CSV comparison table
- `rf_model_v2_stratified_80_20_weighted.pkl` - RF with class weights (v2)
- `rf_model_v3_stratified_85_15_weighted.pkl` - RF with class weights (v3)
- `xgb_model_v2_stratified_80_20_weighted.pkl` - XGB with class weights (v2)
- `xgb_model_v3_stratified_85_15_weighted.pkl` - XGB with class weights (v3)

---

## Conclusion

The re-evaluation with class imbalance handling provides a more realistic assessment of model performance.
The extremely low minority class metrics reveal the actual challenge: with only 2 samples total,
robust evaluation is impossible. **Collecting more labeled data should be the priority.**
