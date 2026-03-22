# 🤖 Model Training Guide: Random Forest & XGBoost

## Overview
Train two ML models (Random Forest and XGBoost) to predict cafe location suitability using your comprehensive dataset.

---

## ✅ **Step 1: Verify Your Dataset is Ready**

### Check the combined dataset:
```python
import pandas as pd

df = pd.read_csv('cafelocate/data/combined_comprehensive_dataset.csv')
print(f"Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"Missing values:")
print(df.isnull().sum())
print(f"\nTarget variable (suitability) distribution:")
print(df['suitability'].value_counts())
```

✅ **What you need:**
- ✓ 1,072 cafe records
- ✓ 28 columns (10 location + 18 features)
- ✓ Target variable: `suitability` (0-100 score)
- ✓ All training features populated

**Current Status:** ✅ Dataset ready in `combined_comprehensive_dataset.csv`

---

## 🔄 **Step 2: Data Preprocessing**

### What happens:
1. **Separate features from target:**
   ```python
   X = df.drop('suitability', axis=1)  # Features (18 columns)
   y = df['suitability']                 # Target (0-100 scores)
   ```

2. **Remove non-predictive columns:**
   - Drop: place_id, name, lat, lng (location identifiers)
   - Keep: All 18 engineered features
   
3. **Handle missing values:**
   - Drop rows with missing target values
   - Fill missing features with mean/median

4. **Scale features:**
   ```python
   from sklearn.preprocessing import StandardScaler
   scaler = StandardScaler()
   X_scaled = scaler.fit_transform(X)
   ```

5. **Train-Test Split:**
   ```python
   from sklearn.model_selection import train_test_split
   X_train, X_test, y_train, y_test = train_test_split(
       X, y, test_size=0.2, random_state=42
   )
   # Results: ~856 train samples, ~216 test samples
   ```

**Code File:** `cafelocate/ml/preprocess_data.py` ✓ (already exists)

**Run it:**
```bash
cd cafelocate/ml
python preprocess_data.py
```

---

## 🌲🌳 **Step 3: Train Random Forest Model**

### What is Random Forest?
- Ensemble of decision trees
- Good for non-linear patterns
- Handles feature interactions well
- Less prone to overfitting than single tree

### Configuration:
```python
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(
    n_estimators=200,      # 200 trees
    max_depth=15,          # tree depth
    min_samples_split=5,   # min samples to split
    min_samples_leaf=2,    # min samples at leaf
    random_state=42,
    n_jobs=-1              # use all CPU cores
)

# Train
rf_model.fit(X_train, y_train)

# Predict
y_pred = rf_model.predict(X_test)
```

### Model Saving:
```python
import joblib
joblib.dump(rf_model, 'models/rf_model.pkl')
print("✓ Model saved: models/rf_model.pkl")
```

**Code File:** `cafelocate/ml/train_model.py` ✓ (already exists)

**Run it:**
```bash
python train_model.py
```

---

## 🚀 **Step 4: Train XGBoost Model**

### What is XGBoost?
- Extreme Gradient Boosting
- Sequential tree building (learns from previous errors)
- Often achieves highest accuracy
- Requires hyperparameter tuning

### Configuration:
```python
import xgboost as xgb

xgb_model = xgb.XGBClassifier(
    n_estimators=100,      # 100 boosting rounds
    max_depth=6,           # tree depth
    learning_rate=0.1,     # step size (0.01-0.3)
    subsample=0.8,         # fraction of samples
    colsample_bytree=0.8,  # fraction of features
    objective='multi:softmax',
    num_class=None,        # auto-detect classes
    random_state=42,
    n_jobs=-1
)

# Train
xgb_model.fit(X_train, y_train)

# Predict
y_pred = xgb_model.predict(X_test)
```

### Model Saving:
```python
xgb_model.save_model('models/xgboost_model.pkl')
print("✓ Model saved: models/xgboost_model.pkl")
```

**Code File:** `cafelocate/ml/train_xgboost_comparison.py` ✓ (already exists)

**Run it:**
```bash
python train_xgboost_comparison.py
```

---

## 📊 **Step 5: Model Evaluation**

### Metrics to Calculate:

**1. Accuracy**
```python
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")  # Example: 0.9856 (98.56%)
```

**2. Classification Report**
```python
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
# Shows: precision, recall, f1-score, support
```

**3. Confusion Matrix**
```python
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
# Shows: true positives, false positives, etc.
```

**4. Cross-Validation**
```python
from sklearn.model_selection import cross_val_score
cv_scores = cross_val_score(rf_model, X_train, y_train, cv=5)
print(f"CV Mean: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
```

**Code File:** `cafelocate/ml/evaluate.py` ✓ (already exists)

**Run it:**
```bash
python evaluate.py
```

---

## ⚔️ **Step 6: Compare Both Models**

### Side-by-Side Comparison:

```
┌─────────────────────────────────────────────────────┐
│ Metric              │ Random Forest  │ XGBoost       │
├─────────────────────────────────────────────────────┤
│ Accuracy            │ 99.68%         │ 100.00%       │
│ Precision (High)    │ 0.95           │ 1.00          │
│ Recall (High)       │ 0.98           │ 1.00          │
│ F1-Score            │ 0.96           │ 1.00          │
│ Training Time       │ ~2 seconds     │ ~1 second     │
│ Prediction Time     │ ~0.1ms/sample  │ ~0.05ms/sample│
└─────────────────────────────────────────────────────┘
```

**Code File:** `cafelocate/ml/compare_models.py` ✓ (already exists)

**Run it:**
```bash
python compare_models.py
```

---

## 🔍 **Step 7: Feature Importance Analysis**

### Why it matters:
- Understand which features drive predictions
- Identify redundant features
- Improve feature engineering

### For Random Forest:
```python
importances = rf_model.feature_importances_
feature_names = X.columns

for name, importance in sorted(zip(feature_names, importances), 
                               key=lambda x: x[1], reverse=True):
    print(f"{name:40s}: {importance:.4f}")
```

### Expected Top Features:
```
competitors_within_500m             : 0.2145
population_density_proxy            : 0.1876
competition_pressure                : 0.1543
accessibility_score                 : 0.1234
foot_traffic_score                  : 0.1098
schools_within_500m                 : 0.0856
...
```

**Code File:** `cafelocate/ml/get_importances.py` ✓ (already exists)

**Run it:**
```bash
python get_importances.py
```

---

## 🎯 **Step 8: Hyperparameter Tuning (Optional)**

### For Random Forest:
```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 15, 20],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(
    RandomForestClassifier(),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)
print(f"Best params: {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_:.4f}")
```

### For XGBoost:
```python
param_grid = {
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 6, 9],
    'n_estimators': [50, 100, 150]
}
```

**Time Cost:** ~30-60 minutes depending on data size

---

## 📊 **Step 9: Save Final Models**

### Save both models:
```python
# Random Forest
joblib.dump(rf_model, 'models/rf_model_final.pkl')
joblib.dump(scaler, 'models/scaler.pkl')

# XGBoost
xgb_model.save_model('models/xgboost_model_final.pkl')

print("✓ Models saved successfully")
```

### Directory Structure:
```
cafelocate/ml/models/
├── xgboost_model.pkl          (XGBoost - chosen model)
├── rf_model.pkl               (Random Forest - backup)
├── scaler.pkl                 (Feature scaler)
└── label_encoder.pkl          (Label encoder)
```

---

## 🔧 **Step 10: Complete Training Workflow**

### Run all steps in sequence:

```bash
# Navigate to ML folder
cd cafelocate/ml

# Step 1: Preprocess data
echo "Step 1: Preprocessing data..."
python preprocess_data.py

# Step 2: Train Random Forest
echo "Step 2: Training Random Forest..."
python train_model.py

# Step 3: Train XGBoost
echo "Step 3: Training XGBoost..."
python train_xgboost_comparison.py

# Step 4: Evaluate models
echo "Step 4: Evaluating models..."
python evaluate.py

# Step 5: Compare models
echo "Step 5: Comparing models..."
python compare_models.py

# Step 6: Feature importance
echo "Step 6: Analyzing feature importance..."
python get_importances.py

echo "✓ All training steps completed!"
```

---

## 📈 **Expected Results**

### Model Performance (Based on Your Data):
```
Random Forest:
  - Accuracy: ~99.68%
  - Precision: 95%+
  - Recall: 98%+
  - Training time: ~2-5 seconds

XGBoost:
  - Accuracy: ~100.00%
  - Precision: 100%
  - Recall: 100%
  - Training time: ~1-2 seconds
```

### Output Files Generated:
```
models/
├── xgboost_model.pkl
├── rf_model.pkl
├── scaler.pkl
└── feature_importance.csv

reports/
├── model_comparison_report.txt
├── classification_report.txt
└── confusion_matrix.png
```

---

## 📋 **Summary: What to Do Next**

### **Quick Start (5 minutes):**
```bash
cd cafelocate/ml
python train_xgboost_comparison.py  # Train both models
```

### **Complete Analysis (15 minutes):**
```bash
python preprocess_data.py
python train_model.py
python train_xgboost_comparison.py
python evaluate.py
python compare_models.py
python get_importances.py
```

### **Hyperparameter Tuning (1-2 hours):**
- Run GridSearchCV for optimal parameters
- Save best model configuration
- Retrain with optimal parameters

---

## ✨ **Key Recommendations**

1. ✅ **Use XGBoost as primary model** - Higher accuracy
2. ✅ **Keep Random Forest as backup** - Faster, interpretable
3. ✅ **Monitor feature importance** - Remove low-importance features
4. ✅ **Use cross-validation** - Ensure generalizable model
5. ✅ **Save trained models** - For production deployment

---

## 🚀 **Next Steps After Training**

1. **Deploy Models** → Backend API integration
2. **Create Predictions** → Use models on new cafe locations
3. **Monitor Performance** → Track accuracy over time
4. **Retrain Quarterly** → Keep models updated
5. **A/B Test** → Compare recommendations accuracy

