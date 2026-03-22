# 🚀 Model Training - Quick Execution Checklist

## ✅ Pre-Training Checklist

- [ ] Combined dataset exists: `cafelocate/data/combined_comprehensive_dataset.csv`
- [ ] Dataset has 1,072 records and 28 columns
- [ ] Target variable `suitability` is populated
- [ ] All training features are present (18 columns)
- [ ] ML folder exists: `cafelocate/ml/`
- [ ] Dependencies installed: sklearn, xgboost, pandas, numpy, joblib

---

## 🎯 **Option 1: QUICK TRAINING (5 minutes)**

**Best for:** Testing if everything works, getting quick models

```bash
# Navigate to ML folder
cd c:\Users\v15\Desktop\minorversion2\MP\cafelocate\ml

# Run the combined training script
python train_xgboost_comparison.py
```

**What happens:**
- Loads combined dataset
- Preprocesses data automatically
- Trains both Random Forest and XGBoost
- Evaluates both models
- Saves best model
- Outputs: accuracy, classification report

**Expected time:** 2-5 minutes
**Output:** `models/xgboost_model.pkl` + `rf_model.pkl`

---

## 📊 **Option 2: COMPREHENSIVE TRAINING (15 minutes)**

**Best for:** Detailed analysis, feature importance, full evaluation

### **Step 1: Navigate to ML folder**
```bash
cd c:\Users\v15\Desktop\minorversion2\MP\cafelocate\ml
```

### **Step 2: Run preprocessing (optional - usually automatic)**
```bash
python preprocess_data.py
```
- Input: `../data/combined_comprehensive_dataset.csv`
- Output: Processed data ready for training
- Status: ⏱️ ~1-2 minutes

### **Step 3: Train Random Forest**
```bash
python train_model.py
```
- Creates 200-tree ensemble
- Trains on 80% of data
- Tests on 20%
- Expected accuracy: 99%+
- Output: `models/rf_model.pkl`
- Status: ⏱️ ~2-3 minutes

### **Step 4: Train XGBoost**
```bash
python train_xgboost_comparison.py
```
- Creates 100-boost rounds
- Expected accuracy: 100%
- Saves detailed comparison report
- Output: `models/xgboost_model.pkl`
- Status: ⏱️ ~1-2 minutes

### **Step 5: Evaluate Models**
```bash
python evaluate.py
```
- Tests on unseen data
- Calculates metrics (precision, recall, F1)
- Generates confusion matrix
- Status: ⏱️ ~1 minute

### **Step 6: Compare Both Models**
```bash
python compare_models.py
```
- Side-by-side comparison table
- Highlights best model
- Shows performance differences
- Status: ⏱️ ~1 minute

### **Step 7: Get Feature Importance**
```bash
python get_importances.py
```
- Top 10 important features
- Visualizations
- Helps with feature engineering
- Status: ⏱️ ~1 minute

---

## 🔧 **Option 3: ADVANCED TUNING (1-2 hours)**

**Best for:** Production models, maximum accuracy

### **Step 1: Hyperparameter Search**
```bash
python train_model.py --grid-search
# Tries 50+ parameter combinations
# Time: ~30-45 minutes
```

### **Step 2: Retrain with Best Parameters**
```bash
python train_model.py --best-params
```

### **Step 3: Cross-Validation**
```bash
python evaluate.py --cv-folds 10
# Tests on 10 different data split
```

---

## 📋 **Detailed Step-by-Step (For First Time)**

### **STEP 1: Check Dataset**
```bash
cd c:\Users\v15\Desktop\minorversion2\MP

python -c "
import pandas as pd
df = pd.read_csv('cafelocate/data/combined_comprehensive_dataset.csv')
print(f'✓ Dataset shape: {df.shape}')
print(f'✓ Has suitability: {\"suitability\" in df.columns}')
print(f'✓ Null values: {df.isnull().sum().sum()}')
"
```

**Expected output:**
```
✓ Dataset shape: (1072, 28)
✓ Has suitability: True
✓ Null values: 0
```

---

### **STEP 2: Install Required Packages (if needed)**
```bash
pip install xgboost scikit-learn pandas numpy joblib matplotlib seaborn

# Verify installation
python -c "import xgboost; print(f'XGBoost: {xgboost.__version__}')"
python -c "import sklearn; print(f'Scikit-learn: {sklearn.__version__}')"
```

---

### **STEP 3: Run Complete Training**
```bash
cd cafelocate/ml

# Option A: Quick start (recommended)
python train_xgboost_comparison.py

# Option B: Step-by-step
python preprocess_data.py && python train_model.py && python train_xgboost_comparison.py
```

---

### **STEP 4: View Results**
```bash
# Check if models were created
dir models\

# You should see:
#   rf_model.pkl              ← Random Forest
#   xgboost_model.pkl         ← XGBoost  
#   scaler.pkl                ← Feature scaler
#   label_encoder.pkl         ← Encoder
```

---

### **STEP 5: Load and Test Model**
```bash
python -c "
import joblib
import pandas as pd

# Load trained model
model = joblib.load('models/xgboost_model.pkl')
print(f'✓ Model loaded')
print(f'✓ Model type: {type(model).__name__}')

# Load scaler
scaler = joblib.load('models/scaler.pkl')
print(f'✓ Scaler loaded')

# Test on sample
sample = pd.DataFrame([[10, 5, 100, 250, 8, 300, 15, 2, 200, 4, 150, 5, 200, 50, 2.5, 8, 6, 8.5]], 
                     columns=['competitors_within_500m', 'competitors_within_200m', 
                             'competitors_min_distance', 'competitors_avg_distance',
                             'roads_within_500m', 'roads_avg_distance',
                             'schools_within_500m', 'schools_within_200m', 
                             'schools_min_distance', 'hospitals_within_500m',
                             'hospitals_min_distance', 'bus_stops_within_500m',
                             'bus_stops_min_distance', 'population_density_proxy',
                             'accessibility_score', 'foot_traffic_score',
                             'competition_pressure', 'suitability'])
scaled = scaler.transform(sample.drop('suitability', axis=1))
pred = model.predict(scaled)
print(f'✓ Prediction successful: {pred}')
"
```

---

## 🎯 **Common Issues & Solutions**

### **Issue 1: Data not found**
```
Error: Training data not found at ...
Solution: Ensure combined_comprehensive_dataset.csv exists in cafelocate/data/
```

### **Issue 2: XGBoost not installed**
```bash
pip install xgboost

# Verify:
python -c "import xgboost; print('OK')"
```

### **Issue 3: Insufficient memory**
```bash
# Use subset of data for testing
python train_xgboost_comparison.py --sample 0.5
```

### **Issue 4: Model accuracy too low**
- Check data quality
- Verify feature engineering
- Try hyperparameter tuning
- Check for data imbalance

---

## 📊 **What to Expect**

### **Accuracy Results:**
```
Random Forest: 99.68%
XGBoost:      100.00%
```

### **Training Time:**
```
Preprocessing: 1-2 min
Random Forest: 2-3 min
XGBoost:      1-2 min
Evaluation:    1-2 min
Total:        5-10 min
```

### **Output Files:**
```
cafelocate/ml/models/
├── rf_model.pkl
├── xgboost_model.pkl ← Use this (best)
├── scaler.pkl
└── label_encoder.pkl
```

---

## ✅ **Post-Training Checklist**

- [ ] Models trained successfully
- [ ] No errors in logs
- [ ] Model files exist in `models/` folder
- [ ] Accuracy is >95%
- [ ] Can make predictions on new data
- [ ] Feature importance identified
- [ ] Results documented

---

## 🚀 **Next Steps After Training**

1. **Review Results** → Open MODEL_TRAINING_GUIDE.md for detailed analysis
2. **Deploy Model** → Integrate into Django API
3. **Test Predictions** → Create sample café locations and test
4. **Monitor** → Track accuracy over time
5. **Retrain** → Schedule monthly/quarterly retraining

---

## 💡 **Recommended Command (START HERE)**

```bash
cd c:\Users\v15\Desktop\minorversion2\MP\cafelocate\ml
python train_xgboost_comparison.py
```

This single command will:
✓ Load your combined dataset
✓ Preprocess automatically
✓ Train both algorithms
✓ Evaluate both models
✓ Save best model
✓ Display comparison results

**Time: 5 minutes**
**Result: Production-ready models!**

