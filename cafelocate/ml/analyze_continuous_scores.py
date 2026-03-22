"""
Continuous Score Assignment: Synthetic vs Real-World Analysis
Shows the gap between synthetic suitability and real-world café success
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt

print("=" * 100)
print("CONTINUOUS SCORE ASSIGNMENT: SYNTHETIC vs REAL-WORLD ANALYSIS")
print("=" * 100)

# Step 1: Load data
print("\n[Step 1] Analyzing current synthetic score formula...")
data_path = "../data/combined_comprehensive_dataset.csv"
df = pd.read_csv(data_path)

print(f"\nDataset Shape: {df.shape}")
print(f"Samples with labels: {df['suitability'].notna().sum()}")

# Step 2: Reverse-engineer and understand synthetic scoring
print("\n[Step 2] Understanding Current Synthetic Scoring Formula...")

# The formula used to generate labels
synthetic_weights = {
    'population_density_proxy': 0.20,
    'accessibility_score': 0.15,
    'foot_traffic_score': 0.15,
    'schools_within_500m': 0.10,
    'bus_stops_within_500m': 0.10,
    'competition_pressure': -0.20,
    'competitors_within_200m': -0.10
}

print("\nCurrent Synthetic Score Formula:")
print("Score = (0.20 * pop_density) + (0.15 * accessibility) + (0.15 * foot_traffic)")
print("        + (0.10 * schools_500m) + (0.10 * bus_stops_500m)")
print("        - (0.20 * competition_pressure) - (0.10 * competitors_200m)")

print("\nProblems with this formula:")
print("  ❌ Weights are arbitrary (why 0.20 for density, not 0.25?)")
print("  ❌ No business validation (does it predict actual café success?)")
print("  ❌ All cafés same weight (both in high-traffic areas get same score)")
print("  ❌ No temporal aspect (doesn't account for changing conditions)")
print("  ❌ No market segmentation (coffee shops vs restaurants vs co-working spaces)")

# Calculate synthetic scores
df['synthetic_score'] = (
    0.20 * df['population_density_proxy'].fillna(df['population_density_proxy'].mean()) +
    0.15 * df['accessibility_score'].fillna(df['accessibility_score'].mean()) +
    0.15 * df['foot_traffic_score'].fillna(df['foot_traffic_score'].mean()) +
    0.10 * df['schools_within_500m'].fillna(df['schools_within_500m'].mean()) +
    0.10 * df['bus_stops_within_500m'].fillna(df['bus_stops_within_500m'].mean()) -
    0.20 * df['competition_pressure'].fillna(df['competition_pressure'].mean()) -
    0.10 * df['competitors_within_200m'].fillna(df['competitors_within_200m'].mean())
)

# Normalize to 0-100
df['synthetic_score_normalized'] = (
    (df['synthetic_score'] - df['synthetic_score'].min()) /
    (df['synthetic_score'].max() - df['synthetic_score'].min()) * 100
)

print(f"\nSynthetic Score Distribution:")
print(f"  Min: {df['synthetic_score_normalized'].min():.2f}")
print(f"  Max: {df['synthetic_score_normalized'].max():.2f}")
print(f"  Mean: {df['synthetic_score_normalized'].mean():.2f}")
print(f"  Std: {df['synthetic_score_normalized'].std():.2f}")

# Step 3: Define what REAL-WORLD scores would be
print("\n[Step 3] What Would REAL-WORLD Continuous Scores Look Like?")

real_world_factors = {
    'Café Success Metrics': [
        'Is café still operating? (0-1 binary)',
        'Years in operation (0-20+ years)',
        'Customer foot traffic (count/day)',
        'Revenue/profitability (0-100 business score)',
        'Online reviews/ratings (0-5 stars)',
        'Social media engagement (0-100)',
        'Prime location indicator (0-1)',
        'Growth trajectory (declining/stable/growing)',
    ],
    'Market Factors': [
        'Neighborhood gentrification rate',
        'Tourism flow changes',
        'Competitor openings/closings',
        'Employee cost trends (wages)',
        'Rent price trends',
        'Demographics shift',
    ],
    'Operational Factors': [
        'Owner experience/expertise',
        'Business model (dine-in/takeout/delivery)',
        'Price point (budget/mid/premium)',
        'Seating capacity utilization',
        'Menu innovation rate',
    ]
}

print("\nFACTORS IN REAL-WORLD SUITABILITY:\n")
for category, factors in real_world_factors.items():
    print(f"{category}:")
    for factor in factors:
        print(f"  • {factor}")

# Step 4: Compare scoring approaches
print("\n[Step 4] Comparing Score Assignment Approaches")

comparison_data = {
    'Approach': [
        'Synthetic Formula\n(Current)',
        'Continuous Synthetic\n(Better Version)',
        'Real-World Based\n(Best - Not Current)'
    ],
    'Score Type': [
        'Discrete Categories\n(High/Med/Low)',
        'Continuous 0-100\n(No boundaries)',
        'Continuous 0-100\n(Validated to reality)'
    ],
    'Data Source': [
        'Mathematical Formula\n(Fixed weights)',
        'Mathematical Formula\n(Fixed weights)',
        'Actual Business Data\n(Success metrics)'
    ],
    'Information Loss': [
        '❌ High\n(Boundaries)',
        '✅ None\n(Continuous)',
        '✅ None\n(Continuous)'
    ],
    'Real-World Valid': [
        '❌ No\n(Untested)',
        '❌ No\n(Still synthetic)',
        '✅ Yes\n(Validated)'
    ],
    'Business Use': [
        'Limited\n(Weak interpretation)',
        'Better\n(Score-based ranking)',
        'Strong\n(Proven predictor)'
    ],
    'Examples': [
        'Café A: "High"\nCafé B: "High"\n(Both same category)',
        'Café A: 67.5\nCafé B: 63.2\n(Distinguishable)',
        'Café A: 67.5\n(Likely to succeed)\nCafé B: 35.1\n(Likely to fail)'
    ]
}

comparison_df = pd.DataFrame(comparison_data)

print("\nComparison Table:")
print(comparison_df.to_string(index=False))

# Step 5: Example: How Real-World Scoring Would Work
print("\n[Step 5] How Real-World Scoring Would Actually Work")

example_real_world_scoring = """
REAL-WORLD APPROACH (What You Should Do):

Step 1: Collect Historical Café Data
┌─────────────────────────────────────────────────┐
│ For each café location in dataset, find:        │
│ • Is the café still open? (success indicator)   │
│ • How long has it been operating?               │
│ • Customer reviews/ratings on Google, TripAd...│
│ • Estimated daily foot traffic (from Google)   │
│ • Local news mentions (growth/closure events)   │
│ • Lease information (prime location?)           │
└─────────────────────────────────────────────────┘

Step 2: Calculate Real Suitability Score
For Café A:
  • Operating: Yes (since 2019, 5 years)
  • Google Rating: 4.7/5 (excellent)
  • Estimated Traffic: 1,200 visitors/day (high)
  • Reviews: "Always packed, great location"
  → Real Suitability Score: 82/100 (validated success)

For Café B:
  • Operating: Closed (2023)
  • Google Rating: 2.1/5 (poor reviews before closure)
  • Near closure: Complaints about low traffic
  • High rent area
  → Real Suitability Score: 31/100 (known failure)

Step 3: Build Predictive Model
Train model: Location Features → Real Suitability Score
  • Now the model learns TRUE pattern
  • High scores correlate with actual success
  • Low scores correlate with actual failure
  • Scores validatable against new locations

Step 4: Deploy with Confidence
Model predicts: New Location → Suitability 72/100
Meaning: "This location has 72% likelihood of success"
(Based on learned patterns from real outcomes)
"""

print(example_real_world_scoring)

# Step 6: Gap Analysis - What's Missing
print("\n[Step 6] Gap Analysis: What's Needed for Real-World Modeling")

gaps = {
    'What You Have': [
        '✅ Geographic features (competitors, density, traffic)',
        '✅ Accessibility metrics (roads, schools, transit)',
        '✅ Location coordinates',
        '✅ Synthetic suitability labels',
        '✅ 1,000+ café locations'
    ],
    'What You\'re Missing': [
        '❌ Café operating status (open/closed)',
        '❌ Historical café data (when opened, when closed)',
        '❌ Revenue/profitability data',
        '❌ Customer satisfaction scores',
        '❌ Actual foot traffic counts',
        '❌ Business performance metrics',
        '❌ Real outcome labels (success/failure)',
        '❌ Time series data (trends over years)'
    ],
    'How to Get Missing Data': [
        '📍 Google Maps: Is café currently shown? When did it close?',
        '🌐 Web archives: Wayback Machine to find historical data',
        '⭐ Reviews: Google/TripAdvisor ratings = satisfaction proxy',
        '📊 Business databases: OpenTable, Yelp, local registration',
        '🚶 Foot traffic: Google Popular Times, foot traffic APIs',
        '💰 Financial: Better Business Bureau, D&B scores',
        '📰 News: Local news for café closures, expansions',
        '📸 Satellite: Google Earth historical images'
    ]
}

gap_df = pd.DataFrame([
    {'Category': 'Have', 'Item': item} for item in gaps['What You Have']
] + [
    {'Category': 'Missing', 'Item': item} for item in gaps['What You\'re Missing']
])

print("\nData Gap Analysis:\n")
for category in ['Have', 'Missing']:
    items = gaps['What You Have'] if category == 'Have' else gaps['What You\'re Missing']
    print(f"{category.upper()}:")
    for item in items:
        print(f"  {item}")
print()

# Step 7: Roadmap
print("\n[Step 7] Roadmap: From Synthetic to Real-World")

roadmap = """
CURRENT STATE (Synthetic)
├─ Advantages:
│  ✓ Can train models immediately
│  ✓ Proof-of-concept works
│  └─ Good for learning/development
└─ Disadvantages:
   ✗ Not validated to reality
   ✗ Perfect scores are meaningless
   ✗ Can't trust predictions on real locations
   ✗ No business value without validation

PHASE 1 (Next Step - RECOMMENDED)
├─ Use continuous scores instead of categories
├─ Advantages:
│  ✓ No information loss
│  ✓ Scores are more interpretable
│  ✓ Foundation for real-world transition
│  └─ Better for ranking cafés
└─ Disadvantages:
   ✗ Still synthetic
   ✗ Still not validated

PHASE 2 (Medium-term)
├─ Collect real outcome data (use methods above)
├─ Advantages:
│  ✓ Ground truth for validation
│  ✓ Can measure prediction accuracy
│  ✓ Build trust in model
│  └─ Real business value
└─ Time: 6-12 months (data collection)

PHASE 3 (Long-term)
├─ Retrain on real data
├─ Deploy with confidence
└─ Continuously update with new data
"""

print(roadmap)

# Step 8: Save analysis report
print("\n[Step 8] Saving comprehensive analysis...")

analysis_report = {
    'timestamp': datetime.now().isoformat(),
    'question': 'How will continuous suitability scores be assigned? Does it make this real-world?',
    'answer': 'Continuous scores are BETTER than categories, but still SYNTHETIC without real outcome data',
    'key_findings': {
        'current_state': 'Using synthetic formula to generate labels',
        'continuous_scores_help': 'Yes - reduces information loss and removes arbitrary boundaries',
        'makes_real_world': 'No - still predicting synthetic scores, not actual business outcomes',
        'validation_needed': 'Must collect real café success/failure data to validate predictions'
    },
    'score_assignment_methods': {
        'current_synthetic': {
            'method': 'Mathematical formula with fixed weights',
            'formula': '0.20*density + 0.15*accessibility + 0.15*traffic - 0.20*competition - 0.10*competitors_200m',
            'weights': synthetic_weights,
            'scores': {
                'min': float(df['synthetic_score_normalized'].min()),
                'max': float(df['synthetic_score_normalized'].max()),
                'mean': float(df['synthetic_score_normalized'].mean()),
                'std': float(df['synthetic_score_normalized'].std())
            },
            'validation': None
        },
        'real_world_needed': {
            'method': 'Regression on actual café success metrics',
            'data_sources': [
                'Café operating status (open/closed)',
                'Years in operation',
                'Customer reviews',
                'Foot traffic estimates',
                'Revenue/profitability',
                'Business performance metrics'
            ],
            'validation': 'Scores predict actual business outcomes'
        }
    },
    'gap_analysis': gaps,
    'recommendations': [
        'Phase 1: Implement continuous regression scores (better than categories)',
        'Phase 2: Collect real café outcome data (6-12 months)',
        'Phase 3: Retrain models on real data',
        'Phase 4: Validate predictions against new locations'
    ]
}

models_dir = Path("models")
report_path = models_dir / "continuous_score_analysis.json"
with open(report_path, 'w', encoding='utf-8') as f:
    json.dump(analysis_report, f, indent=2, ensure_ascii=False)
print(f"✓ Report saved: {report_path.name}")

# Step 9: Create summary document
summary = f"""
# CONTINUOUS SCORE ASSIGNMENT: SYNTHETIC vs REAL-WORLD

**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Question
How will continuous suitability scores be assigned to labels?
Does this make the project model real-world scenarios?

## Answer

### Short Answer
**Continuous scores are BETTER than categories, but STILL SYNTHETIC.**

- ✅ Continuous scores reduce information loss
- ✅ No arbitrary boundaries
- ❌ Still not validated to real-world outcomes
- ❌ Perfect predictions don't mean anything without real data

---

## 1. Current Approach: Synthetic Scoring

### Formula
Score = 0.20 × density + 0.15 × accessibility + 0.15 × traffic
        + 0.10 × schools + 0.10 × transit - 0.20 × competition

### Problems
1. **Arbitrary Weights**: Why 0.20 for density? No justification.
2. **Unvalidated**: Does a score of 75 actually predict café success?
3. **Missing Ground Truth**: No actual outcome data (open/closed, revenue, etc.)
4. **No Causation**: Formula doesn't capture why locations succeed/fail.
5. **No Updates**: Weights never change based on real results.

---

## 2. Continuous vs Categorical Scores

### Categorical (Currently Used - BAD)
```
Café A: Score 15.1 → "High Suitability"
Café B: Score 14.9 → "Medium Suitability"
Problem: Almost identical scores placed in different categories!
```

### Continuous (BETTER)
```
Café A: Score 76.8/100
Café B: Score 71.2/100
Advantage: Subtle differences preserved, no artificial boundaries
```

### But: Both Are Still SYNTHETIC
- Both predict a score generated by a formula
- Neither uses real café success/failure data
- Neither can be validated against reality

---

## 3. What Would REAL-WORLD Scoring Look Like?

### Real-World Data You'd Need

**Café Operating Status:**
- **Café operating status** (open/closed): {df['suitability'].notna().sum()} labeled locations

**Success Metrics:**
- Customer satisfaction (Google reviews: 4.7/5 = excellent)
- Foot traffic volume (estimated from Google Popular Times)
- Revenue/profitability (if available)
- Business longevity (years operating)
- Growth signs (expansion, renovations, positive news)
- Failure signs (closures, negative reviews, declining traffic)

**Market Validation:**
```
Example Real-World Scoring:

Café A - HILO Coffee (Kathmandu)
├─ Location Features: competitors=3, density=high, foot_traffic=0.92
├─ Operating Status: ✓ Open since 2018 (6 years)
├─ Google Reviews: 4.6/5 (200+ reviews)
├─ Estimated Foot Traffic: 1,200+ visitors/day
├─ Market Signals: Growing, expanded menu, social media active
└─ Real-World Suitability Score: 81/100
    (High score backed by actual success metrics)

Café B - OLD PLACE (Kathmandu)
├─ Location Features: competitors=5, density=low, foot_traffic=0.31
├─ Operating Status: ✗ Closed 2022
├─ Google Reviews: 2.3/5 (10 reviews, mostly negative)
├─ Failure Signals: "Always empty", "Poor service", "Better options nearby"
├─ Market Signals: Closed due to foot traffic, location wasn't profitable
└─ Real-World Suitability Score: 28/100
    (Low score backed by actual failure)
```

---

## 4. Does Continuous Scoring Make It Real-World? NO

| Aspect | Categorical | Continuous | Real-World Validated |
|--------|-------------|-----------|---------------------|
| **Type** | Discrete | Continuous | Continuous |
| **Info Loss** | ❌ High | ✅ None | ✅ None |
| **Synthetic** | ❌ Yes | ❌ Yes | ✅ No |
| **Validated** | ❌ No | ❌ No | ✅ Yes |
| **Business Value** | ❌ Low | ✅ Medium | ✅ High |

**Key Insight:** Continuous scores are an improvement, but they don't automatically make it real-world. Both synthetic and real-world approaches can use continuous scores.

The difference is:
- **Synthetic Continuous**: Predicts formula-generated scores
- **Real-World Continuous**: Predicts actual café success likelihood

---

## 5. How to Transition to Real-World

### Phase 1: Current (Done)
- ✅ Synthetic scores generated
- ✅ Models trained on synthetic data
- ❌ No real validation

### Phase 2: Switch to Continuous Regression (NEXT)
```python
# Instead of classification
reg = RandomForestRegressor()
reg.fit(X_train, synthetic_continuous_score)  # 0-100 scores
predictions = reg.predict(X_test)  # [67.3, 45.2, 88.9, ...]

# Advantages:
# ✓ No boundary artifacts
# ✓ Predictions more interpretable
# ✓ Foundation for real-world data
```

### Phase 3: Collect Real Outcome Data (6-12 months)
```
Data to Collect:
├─ Café operating status (Google Maps, web archives)
├─ Customer reviews (Google, TripAdvisor, Yelp)
├─ Foot traffic proxies (Google Popular Times, observations)
├─ Business news (local news, social media)
├─ Financial proxies (Yelp reservation activity, etc.)
└─ Owner feedback (surveys, interviews)

Mapping: Location Features → Real Outcomes
├─ Input: [competitors, density, traffic, schools, ...]
└─ Output: Success/failure, ratings, foot traffic, revenue
```

### Phase 4: Retrain on Real Data
```python
# Now using REAL outcomes
real_outcomes = ['Successful', 'Failed', 'Marginal']  # Actual data
reg = RandomForestRegressor()
reg.fit(X_real, y_real_outcomes)  # Real validation!

# Results:
# ✓ Scores = actual success likelihood
# ✓ Predictions = trustworthy
# ✓ Business can use for real decisions
```

---

## 6. Critical Gap: What's Missing?

### You HAVE:
✅ Geographic features (18 engineered features)
✅ Café locations (1,000+ samples)
✅ Synthetic labels
✅ Trained models

### You NEED (for real-world):
❌ Café operating status (open/closed)
❌ Customer satisfaction data (reviews)
❌ Business success/failure outcomes
❌ Foot traffic validation
❌ Revenue/profitability metrics
❌ Historical time series data

### How to Get It:
- 📍 **Google Maps API**: Current status, photos, reviews
- 🌐 **Web Archives**: Wayback Machine for historical data
- ⭐ **Google Reviews**: Rating = satisfaction proxy
- 📊 **Business Registries**: Operating status from government
- 🚶 **Popular Times**: Google's foot traffic heatmaps
- 📰 **News Scraping**: Café closures, openings mentioned
- 🗺️ **Street View History**: Visual confirmation of status

---

## 7. Why This Matters

### Current (Synthetic) Problems
```
Perfect Accuracy = Meaningless
├─ Model learns synthetic formula
├─ Real-world predictions unvalidated
└─ Can't trust on actual new locations
```

### Real-World (Validated) Success
```
Moderate Accuracy = Meaningful
├─ Model learns actual success patterns
├─ Predictions match real outcomes
└─ Can trust on new locations
└─ Provides business value
```

---

## Key Takeaways

1. **Continuous scores are BETTER than categories** ✅
   - Preserve information
   - No boundary artifacts
   - More interpretable

2. **But continuous doesn't mean real-world** ⚠️
   - Still synthetic if based on formula
   - Need validation against outcomes
   - Require real data to be meaningful

3. **To model real-world scenarios, you must:**
   - Collect actual café success/failure data
   - Map features to real outcomes
   - Validate predictions
   - Update continuously with new data

4. **Roadmap:**
   - Phase 1: Implement continuous regression (better structure)
   - Phase 2: Collect real outcome data (6-12 months)
   - Phase 3: Retrain on real data (validation)
   - Phase 4: Deploy with confidence (trustworthy predictions)

---

## Conclusion

**Answer to your question:**

Continuous suitability scores ≠ Real-world modeling

Instead:
- Continuous Scores + Synthetic Data = Better structure, still synthetic
- Continuous Scores + Real Data = Real-world validated model

The project currently operates at: **Continuous Scores + Synthetic Data**

To model real-world scenarios, you must reach: **Continuous Scores + Real Data**

The gap is the **outcome data**, which requires 6-12 months of collection and research.
"""

summary_path = models_dir / "CONTINUOUS_SCORE_REAL_WORLD_ANALYSIS.md"
with open(summary_path, 'w', encoding='utf-8') as f:
    f.write(summary)
print(f"✓ Summary saved: {summary_path.name}")

print(f"\n{'='*100}")
print("✅ ANALYSIS COMPLETE!")
print(f"{'='*100}")
print(f"\n📊 Key Findings:")
print(f"  1. Continuous scores are BETTER than categories ✅")
print(f"  2. But NOT automatically real-world ⚠️")
print(f"  3. Need actual café outcome data for real-world validation")
print(f"  4. Requires 6-12 months of data collection")
print(f"\n📁 Generated Files:")
print(f"  ✓ continuous_score_analysis.json")
print(f"  ✓ CONTINUOUS_SCORE_REAL_WORLD_ANALYSIS.md")
