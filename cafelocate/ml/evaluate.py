"""
Evaluate the trained ML model
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
import os

def evaluate_model():
    """
    Evaluate the trained ML model on test data
    """
    # Load the trained model
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    model_path = os.path.join(models_dir, 'rf_model.pkl')
    encoder_path = os.path.join(models_dir, 'label_encoder.pkl')

    model = joblib.load(model_path)
    label_encoder = joblib.load(encoder_path)

    # Create the same synthetic test data used for training
    np.random.seed(42)
    n_samples = 1000

    data = []
    for i in range(n_samples):
        competitor_count = np.random.randint(0, 30)
        avg_rating = np.random.uniform(3.0, 5.0)
        road_length_m = np.random.uniform(0, 5000)
        population_density = np.random.uniform(1000, 20000)

        if population_density > 12000 and competitor_count < 10:
            cafe_type = 'coffee_shop'
        elif road_length_m > 2500 and avg_rating > 4.0:
            cafe_type = 'bakery'
        elif competitor_count > 15 and population_density > 8000:
            cafe_type = 'dessert_shop'
        else:
            cafe_type = 'restaurant'

        data.append({
            'competitor_count': competitor_count,
            'avg_competitor_rating': avg_rating,
            'road_length_m': road_length_m,
            'population_density': population_density,
            'cafe_type': cafe_type
        })

    df = pd.DataFrame(data)
    feature_cols = ['competitor_count', 'avg_competitor_rating', 'road_length_m', 'population_density']
    X = df[feature_cols]
    y = df['cafe_type']
    y_encoded = label_encoder.transform(y)

    # Split with same random state as training
    _, X_test, _, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print("=== ML MODEL EVALUATION REPORT ===")
    print(f"Model: Random Forest Classifier")
    print(f"Test Set Size: {len(X_test)} samples")
    print(f"Features: {feature_cols}")
    print(f"Classes: {list(label_encoder.classes_)}")
    print()
    print(f"OVERALL ACCURACY: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print()
    print("CLASSIFICATION REPORT:")
    print("-" * 50)
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
    print()
    print("CONFUSION MATRIX:")
    print("-" * 50)
    cm = confusion_matrix(y_test, y_pred)
    print("Predicted →")
    print("Actual ↓")
    print(cm)
    print()
    print("INTERPRETATION:")
    print("-" * 50)
    if accuracy > 0.85:
        print("✓ EXCELLENT: Model performs very well on test data")
    elif accuracy > 0.75:
        print("✓ GOOD: Model performs adequately on test data")
    elif accuracy > 0.65:
        print("⚠ FAIR: Model needs improvement")
    else:
        print("✗ POOR: Model requires significant improvement")

if __name__ == "__main__":
    evaluate_model()
