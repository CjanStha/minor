#!/usr/bin/env python
"""
ML Training Startup Script
Checks environment, verifies data, and optionally runs training
"""

import os
import sys
import subprocess
from pathlib import Path
import pandas as pd

def print_header(text):
    """Print section header"""
    print("\n" + "="*80)
    print(f"  {text}")
    print("="*80)

def check_environment():
    """Check if all required packages are installed"""
    print_header("CHECKING ENVIRONMENT")
    
    required_packages = {
        'pandas': 'Data manipulation',
        'numpy': 'Numerical computing',
        'sklearn': 'Machine learning',
        'xgboost': 'XGBoost algorithm',
        'joblib': 'Model serialization'
    }
    
    missing = []
    
    for package, description in required_packages.items():
        try:
            __import__(package)
            print(f"  ✅ {package:15} - {description}")
        except ImportError:
            print(f"  ❌ {package:15} - {description} (MISSING)")
            missing.append(package)
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print("\nInstall them with:")
        print(f"  pip install {' '.join(missing)}")
        return False
    
    print("\n  ✅ All dependencies available")
    return True

def check_data():
    """Check if combined dataset exists and is valid"""
    print_header("CHECKING DATASET")
    
    data_path = Path("cafelocate/data/combined_comprehensive_dataset.csv")
    
    if not data_path.exists():
        print(f"  ❌ Dataset not found: {data_path}")
        return False
    
    print(f"  ✅ Dataset found: {data_path.name}")
    
    try:
        df = pd.read_csv(data_path)
        print(f"\n  📊 Dataset Statistics:")
        print(f"     - Records: {len(df):,}")
        print(f"     - Columns: {len(df.columns)}")
        print(f"     - File size: {data_path.stat().st_size / 1024 / 1024:.2f} MB")
        
        # Check required columns
        required_cols = ['suitability', 'competitors_within_500m', 'population_density_proxy']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"  ❌ Missing columns: {missing_cols}")
            return False
        
        print(f"  ✅ All required columns present")
        
        # Check missing values in target
        if df['suitability'].isnull().any():
            null_count = df['suitability'].isnull().sum()
            print(f"  ⚠️  Warning: {null_count} missing target values")
            return False
        
        print(f"  ✅ No missing target values")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error reading dataset: {e}")
        return False

def check_ml_scripts():
    """Check if all ML scripts exist"""
    print_header("CHECKING ML SCRIPTS")
    
    required_scripts = {
        'train_xgboost_comparison.py': 'Main training script (recommended)',
        'train_model.py': 'Random Forest training',
        'evaluate.py': 'Model evaluation',
        'compare_models.py': 'Model comparison',
        'get_importances.py': 'Feature importance',
        'preprocess_data.py': 'Data preprocessing'
    }
    
    ml_dir = Path("cafelocate/ml")
    
    if not ml_dir.exists():
        print(f"  ❌ ML directory not found: {ml_dir}")
        return False
    
    all_exist = True
    for script, description in required_scripts.items():
        script_path = ml_dir / script
        if script_path.exists():
            print(f"  ✅ {script:35} - {description}")
        else:
            print(f"  ❌ {script:35} - NOT FOUND")
            all_exist = False
    
    return all_exist

def run_training():
    """Run the main training script"""
    print_header("STARTING MODEL TRAINING")
    
    print("\n  📊 This will:")
    print("     1. Load and preprocess data (80/20 split)")
    print("     2. Train Random Forest (200 estimators)")
    print("     3. Train XGBoost (100 estimators)")
    print("     4. Evaluate both models")
    print("     5. Compare performance")
    print("     6. Save best model")
    print("\n  ⏱️  Estimated time: 5-10 minutes\n")
    
    response = input("  Continue with training? (yes/no): ").strip().lower()
    
    if response not in ['yes', 'y']:
        print("\n  ❌ Training cancelled")
        return False
    
    os.chdir("cafelocate/ml")
    
    try:
        print("\n  🚀 Running: python train_xgboost_comparison.py\n")
        result = subprocess.run(
            [sys.executable, 'train_xgboost_comparison.py'],
            capture_output=False
        )
        
        if result.returncode == 0:
            print("\n  ✅ Training completed successfully!")
            return True
        else:
            print("\n  ❌ Training encountered errors")
            return False
            
    except Exception as e:
        print(f"\n  ❌ Error running training: {e}")
        return False

def verify_models():
    """Check if models were created"""
    print_header("VERIFYING TRAINED MODELS")
    
    models_dir = Path("cafelocate/ml/models")
    
    if not models_dir.exists():
        print("  ❌ Models directory not found")
        return False
    
    required_models = {
        'xgboost_model.pkl': 'XGBoost model (primary)',
        'rf_model.pkl': 'Random Forest model (backup)',
        'scaler.pkl': 'Feature scaler',
        'label_encoder.pkl': 'Label encoder'
    }
    
    all_exist = True
    for model, description in required_models.items():
        model_path = models_dir / model
        if model_path.exists():
            size_mb = model_path.stat().st_size / 1024 / 1024
            print(f"  ✅ {model:25} - {description:40} ({size_mb:.2f} MB)")
        else:
            print(f"  ⚠️  {model:25} - {description:40} (NOT FOUND)")
            all_exist = all_exist and (model in ['rf_model.pkl', 'label_encoder.pkl'])
    
    return all_exist

def main():
    """Main execution"""
    print("\n")
    print("╔" + "="*78 + "╗")
    print("║" + " "*78 + "║")
    print("║" + "  🤖 CAFELOCATE ML TRAINING STARTUP SCRIPT".center(78) + "║")
    print("║" + "  Random Forest & XGBoost Model Training".center(78) + "║")
    print("║" + " "*78 + "║")
    print("╚" + "="*78 + "╝")
    
    # Run checks
    checks = {
        'Environment': check_environment,
        'Dataset': check_data,
        'ML Scripts': check_ml_scripts,
    }
    
    all_passed = True
    for check_name, check_func in checks.items():
        if not check_func():
            all_passed = False
    
    # Summary
    print_header("SUMMARY")
    
    if all_passed:
        print("\n  ✅ All checks passed! Ready for training.\n")
        
        response = input("  Start training now? (yes/no/setup): ").strip().lower()
        
        if response in ['yes', 'y']:
            if run_training():
                if verify_models():
                    print_header("TRAINING SUCCESSFUL! 🎉")
                    print("\n  ✅ Models trained and saved")
                    print("\n  Next steps:")
                    print("     1. Review results in cafelocate/ml/models/")
                    print("     2. Integrate into Django API")
                    print("     3. Test predictions")
                    print("     4. Deploy to production")
                    return 0
        
        elif response == 'setup':
            print("\n  📋 SETUP INSTRUCTIONS:")
            print("     1. Install missing packages: pip install xgboost scikit-learn")
            print("     2. Verify dataset exists")
            print("     3. Run: python setup_ml.py")
            return 0
        
        else:
            print("\n  ℹ️  You can run training manually:")
            print("     cd cafelocate/ml")
            print("     python train_xgboost_comparison.py")
            return 0
    
    else:
        print("\n  ❌ Some checks failed. Please fix issues above.\n")
        print("  For help, see:")
        print("     - MODEL_TRAINING_GUIDE.md")
        print("     - TRAINING_CHECKLIST.md")
        print("     - TRAINING_WORKFLOW.md")
        return 1

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n  ⛔ Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n  ❌ Unexpected error: {e}")
        sys.exit(1)
