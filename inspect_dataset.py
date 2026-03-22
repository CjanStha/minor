"""
Quick inspection script for combined_all_datasets.csv
"""
import pandas as pd
import os

data_path = os.path.join('cafelocate', 'data', 'raw_data', 'combined_all_datasets.csv')

print("=" * 100)
print(f"Inspecting: {data_path}")
print("=" * 100)

# Read only first few rows to check structure
df = pd.read_csv(data_path, nrows=2)

print(f"\nFile size: ~5.9 MB")
print(f"Shape (first 2 rows): {df.shape}")
print(f"\nColumn count: {len(df.columns)}")
print(f"\nColumns:")
for i, col in enumerate(df.columns, 1):
    print(f"  {i:2d}. {col}")

# Try to get full shape without loading all data
print(f"\nTrying to read full dataset...")
try:
    df_full = pd.read_csv(data_path)
    print(f"Full dataset shape: {df_full.shape}")
    print(f"Target variable (suitability) - non-null count: {df_full['suitability'].notna().sum()}")
except Exception as e:
    print(f"Error reading full dataset: {e}")
