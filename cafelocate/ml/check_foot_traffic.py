import pandas as pd
import os

# Load dataset
data_path = '../data/combined_comprehensive_dataset.csv'
df = pd.read_csv(data_path)

print("Dataset Shape:", df.shape)
print("\nAll Columns:")
for i, col in enumerate(df.columns):
    print(f"  {i+1}. {col}")

print("\n" + "="*60)
print("FOOT TRAFFIC DATA STATUS:")
print("="*60)

if 'foot_traffic_score' in df.columns:
    print(f"\n✓ foot_traffic_score EXISTS")
    print(f"  Missing values: {df['foot_traffic_score'].isna().sum()} out of {len(df)}")
    print(f"  Data type: {df['foot_traffic_score'].dtype}")
    print(f"\n  Statistics:")
    print(df['foot_traffic_score'].describe())
else:
    print("\n✗ foot_traffic_score NOT FOUND in dataset")

# Check related traffic columns
traffic_cols = [col for col in df.columns if 'traffic' in col.lower() or 'foot' in col.lower()]
print(f"\nRelated traffic columns found: {traffic_cols if traffic_cols else 'None'}")

# Show first few rows with location data
print(f"\nFirst few rows (lat/lon and traffic data):")
location_cols = [col for col in df.columns if col in ['latitude', 'longitude', 'lat', 'lon', 'foot_traffic_score', 'traffic_score']]
if location_cols:
    print(df[location_cols].head(10))
