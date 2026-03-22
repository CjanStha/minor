import pandas as pd

df = pd.read_csv('data/raw_data/combined_all_datasets.csv', low_memory=False)
print(f'Dataset: combined_all_datasets.csv')
print(f'Shape: {df.shape}')
print(f'\nAll {len(df.columns)} Columns:')
for i, col in enumerate(df.columns, 1):
    print(f'{i:2d}. {col}')
print(f'\nData Info:')
print(f'  Rows: {len(df):,}')
print(f'  Columns: {len(df.columns)}')
if 'suitability' in df.columns:
    print(f'\nSuitability (target) statistics:')
    print(f'  Non-null: {df["suitability"].notna().sum():,}')
    print(f'  Null: {df["suitability"].isna().sum():,}')
