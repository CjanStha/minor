import os
import pandas as pd

print('=' * 100)
print('FINAL DATA STRUCTURE')
print('=' * 100)

print('\n📁 MAIN DATA FOLDER (cafelocate/data/)')
print('-' * 100)

main_files = [f for f in os.listdir('.') if os.path.isfile(f) and not f.startswith('create_')]
for f in sorted(main_files):
    if f.endswith('.csv'):
        df = pd.read_csv(f)
        size = os.path.getsize(f) / 1024
        print(f'{f:50s} | {len(df):>6,d} records | {len(df.columns):>2d} cols | {size:>10.1f} KB')
    elif f.endswith('.md'):
        size = os.path.getsize(f) / 1024
        print(f'{f:50s} | Documentation')

print('\n📁 RAW_DATA FOLDER (cafelocate/data/raw_data/)')
print('-' * 100)

if os.path.exists('raw_data'):
    raw_files = [f for f in os.listdir('raw_data')]
    for f in sorted(raw_files):
        path = os.path.join('raw_data', f)
        size = os.path.getsize(path) / 1024
        if f.endswith('.csv'):
            df = pd.read_csv(path)
            print(f'{f:50s} | {len(df):>6,d} records | {len(df.columns):>2d} cols | {size:>10.1f} KB')
        else:
            print(f'{f:50s} | {size:>10.1f} KB')

print('\n' + '=' * 100)
print('COMBINED DATASET DETAILS')
print('=' * 100)

combined = pd.read_csv('combined_comprehensive_dataset.csv')
print(f'\nFile: combined_comprehensive_dataset.csv')
print(f'Records: {len(combined):,}')
print(f'Columns: {len(combined.columns)}')
print(f'\nColumn List:')
for i, col in enumerate(combined.columns, 1):
    print(f'  {i:2d}. {col}')

print(f'\nData Summary:')
print(f'  Cafés covered: {len(combined):,}')
print(f'  Geographic range: {combined["lat"].min():.4f} to {combined["lat"].max():.4f}°N')
print(f'                    {combined["lng"].min():.4f} to {combined["lng"].max():.4f}°E')
print(f'  Missing values: {combined.isnull().sum().sum():,}')
