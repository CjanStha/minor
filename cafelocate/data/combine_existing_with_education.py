"""Combine existing combined dataset with kathmandu_education_cleaned.csv.

This script produces a new CSV with row-wise union of both datasets.
"""

import pandas as pd
import os

base_dir = os.path.dirname(__file__)
raw_data_dir = os.path.join(base_dir, 'raw_data')

existing_path = os.path.join(raw_data_dir, 'combined_all_datasets.csv')
edu_path = os.path.join(raw_data_dir, 'kathmandu_education_cleaned.csv')
output_path = os.path.join(raw_data_dir, 'combined_all_datasets_with_education.csv')

if not os.path.exists(existing_path):
    raise FileNotFoundError(f'Existing dataset not found: {existing_path}')
if not os.path.exists(edu_path):
    raise FileNotFoundError(f'Education dataset not found: {edu_path}')

print('Loading existing combined dataset:', existing_path)
existing_df = pd.read_csv(existing_path)
print('Existing shape:', existing_df.shape)

print('Loading kathmandu_education_cleaned dataset:', edu_path)
edu_df = pd.read_csv(edu_path)
print('Education shape:', edu_df.shape)

# If existing dataset already has candidate columns from education, keep them.
# Use concat to include all rows and preserve all columns.
combined_df = pd.concat([existing_df, edu_df], axis=0, ignore_index=True, sort=False)
print('Combined shape:', combined_df.shape)

combined_df.to_csv(output_path, index=False)
print('Saved combined dataset:', output_path)
