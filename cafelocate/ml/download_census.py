"""
Download Nepal Census 2021 ward-level data for Kathmandu
"""

import requests
import pandas as pd
import os

def download_kathmandu_census():
    """
    Download ward-level census data for Kathmandu Metropolitan City
    from Nepal Census 2021
    """
    # Nepal Census 2021 data source
    # Note: This is a simplified version. In practice, you would need to:
    # 1. Visit https://censusnepal.cbs.gov.np/
    # 2. Download the ward-level data for Bagmati Province > Kathmandu District > Kathmandu Metropolitan City
    # 3. Extract the relevant columns

    print("Nepal Census 2021 data needs to be downloaded manually from:")
    print("https://censusnepal.cbs.gov.np/")
    print("\nSteps:")
    print("1. Go to 'Results' > 'Ward Level' > 'Population'")
    print("2. Select: Province = Bagmati, District = Kathmandu, Municipality = Kathmandu Metropolitan City")
    print("3. Download CSV with columns: ward_no, total_population, total_households")
    print("4. Save as kathmandu_census.csv in the data/ folder")

    # For now, create a sample dataset with realistic Kathmandu ward data
    # This is based on typical ward sizes in Kathmandu (population ranges from ~5k to 25k per ward)

    sample_data = []
    for ward_num in range(1, 33):  # Kathmandu has 32 wards
        # Realistic population distribution (some wards are denser)
        if ward_num <= 10:  # Core city wards
            population = 15000 + (ward_num * 500)  # 15k-20k
            households = int(population * 0.85 / 4.2)  # avg household size ~4.2
        elif ward_num <= 20:  # Mid-city wards
            population = 12000 + ((ward_num-10) * 300)  # 12k-15k
            households = int(population * 0.82 / 4.2)
        else:  # Outer wards
            population = 8000 + ((ward_num-20) * 400)  # 8k-12k
            households = int(population * 0.78 / 4.2)

        sample_data.append({
            'ward_no': ward_num,
            'population': population,
            'households': households,
            'area_sqkm': 2.5 + (ward_num % 3) * 0.5,  # 2.5-4.0 sqkm
        })

    df = pd.DataFrame(sample_data)
    df['population_density'] = df['population'] / df['area_sqkm']

    # Save to CSV
    output_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'kathmandu_census.csv')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df.to_csv(output_path, index=False)
    print(f"\nCreated sample census data with {len(df)} wards at {output_path}")
    print("⚠️  WARNING: This is SAMPLE data. Replace with real Nepal Census 2021 data!")

    return df

if __name__ == "__main__":
    download_kathmandu_census()