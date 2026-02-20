"""
Load café data into Django/PostGIS database
"""

import os
import django
import pandas as pd

# Setup Django environment
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'cafelocate.settings')
django.setup()

from api.models import Cafe

def load_cafe_data():
    """
    Load café data from CSV into the database
    """
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'kathmandu_cafes.csv')

    if not os.path.exists(csv_path):
        print(f"Error: Cafe CSV not found at {csv_path}")
        print("Run collect_data.py first to collect the data")
        return

    print("Loading café data...")

    # Read CSV
    df = pd.read_csv(csv_path)
    print(f"Found {len(df)} cafes in CSV")

    # Clear existing data
    Cafe.objects.all().delete()
    print("Cleared existing café data")

    loaded_count = 0
    for _, row in df.iterrows():
        try:
            # Create Point geometry
            lat = float(row['lat'])
            lng = float(row['lng'])
            location = {
                'type': 'Point',
                'coordinates': [lng, lat]
            }

            # Create Cafe object
            cafe = Cafe(
                place_id=row['place_id'],
                name=row['name'],
                cafe_type=row.get('type', 'cafe'),  # Default to 'cafe' if not specified
                latitude=lat,
                longitude=lng,
                location=location,
                rating=row.get('rating') if pd.notna(row.get('rating')) else None,  # Handle NaN
                review_count=int(row.get('review_count', 0)) if pd.notna(row.get('review_count')) else 0,
                is_open=row.get('is_operational', True)
            )
            cafe.save()
            loaded_count += 1

        except Exception as e:
            print(f"Error loading cafe {row.get('name', 'unknown')}: {e}")
            continue

    print(f"Successfully loaded {loaded_count} cafes")

if __name__ == "__main__":
    load_cafe_data()