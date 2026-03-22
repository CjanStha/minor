"""
Load ward boundary data into Django/PostGIS database
"""

import os
import django
import pandas as pd
import json

# Setup Django environment
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'cafelocate.settings')
django.setup()

from api.models import Ward

try:
    from shapely.wkt import loads as wkt_loads
    from shapely.geometry import mapping
    SHAPELY_AVAILABLE = True
except ImportError:
    SHAPELY_AVAILABLE = False

def load_ward_boundaries():
    """
    Load ward boundary data from CSV into the database
    """
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'kathmandu_wards_boundary_sorted.csv')

    if not os.path.exists(csv_path):
        print(f"Error: Ward boundary CSV not found at {csv_path}")
        return

    print("Loading ward boundary data...")

    # Read CSV
    df = pd.read_csv(csv_path)
    print(f"Found {len(df)} wards in CSV")

    # Clear existing data
    Ward.objects.all().delete()
    print("Cleared existing ward data")

    loaded_count = 0
    for _, row in df.iterrows():
        try:
            # Use the ward_number column directly (sequential 1-32)
            ward_number = int(row['ward_number'])

            # Parse WKT geometry and convert to GeoJSON
            wkt_geom = row['geometry_wkt']
            
            # Convert WKT to GeoJSON using shapely
            geometry_json = None
            if SHAPELY_AVAILABLE:
                try:
                    geom = wkt_loads(wkt_geom)
                    geometry_json = mapping(geom)  # Convert to GeoJSON
                except Exception as e:
                    print(f"Warning: Could not convert WKT for ward {ward_number}: {e}")
                    # Fallback to storing WKT
                    geometry_json = {
                        'type': 'wkt',
                        'wkt': wkt_geom
                    }
            else:
                # Fallback: store WKT as-is
                geometry_json = {
                    'type': 'wkt',
                    'wkt': wkt_geom
                }

            # Create Ward object
            ward = Ward(
                ward_number=ward_number,
                population=0,  # Will be updated when census data is loaded
                households=0,
                area_sqkm=0.0,  # Will be calculated from geometry
                population_density=0.0,
                boundary=geometry_json
            )
            ward.save()
            loaded_count += 1

        except Exception as e:
            print(f"Error loading ward {row.get('ward_id', 'unknown')}: {e}")
            continue

    print(f"Successfully loaded {loaded_count} ward boundaries")

if __name__ == "__main__":
    load_ward_boundaries()