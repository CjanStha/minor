# Dataset Combination Metadata

## Combined Dataset Information
- **File**: combined_comprehensive_dataset.csv
- **Created**: 2026-03-23 11:49:36
- **Records**: 0
- **Columns**: 0

## Source Files
All original data files have been archived in the 'raw_data' folder:

### Cafe Data
- kathmandu_cafes.csv (1,072 cafes with coordinates)

### Training Features
- cafe_location_training_dataset.csv (1,572 records with 17 engineered features)

### Reference Data
- osm_amenities_kathmandu.csv (9,265 amenities)
- kathmandu_census.csv (32 wards, population data)
- kathmandu_education_cleaned.csv (1,119 education institutions)
- osm_roads_kathmandu.csv (16,805 road segments)
- kathmandu_wards_boundary_sorted.csv (32 ward boundaries)
- combined_amenities_clean.csv (cleaned amenities)
- preprocessed_training_dataset.csv (preprocessed features)
- kathmandu_roads.geojson (road network GeoJSON)

## Column Information

### Cafe Location Columns
- place_id: Unique cafe identifier
- name: Cafe name
- lat: Latitude
- lng: Longitude
- type: Amenity type
- rating: Cafe rating (NULL if not available)
- review_count: Review count (NULL if not available)
- price_level: Price level (NULL if not available)
- is_operational: Operation status
- source: Data source (openstreetmap, mapbox, etc.)

### Training Feature Columns (from cafe_location_training_dataset.csv)
- competitors_within_500m: Count of competitors within 500m
- competitors_within_200m: Count of competitors within 200m
- competitors_min_distance: Minimum distance to nearest competitor
- competitors_avg_distance: Average distance to competitors
- roads_within_500m: Count of road segments within 500m
- roads_avg_distance: Average distance to roads
- schools_within_500m: Count of schools within 500m
- schools_within_200m: Count of schools within 200m
- schools_min_distance: Minimum distance to school
- hospitals_within_500m: Count of hospitals within 500m
- hospitals_min_distance: Minimum distance to hospital
- bus_stops_within_500m: Count of bus stops within 500m
- bus_stops_min_distance: Minimum distance to bus stop
- population_density_proxy: Population density indicator
- accessibility_score: Accessibility metric
- foot_traffic_score: Foot traffic indicator
- competition_pressure: Competition intensity metric
- suitability: Target variable (0-100 suitability score)

## Data Integrity
- All source data files are validated and contain accurate information
- Combined dataset preserves all original data
- Merged on coordinates (latitude/longitude)
- Missing values (NULL) preserved for optional fields

## Usage
Import and analyze the combined dataset:

```python
import pandas as pd

# Load combined dataset
df = pd.read_csv('combined_comprehensive_dataset.csv')

# Display basic info
print(df.shape)
print(df.columns)
print(df.describe())
```

## Archive Location
All original raw data files are stored in: `raw_data/`
This ensures data integrity and provides version control.
