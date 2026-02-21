"""
Download Kathmandu road network from OpenStreetMap
"""

import requests
import json
import os

def download_kathmandu_roads():
    """
    Download road network data for Kathmandu from OpenStreetMap
    """
    overpass_url = "http://overpass-api.de/api/interpreter"

    # Kathmandu bounding box (more precise than area query)
    # Format: (min_lon, min_lat, max_lon, max_lat)
    kathmandu_bbox = "85.25,27.65,85.40,27.75"

    # Query for roads in Kathmandu bounding box
    # This includes major roads that would affect café accessibility
    query = f"""
    [out:json][timeout:60];
    (
      way["highway"="primary"]({kathmandu_bbox});
      way["highway"="secondary"]({kathmandu_bbox});
      way["highway"="tertiary"]({kathmandu_bbox});
      way["highway"="residential"]({kathmandu_bbox});
      way["highway"="unclassified"]({kathmandu_bbox});
    );
    out geom;
    """

    print("Downloading road network from OpenStreetMap...")
    print("This may take a few minutes...")

    try:
        response = requests.post(overpass_url, data={'data': query}, timeout=120)
        response.raise_for_status()
        data = response.json()

        # Convert to GeoJSON format
        geojson = {
            "type": "FeatureCollection",
            "features": []
        }

        for element in data.get('elements', []):
            if element.get('type') == 'way' and 'geometry' in element:
                # Convert OSM way to GeoJSON LineString
                coordinates = []
                for node in element['geometry']:
                    coordinates.append([node['lon'], node['lat']])

                feature = {
                    "type": "Feature",
                    "properties": {
                        "osm_id": element['id'],
                        "highway": element.get('tags', {}).get('highway', 'unclassified'),
                        "name": element.get('tags', {}).get('name', ''),
                        "lanes": element.get('tags', {}).get('lanes', ''),
                        "maxspeed": element.get('tags', {}).get('maxspeed', ''),
                    },
                    "geometry": {
                        "type": "LineString",
                        "coordinates": coordinates
                    }
                }
                geojson["features"].append(feature)

        # Save to file
        output_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'kathmandu_roads.geojson')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(geojson, f, indent=2, ensure_ascii=False)

        print(f"✓ Downloaded {len(geojson['features'])} road segments")
        print(f"  Saved to {output_path}")

        return geojson

    except requests.RequestException as e:
        print(f"Error downloading road data: {e}")
        print("Creating fallback sample data...")

        # Fallback sample data
        sample_roads = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "LineString",
                        "coordinates": [
                            [85.30, 27.70], [85.32, 27.71], [85.34, 27.72]
                        ]
                    },
                    "properties": {
                        "highway": "primary",
                        "name": "Sample Main Road",
                        "lanes": "2"
                    }
                }
            ]
        }

        output_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'kathmandu_roads.geojson')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(sample_roads, f, indent=2)

        print(f"✓ Created sample road data with {len(sample_roads['features'])} segments")
        print(f"  Saved to {output_path}")
        print("⚠️  WARNING: Using sample data. Real OSM download failed!")

        return sample_roads

if __name__ == "__main__":
    download_kathmandu_roads()