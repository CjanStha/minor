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
    # Prefer a reliable HTTPS Overpass endpoint; try a couple if one fails
    overpass_endpoints = [
        "https://overpass.openstreetmap.fr/api/interpreter",
        "https://overpass.kumi.systems/api/interpreter",
        "https://overpass-api.de/api/interpreter",
    ]

    # Kathmandu bounding box (more precise than area query)
    # Format: (min_lon, min_lat, max_lon, max_lat)
    # Slightly expanded Kathmandu bbox to capture edge roads
    kathmandu_bbox = "85.20,27.60,85.45,27.80"

    # Query for a wide set of highway types in the bounding box.
    # Include links, living_street, service, and others commonly used.
    highway_tags = [
        'motorway','trunk','primary','secondary','tertiary','motorway_link',
        'trunk_link','primary_link','secondary_link','tertiary_link',
        'residential','living_street','service','unclassified','road'
    ]

    query_parts = "\n".join([f"  way[\"highway\"=\"{t}\"]({kathmandu_bbox});" for t in highway_tags])

    query = f"""
    [out:json][timeout:120];
    (
{query_parts}
    );
    out geom;
    """

    print("Downloading road network from OpenStreetMap...")
    print("This may take a few minutes...")

    try:
        # Try endpoints until one succeeds
        data = None
        last_err = None
        for overpass_url in overpass_endpoints:
            try:
                response = requests.post(overpass_url, data={'data': query}, timeout=180)
                response.raise_for_status()
                data = response.json()
                break
            except Exception as e:
                last_err = e
                print(f"Endpoint {overpass_url} failed: {e}")

        if data is None:
            raise last_err

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