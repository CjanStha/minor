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

    # Query for roads in Kathmandu area
    # This includes major roads that would affect café accessibility
    query = """
    [out:json][timeout:60];
    area["name"="Kathmandu"]->.kathmandu;
    (
      way["highway"="primary"](area.kathmandu);
      way["highway"="secondary"](area.kathmandu);
      way["highway"="tertiary"](area.kathmandu);
      way["highway"="residential"](area.kathmandu);
      way["highway"="unclassified"](area.kathmandu);
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

        print(f"Downloaded {len(geojson['features'])} road segments")
        print(f"Saved to {output_path}")

        return geojson

    except requests.RequestException as e:
        print(f"Error downloading road data: {e}")
        return None

if __name__ == "__main__":
    download_kathmandu_roads()