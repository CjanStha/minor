import requests

overpass_url = "http://overpass-api.de/api/interpreter"

# Simple query for cafes in Kathmandu
query = """
[out:json][timeout:30];
area["name"="Kathmandu"]->.kathmandu;
node["amenity"="cafe"](area.kathmandu);
out;
"""

try:
    response = requests.post(overpass_url, data={'data': query}, timeout=30)
    print(f'Status code: {response.status_code}')
    if response.status_code == 200:
        data = response.json()
        elements = data.get('elements', [])
        print(f'Found {len(elements)} cafe elements')
        if elements:
            print('Sample element:', elements[0])
    else:
        print(f'Error response: {response.text[:200]}')
except Exception as e:
    print(f'Exception: {e}')