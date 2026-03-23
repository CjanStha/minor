from .models import Ward


def point_in_polygon(point_lng, point_lat, polygon_geojson):
    """
    Ray-casting algorithm to check if a point is inside a GeoJSON polygon or WKT geometry.
    Handles both GeoJSON and WKT formats.
    """
    if not polygon_geojson:
        return False

    geom_type = polygon_geojson.get('type')

    if geom_type == 'wkt':
        return _point_in_wkt_polygon(point_lng, point_lat, polygon_geojson.get('wkt', ''))

    coordinates = polygon_geojson.get('coordinates', [])
    if not coordinates:
        return False

    rings = []
    if geom_type == 'Polygon':
        rings = [coordinates[0]] if coordinates else []
    elif geom_type == 'MultiPolygon':
        for polygon in coordinates:
            if polygon:
                rings.append(polygon[0])
    else:
        return False

    for exterior_ring in rings:
        n = len(exterior_ring)
        inside = False
        p1x, p1y = exterior_ring[0]
        for i in range(1, n + 1):
            p2x, p2y = exterior_ring[i % n]
            if point_lat > min(p1y, p2y):
                if point_lat <= max(p1y, p2y):
                    if point_lng <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (point_lat - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or point_lng <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        if inside:
            return True

    return False


def _point_in_wkt_polygon(point_lng, point_lat, wkt_string):
    """
    Parse WKT polygon string and check if point is inside using ray-casting.
    Handles POLYGON and MULTIPOLYGON WKT formats.
    """
    import re

    if not wkt_string:
        return False

    wkt_string = wkt_string.strip().upper()

    if wkt_string.startswith('MULTIPOLYGON'):
        coord_pattern = r'\(\(([^)]+)\)\)'
        matches = re.findall(coord_pattern, wkt_string)
        polygons = []
        for match in matches:
            coords = _parse_wkt_coords(match)
            if coords:
                polygons.append(coords)
    elif wkt_string.startswith('POLYGON'):
        coord_pattern = r'\(\(([^)]+)\)\)'
        match = re.search(coord_pattern, wkt_string)
        if match:
            coords = _parse_wkt_coords(match.group(1))
            polygons = [coords] if coords else []
        else:
            polygons = []
    else:
        return False

    for ring in polygons:
        if len(ring) < 3:
            continue

        n = len(ring)
        inside = False
        p1x, p1y = ring[0]

        for i in range(1, n + 1):
            p2x, p2y = ring[i % n]
            if point_lat > min(p1y, p2y):
                if point_lat <= max(p1y, p2y):
                    if point_lng <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (point_lat - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or point_lng <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y

        if inside:
            return True

    return False


def _parse_wkt_coords(coord_string):
    """
    Parse WKT coordinate string: 'lng lat, lng lat, ...'
    Returns list of (lng, lat) tuples.
    """
    import re

    coord_pattern = r'(-?\d+\.?\d*)\s+(-?\d+\.?\d*)'
    matches = re.findall(coord_pattern, coord_string)

    coords = []
    for lng_str, lat_str in matches:
        try:
            coords.append((float(lng_str), float(lat_str)))
        except ValueError:
            continue

    return coords


def is_within_kathmandu_metropolitan_city(lat, lng):
    """
    Return True when the point lies inside any stored Kathmandu ward boundary.
    """
    for ward in Ward.objects.all().only('boundary'):
        if ward.boundary and isinstance(ward.boundary, dict):
            if point_in_polygon(lng, lat, ward.boundary):
                return True
    return False
