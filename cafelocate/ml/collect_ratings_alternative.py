"""
Alternative Café Ratings Collection - Implementation Guide
Collects café ratings from OpenStreetMap, TripAdvisor, Yelp without Google API
"""

import requests
import pandas as pd
import json
from typing import List, Dict, Optional
import os
from datetime import datetime

# ══════════════════════════════════════════════════════════════════════════════
# 1. OPENSTREETMAP - FREE (Already Integrated)
# ══════════════════════════════════════════════════════════════════════════════

def collect_osm_cafe_ratings(bbox_dict):
    """
    Extract coffee shop ratings from OpenStreetMap tags
    Most cafes in Nepal won't have ratings, but some tourist-popular ones might
    
    Args:
        bbox_dict: {'min_lat': 27.65, 'max_lat': 27.75, 'min_lng': 85.2, 'max_lng': 85.4}
    
    Returns:
        List of dicts with rating info
    """
    overpass_url = "http://overpass-api.de/api/interpreter"
    
    # Query for cafés WITH rating information
    query = f"""
    [out:json][timeout:30];
    (
      node["amenity"="cafe"](
        {bbox_dict['min_lat']},
        {bbox_dict['min_lng']},
        {bbox_dict['max_lat']},
        {bbox_dict['max_lng']}
      );
      way["amenity"="cafe"](
        {bbox_dict['min_lat']},
        {bbox_dict['min_lng']},
        {bbox_dict['max_lat']},
        {bbox_dict['max_lng']}
      );
    );
    out center;
    """
    
    try:
        response = requests.post(overpass_url, data={'data': query}, timeout=30)
        response.raise_for_status()
        osm_data = response.json()
    except requests.RequestException as e:
        print(f"Error fetching OSM data: {e}")
        return []
    
    cafes = []
    for element in osm_data.get('elements', []):
        # Extract location
        if element.get('type') == 'node':
            lat, lng = element.get('lat'), element.get('lon')
        elif 'center' in element:
            lat = element['center'].get('lat')
            lng = element['center'].get('lon')
        else:
            continue
        
        tags = element.get('tags', {})
        
        # Try to extract rating from various OSM tags
        rating = None
        if 'stars' in tags:
            try:
                rating = float(tags['stars'])
            except:
                pass
        elif 'rating' in tags:
            try:
                rating = float(tags['rating'])
            except:
                pass
        
        cafe = {
            'name': tags.get('name', 'Unknown'),
            'lat': lat,
            'lng': lng,
            'rating': rating,
            'review_count': None,
            'source': 'openstreetmap',
            'osm_id': element.get('id'),
            'website': tags.get('website'),
            'phone': tags.get('phone'),
            'opening_hours': tags.get('opening_hours'),
        }
        cafes.append(cafe)
    
    return cafes


# ══════════════════════════════════════════════════════════════════════════════
# 2. TRIPADVISOR - FREE TIER (Requires API Key)
# ══════════════════════════════════════════════════════════════════════════════

class TripAdvisorCollector:
    """
    Collect café ratings from TripAdvisor API (free tier)
    
    Setup:
    1. Go to https://developer.tripadvisor.com/
    2. Create account & register app
    3. Get API key
    4. Add to .env: TRIPADVISOR_API_KEY=your_key_here
    
    Free tier: ~100 requests/day or 3,000/month
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('TRIPADVISOR_API_KEY')
        self.base_url = "https://api.tripadvisor.com/api/restaurant/2.0"
    
    def search_cafes(self, latitude: float, longitude: float, radius_km: float = 2) -> List[Dict]:
        """
        Search for cafés near a location
        
        Args:
            latitude, longitude: Center point
            radius_km: Search radius
        
        Returns:
            List of cafés with ratings
        """
        if not self.api_key:
            print("⚠️  TripAdvisor API key not set. Skipping TripAdvisor collection.")
            return []
        
        url = f"{self.base_url}/search"
        params = {
            'key': self.api_key,
            'searchQuery': 'cafe coffee',
            'searchRadius': radius_km,
            'latLong': f'{latitude},{longitude}',
            'language': 'en'
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            results = response.json()
        except requests.RequestException as e:
            print(f"❌ TripAdvisor API error: {e}")
            return []
        
        cafes = []
        for item in results.get('data', []):
            cafe = {
                'name': item.get('name'),
                'lat': item.get('latitude'),
                'lng': item.get('longitude'),
                'rating': item.get('rating'),  # 1-5 scale
                'review_count': item.get('review_count', 0),
                'price_level': item.get('price_level'),  # $ to $$$$
                'source': 'tripadvisor',
                'tripadvisor_id': item.get('location_id'),
                'web_url': item.get('web_url'),
            }
            cafes.append(cafe)
        
        return cafes


# ══════════════════════════════════════════════════════════════════════════════
# 3. YELP - FREE TIER (Requires API Key)
# ══════════════════════════════════════════════════════════════════════════════

class YelpCollector:
    """
    Collect café ratings from Yelp API (free tier)
    
    Setup:
    1. Go to https://www.yelp.com/developers/
    2. Create account & register app
    3. Get API key
    4. Add to .env: YELP_API_KEY=your_key_here
    
    Free tier: 5,000 calls/month
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('YELP_API_KEY')
        self.base_url = "https://api.yelp.com/v3"
    
    def search_cafes(self, latitude: float, longitude: float, radius_m: float = 2000) -> List[Dict]:
        """
        Search for cafés in an area
        
        Args:
            latitude, longitude: Center point
            radius_m: Search radius in meters
        
        Returns:
            List of cafés with ratings
        """
        if not self.api_key:
            print("⚠️  Yelp API key not set. Skipping Yelp collection.")
            return []
        
        headers = {'Authorization': f'Bearer {self.api_key}'}
        url = f"{self.base_url}/businesses/search"
        params = {
            'latitude': latitude,
            'longitude': longitude,
            'radius': radius_m,
            'categories': 'cafes',
            'limit': 50,
            'sort_by': 'rating'
        }
        
        try:
            response = requests.get(url, headers=headers, params=params, timeout=10)
            response.raise_for_status()
            results = response.json()
        except requests.RequestException as e:
            print(f"❌ Yelp API error: {e}")
            return []
        
        cafes = []
        for item in results.get('businesses', []):
            cafe = {
                'name': item.get('name'),
                'lat': item.get('coordinates', {}).get('latitude'),
                'lng': item.get('coordinates', {}).get('longitude'),
                'rating': item.get('rating'),  # 1-5 scale
                'review_count': item.get('review_count', 0),
                'price_level': len(item.get('price', '')) if item.get('price') else None,  # $ scale
                'source': 'yelp',
                'yelp_id': item.get('id'),
                'phone': item.get('phone'),
                'website': item.get('website'),
                'url': item.get('url'),
                'categories': [cat.get('title') for cat in item.get('categories', [])],
            }
            cafes.append(cafe)
        
        return cafes


# ══════════════════════════════════════════════════════════════════════════════
# 4. HYBRID COLLECTOR - Combines All Sources
# ══════════════════════════════════════════════════════════════════════════════

class HybridCafeCollector:
    """
    Collects café data from multiple sources and intelligently merges them
    """
    
    def __init__(self):
        self.tripadvisor = TripAdvisorCollector()
        self.yelp = YelpCollector()
    
    def collect_all(self, latitude: float, longitude: float, radius_km: float = 2) -> pd.DataFrame:
        """
        Collect café data from all available sources
        
        Priority for ratings:
        1. TripAdvisor (most reliable for tourism/cafés)
        2. Yelp (good quality reviews)
        3. OpenStreetMap (community-driven)
        4. Aggregate if multiple sources
        
        Returns:
            DataFrame with deduplicated cafés and best available ratings
        """
        print("🔍 Collecting café data from multiple sources...\n")
        
        results = {
            'osm': [],
            'tripadvisor': [],
            'yelp': [],
        }
        
        # 1. OpenStreetMap (always available)
        print("1️⃣  OpenStreetMap...", end=" ")
        bbox = {
            'min_lat': latitude - 0.018,  # ~2 km
            'max_lat': latitude + 0.018,
            'min_lng': longitude - 0.018,
            'max_lng': longitude + 0.018,
        }
        osm_cafes = collect_osm_cafe_ratings(bbox)
        results['osm'] = osm_cafes
        print(f"✓ Found {len(osm_cafes)} cafés")
        
        # 2. TripAdvisor (if API key available)
        print("2️⃣  TripAdvisor...", end=" ")
        ta_cafes = self.tripadvisor.search_cafes(latitude, longitude, radius_km)
        results['tripadvisor'] = ta_cafes
        print(f"✓ Found {len(ta_cafes)} cafés")
        
        # 3. Yelp (if API key available)
        print("3️⃣  Yelp...", end=" ")
        yelp_cafes = self.yelp.search_cafes(latitude, longitude, radius_km * 1000)
        results['yelp'] = yelp_cafes
        print(f"✓ Found {len(yelp_cafes)} cafés")
        
        # 4. Merge and deduplicate
        print("\n4️⃣  Merging and deduplicating...", end=" ")
        merged_df = self._merge_results(results)
        print(f"✓ Final count: {len(merged_df)} unique cafés")
        
        return merged_df
    
    def _merge_results(self, results: Dict[str, List[Dict]]) -> pd.DataFrame:
        """
        Merge cafés from multiple sources, de-duplicate by location
        """
        merged = {}
        
        # Process each source in priority order
        for source in ['osm', 'tripadvisor', 'yelp']:
            for cafe in results.get(source, []):
                if cafe['lat'] is None or cafe['lng'] is None:
                    continue
                
                # Use location as key (rounded to ~100m)
                key = (round(cafe['lat'], 4), round(cafe['lng'], 4))
                
                if key not in merged:
                    merged[key] = {
                        'name': cafe['name'],
                        'lat': cafe['lat'],
                        'lng': cafe['lng'],
                        'sources': {},
                        'ratings': []
                    }
                
                # Track this source
                merged[key]['sources'][source] = cafe
                
                # Collect ratings
                if cafe.get('rating'):
                    merged[key]['ratings'].append({
                        'source': source,
                        'rating': cafe['rating'],
                        'count': cafe.get('review_count', 0)
                    })
        
        # Convert to DataFrame with best rating
        final_records = []
        for (lat, lng), cafe_data in merged.items():
            if not cafe_data['ratings']:
                continue
            
            # Prefer TripAdvisor, then Yelp, then OSM
            rating_by_source = {r['source']: r for r in cafe_data['ratings']}
            
            if 'tripadvisor' in rating_by_source:
                best_rating = rating_by_source['tripadvisor']
                rating_source = 'tripadvisor'
            elif 'yelp' in rating_by_source:
                best_rating = rating_by_source['yelp']
                rating_source = 'yelp'
            else:
                best_rating = rating_by_source['osm']
                rating_source = 'osm'
            
            # Aggregate review count from all sources
            total_reviews = sum(r['count'] for r in cafe_data['ratings'])
            
            record = {
                'name': cafe_data['name'],
                'latitude': lat,
                'longitude': lng,
                'rating': best_rating['rating'],
                'review_count': best_rating['count'],
                'total_review_sources': len(cafe_data['ratings']),
                'primary_rating_source': rating_source,
                'all_sources': list(cafe_data['sources'].keys()),
                'rating_confidence': len(cafe_data['ratings']),  # How many sources
            }
            final_records.append(record)
        
        df = pd.DataFrame(final_records)
        
        # Sort by rating confidence and rating
        if len(df) > 0:
            df = df.sort_values(
                by=['rating_confidence', 'rating'],
                ascending=[False, False]
            )
        
        return df


# ══════════════════════════════════════════════════════════════════════════════
# 5. USAGE EXAMPLES
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    
    # Kathmandu city center coordinates
    KATHMANDU_CENTER = {
        'lat': 27.7172,
        'lng': 85.3240
    }
    
    # Example 1: Using only OpenStreetMap (FREE, no API key needed)
    print("=" * 80)
    print("EXAMPLE 1: OpenStreetMap Only (FREE)")
    print("=" * 80)
    bbox = {
        'min_lat': 27.65,
        'max_lat': 27.75,
        'min_lng': 85.2,
        'max_lng': 85.4,
    }
    osm_cafes = collect_osm_cafe_ratings(bbox)
    df_osm = pd.DataFrame(osm_cafes)
    print(f"\nFound {len(df_osm)} cafés with OSM data")
    print(f"Cafés with ratings: {df_osm['rating'].notna().sum()}")
    print("\nSample:")
    print(df_osm[['name', 'rating', 'website']].head(10))
    
    # Example 2: Using Hybrid Approach (if API keys available)
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Hybrid Collector (OSM + TripAdvisor + Yelp)")
    print("=" * 80)
    collector = HybridCafeCollector()
    df_merged = collector.collect_all(
        KATHMANDU_CENTER['lat'],
        KATHMANDU_CENTER['lng'],
        radius_km=2
    )
    
    if len(df_merged) > 0:
        print("\nTop-rated cafés:")
        print(df_merged[['name', 'rating', 'review_count', 'primary_rating_source']].head(10))
        
        print("\nRating source distribution:")
        print(df_merged['primary_rating_source'].value_counts())
        
        print(f"\nAverage rating: {df_merged['rating'].mean():.2f}")
        print(f"Cafés with multiple source ratings: {(df_merged['total_review_sources'] > 1).sum()}")
        
        # Save to CSV
        output_file = 'kathmandu_cafes_with_ratings_hybrid.csv'
        df_merged.to_csv(output_file, index=False)
        print(f"\n✅ Saved to {output_file}")
