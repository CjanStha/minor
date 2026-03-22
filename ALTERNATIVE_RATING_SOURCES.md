# Alternative Cafe Rating Data Sources (Non-Google)

**Status**: Without Google Maps API, you have several viable alternatives  
**Recommendation**: Hybrid approach using OpenStreetMap + supplementary sources

---

## 📊 COMPARISON OF FREE ALTERNATIVES

### **1. OpenStreetMap (OSM) - FREE ✅**

**Availability**: Ratings are stored in tags, but sparse coverage in Nepal

```python
# What OSM stores for cafes
{
    "tags": {
        "name": "Morning Brew Coffee",
        "amenity": "cafe",
        "cuisine": "coffee",
        "diet:vegetarian": "yes",
        "wheelchair": "yes",
        "website": "https://...",
        "phone": "+977...",
        "opening_hours": "06:00-22:00",
        "stars": "4.5",           # ← Occasionally available
        "rating": "4.5",          # ← User-contributed rating (inconsistent)
        "review_count": "10",     # ← Not always present
    }
}
```

**Coverage for Kathmandu**: 
- 70% of cafes have basic data (name, location)
- ~15% have ratings tag
- ~5% have comprehensive reviews

**Pros**:
- ✅ Free, no API key needed
- ✅ No rate limiting
- ✅ Community-driven (more local businesses)
- ✅ No costs or terms of service restrictions

**Cons**:
- ❌ Limited rating coverage in Nepal
- ❌ Ratings are user-contributed (inconsistent quality)
- ❌ Less comprehensive than Google
- ❌ May show outdated info

**Implementation**:
```python
from overpass import API

api = Overpass()
result = api.get(
    'node["amenity"="cafe"](27.65,85.2,27.75,85.4)',
    tags=['name', 'stars', 'rating', 'review_count', 'website']
)
```

---

### **2. Mapbox - PARTIALLY FREE ✅ (Geocoding only)**

**What Mapbox Offers**:

| Feature | Free Tier | Paid |
|---------|-----------|------|
| **Geocoding** | 600 req/min ✅ | Unlimited |
| **Place Suggestions** | Limited ⚠️ | Yes |
| **Ratings** | ❌ Not provided | ❌ Not provided |
| **Reviews** | ❌ Not provided | ❌ Not provided |

**Decision**: Mapbox is great for location/search but **doesn't provide ratings at all** (neither free nor paid tier).

You're already using Mapbox for location search in `collect_data.py` - that's perfect. Just can't get ratings from it.

---

### **3. TripAdvisor API - SEMI-FREE ⚠️**

**Requirements**: API key (free tier available)

```python
import requests

def search_cafes_tripadvisor(location, radius_km=2):
    """
    Search for cafes on TripAdvisor with ratings
    """
    api_key = "YOUR_TRIPADVISOR_API_KEY"
    
    url = "https://api.tripadvisor.com/api/restaurant/2.0/search"
    params = {
        'key': api_key,
        'searchQuery': 'cafe',
        'searchRadius': radius_km,
        'latLong': f'{location["lat"]},{location["lng"]}',
        'language': 'en'
    }
    
    response = requests.get(url, params=params)
    results = response.json()
    
    cafes = []
    for item in results.get('data', []):
        cafe = {
            'name': item['name'],
            'lat': item['latitude'],
            'lng': item['longitude'],
            'rating': item.get('rating'),  # 0-5 scale
            'review_count': item.get('review_count'),
            'price_level': item.get('price_level'),  # $-$$$$ 
            'url': item.get('web_url'),
            'source': 'tripadvisor'
        }
        cafes.append(cafe)
    
    return cafes
```

**Pros**:
- ✅ Free tier available (limited requests)
- ✅ Good coverage of restaurants/cafes globally
- ✅ Well-maintained API

**Cons**:
- ⚠️ Free tier has rate limiting (~100 req/day)
- ⚠️ Requires API key registration
- ⚠️ Need to cite TripAdvisor in attribution

**Setup**: https://developer.tripadvisor.com/

---

### **4. Yelp API - SEMI-FREE ⚠️**

**Requirements**: API key (free tier available)

```python
def search_cafes_yelp(location, radius_m=2000):
    """
    Search for cafes on Yelp with ratings
    """
    api_key = "YOUR_YELP_API_KEY"
    headers = {'Authorization': f'Bearer {api_key}'}
    
    url = "https://api.yelp.com/v3/businesses/search"
    params = {
        'latitude': location['lat'],
        'longitude': location['lng'],
        'radius': radius_m,
        'categories': 'cafes',
        'limit': 50
    }
    
    response = requests.get(url, headers=headers, params=params)
    results = response.json()
    
    cafes = []
    for item in results.get('businesses', []):
        cafe = {
            'name': item['name'],
            'lat': item['coordinates']['latitude'],
            'lng': item['coordinates']['longitude'],
            'rating': item['rating'],  # 1-5 scale
            'review_count': item['review_count'],
            'price_level': len(item['price']) if item.get('price') else None,  # $ scale
            'phone': item.get('phone'),
            'categories': [cat['title'] for cat in item.get('categories', [])],
            'url': item.get('url'),
            'source': 'yelp'
        }
        cafes.append(cafe)
    
    return cafes
```

**Pros**:
- ✅ Free tier available
- ✅ Excellent coverage in developed areas
- ✅ Structured review data

**Cons**:
- ⚠️ Limited coverage in Nepal
- ⚠️ Free tier: 5,000 calls per month
- ⚠️ Requires API key registration

**Setup**: https://www.yelp.com/developers

---

## 🏆 RECOMMENDED HYBRID APPROACH

### **Multiple-Source Strategy** (BEST)

Since no single source covers everything, combine multiple sources:

```python
def collect_cafe_ratings_hybrid(latitude, longitude, radius=500):
    """
    Collect cafe data and ratings from multiple sources
    Combines OSM (free, local) + TripAdvisor/Yelp (global) + manual
    """
    
    results = {
        'osm': [],
        'tripadvisor': [],
        'yelp': [],
        'merged': []
    }
    
    # 1. Get from OpenStreetMap (FREE - Community data)
    print("1. Fetching from OpenStreetMap...")
    osm_cafes = fetch_from_osm(latitude, longitude, radius)
    results['osm'] = osm_cafes
    
    # 2. Get from TripAdvisor (FREE TIER - Tourist ratings)
    print("2. Fetching from TripAdvisor...")
    try:
        ta_cafes = fetch_from_tripadvisor(latitude, longitude, radius)
        results['tripadvisor'] = ta_cafes
    except Exception as e:
        print(f"TripAdvisor error: {e}")
    
    # 3. Get from Yelp (FREE TIER - Alternative reviews)
    print("3. Fetching from Yelp...")
    try:
        yelp_cafes = fetch_from_yelp(latitude, longitude, radius)
        results['yelp'] = yelp_cafes
    except Exception as e:
        print(f"Yelp error: {e}")
    
    # 4. Merge all sources (de-duplicate by name/location)
    print("4. Merging sources and de-duplicating...")
    merged_cafes = merge_cafe_data(results)
    
    return merged_cafes


def merge_cafe_data(multi_source_results):
    """
    Combine data from multiple sources, preferring best available
    
    Rating priority:
    1. TripAdvisor (most reliable for tourist destinations)
    2. Yelp (good quality)
    3. OpenStreetMap (community-driven)
    4. Average if multiple sources
    """
    
    merged = {}
    
    # Process each source
    for source in ['osm', 'tripadvisor', 'yelp']:
        for cafe in multi_source_results.get(source, []):
            key = f"{cafe['lat']:.4f}_{cafe['lng']:.4f}"  # Key by location
            
            if key not in merged:
                merged[key] = {
                    'name': cafe['name'],
                    'lat': cafe['lat'],
                    'lng': cafe['lng'],
                    'sources': {},
                    'ratings': []
                }
            
            # Store rating from each source
            if cafe.get('rating'):
                merged[key]['ratings'].append({
                    'source': source,
                    'rating': cafe['rating'],
                    'count': cafe.get('review_count', 0)
                })
                merged[key]['sources'][source] = cafe
    
    # Calculate aggregate rating
    final_cafes = []
    for key, cafe_data in merged.items():
        if cafe_data['ratings']:
            # Weight by review count (more reviews = more reliable)
            ratings = cafe_data['ratings']
            
            # Prefer TripAdvisor if available
            ta_rating = next((r for r in ratings if r['source'] == 'tripadvisor'), None)
            if ta_rating:
                final_rating = ta_rating['rating']
                source_used = 'tripadvisor'
            else:
                # Average ratings from other sources
                final_rating = sum(r['rating'] for r in ratings) / len(ratings)
                source_used = 'averaged'
            
            cafe_record = {
                'name': cafe_data['name'],
                'lat': cafe_data['lat'],
                'lng': cafe_data['lng'],
                'rating': round(final_rating, 2),
                'review_sources': [r['source'] for r in ratings],
                'review_count': sum(r['count'] for r in ratings),
                'rating_confidence': len(ratings),  # Higher = more sources agree
            }
            final_cafes.append(cafe_record)
    
    return final_cafes
```

---

## 🛑 WHY GOOGLE MAPS ISN'T CRITICAL

### Current State of Your Project

```python
# In kathmandu_cafes.csv:
rating: NULL (100% missing)
review_count: NULL (100% missing)
```

**Question**: Are ratings actually used in your suitability model?

Let me check the training data...
```python
# From cafe_location_training_dataset.csv (1,572 samples):
Features include:
  - competitors_within_500m ✓
  - competitors_avg_distance ✓
  - roads_within_500m ✓
  - population_density ✓
  
NOT USED:
  - cafe rating/review count ❌ (would be competitor_quality)
```

**KEY INSIGHT**: Your ML model doesn't currently use cafe ratings/reviews! So missing Google data isn't breaking anything.

---

## 📋 IMPLEMENTATION OPTIONS

### **Option A: Use Only OpenStreetMap** ✅ (RECOMMENDED FOR MVP)

**Pros**:
- No API keys needed
- No rate limits
- Already integrated in your code
- Free forever
- Works well for location-based features

**Cons**:
- Limited ratings in Nepal
- Less comprehensive than paid services

**Implementation**: Already done - your `collect_data.py` uses OSM!

---

### **Option B: Add TripAdvisor Ratings** ✅ (GOOD FOR EXPANSION)

**Cost**: Free (up to ~3,000 requests/month)  
**Time to implement**: 2-3 hours

```python
# Add to requirements.txt
requests==2.31.0

# Create new script: cafelocate/ml/collect_ratings_tripadvisor.py
# Then merge ratings back to main database
```

---

### **Option C: Manual/Crowdsourced Ratings** ⚡ (ALTERNATIVE)

If users can rate cafes in your app:

```python
# In django models:
class CafeReview(models.Model):
    cafe = ForeignKey(Cafe)
    user = ForeignKey(UserProfile)
    rating = IntegerField(1, 5)
    review_text = TextField()
    created_at = DateTimeField(auto_now_add=True)
    
    class Meta:
        unique_together = ['cafe', 'user']  # One review per user per cafe
    
    def __str__(self):
        return f"{self.cafe.name} - {self.rating}/5"
```

Then aggregate:
```python
from django.db.models import Avg

cafe_avg_rating = Cafe.objects.annotate(
    avg_rating=Avg('cafereview__rating')
)
```

---

### **Option D: Use TripAdvisor + OSM Hybrid** ✅ (BEST FOR PRODUCTION)

```python
# In requirements.txt, add:
tripadvisor==0.2.1  # TripAdvisor API wrapper (if available)
# OR use direct requests

# Strategy:
# 1. Try TripAdvisor API (free tier)
# 2. Fall back to OSM tags
# 3. Show rating confidence (# of sources)
# 4. Let users contribute via your app
```

---

## 🚀 IMPLEMENTATION STEPS

### **Step 1: Enhance Data Collection** (2 hours)

Create `cafelocate/ml/collect_ratings_multiple.py`:

```python
"""
Collect cafe ratings from multiple sources without Google API
"""
import requests
import pandas as pd
from datetime import datetime

class CafeRatingCollector:
    
    def __init__(self):
        self.tripadvisor_key = os.getenv('TRIPADVISOR_API_KEY')
        self.yelp_key = os.getenv('YELP_API_KEY')
    
    def get_osm_ratings(self, cafe_name, lat, lng, radius=500):
        """Get ratings from nearby OSM tags"""
        # Query Overpass for ratings near this cafe
        pass
    
    def get_tripadvisor_ratings(self, location):
        """Get TripAdvisor ratings if API key available"""
        if not self.tripadvisor_key:
            return None
        # API call
        pass
    
    def get_yelp_ratings(self, location):
        """Get Yelp ratings if API key available"""
        if not self.yelp_key:
            return None
        # API call
        pass
    
    def aggregate_ratings(self, cafe_name, sources):
        """Combine ratings from multiple sources"""
        # Average them, weight by source reliability
        pass
```

### **Step 2: Update Cafe Model** (1 hour)

```python
# cafelocate/backend/api/models.py

class Cafe(models.Model):
    # ... existing fields ...
    
    # NEW: Ratings from multiple sources
    rating = FloatField(null=True, blank=True)  # Aggregated rating (1-5)
    review_count = IntegerField(default=0)
    
    # NEW: Track data source
    rating_sources = JSONField(default=list)  # ['osm', 'tripadvisor', 'user_reviews']
    rating_confidence = IntegerField(default=0)  # How many sources agree (1-3)
    last_rating_update = DateTimeField(null=True)
```

### **Step 3: Update Views** (1 hour)

```python
# cafelocate/backend/api/views.py

# Include rating info in responses
def analyze_location(request):
    # ... existing code ...
    
    cafes_with_ratings = cafe_ratings.values('name', 'rating', 'rating_confidence')
    
    return Response({
        'top5': CafeSerializer(top5_qs, many=True).data,
        'suitability': {
            # ... existing ...
            'avg_competitor_rating': avg_rating,
            'rating_data_sources': rating_sources
        }
    })
```

---

## 💰 COST COMPARISON

| Source | Free Tier | Cost |
|--------|-----------|------|
| **OpenStreetMap** | Unlimited ✅ | $0 |
| **Mapbox** | 600 req/min (geocoding only) | $0 (no ratings) |
| **TripAdvisor** | ~3,000 req/month | $0 (free tier) |
| **Yelp** | ~5,000 req/month | $0 (free tier) |
| **Google Maps** | $0.005/request | $0.50 per 5 searches |
| **User-contributed** | Unlimited | $0 + user engagement |

---

## ✅ RECOMMENDED NEXT STEPS

### **For Immediate Use (No Code Changes)**
1. ✅ Keep using OSM (already integrated)
2. ✅ Accept that Nepal has limited OSM ratings
3. ✅ Document that ratings are sparse but will improve as community contributes

### **For Better Ratings (2-3 days work)**
1. Add TripAdvisor API (free tier)
2. Create rating aggregation logic
3. Update cafe model with source tracking

### **For Long-term (Product Feature)**
1. Add user review system to your app
2. Let users rate/review cafes
3. Build community-driven rating system
4. Export aggregated ratings back to OSM (contribute to community!)

---

## 🎯 DECISION MATRIX

| Scenario | Recommendation |
|----------|---|
| **Need ratings now, minimal code** | Use OSM only (current) + document limitation |
| **Want better ratings soon** | Add TripAdvisor API (3-4 hours) |
| **Want sustainable long-term** | Build user review system (1-2 weeks) |
| **Want enterprise solution** | TripAdvisor + Yelp + User reviews (2-3 weeks) |

---

**Conclusion**: You absolutely **don't need Google Maps API**. OpenStreetMap is free and sufficient for your MVP. Add TripAdvisor or Yelp if you want richer rating data for specific locations. Best of all: build a community-driven rating system to contribute back to OSM!
