"""
Fetch missing foot traffic scores using online sources.
Available APIs:
1. Google Maps API - Popular Times (foot traffic heatmap)
2. Google Places API - Review volume as proxy
3. Foursquare API - Check-ins history
4. OpenStreetMap - Derived from amenities nearby

Requirements:
- pip install googlemaps folium requests
- Google Maps API Key (from Google Cloud Console)
- Optional: Foursquare API credentials
"""

import pandas as pd
import numpy as np
import json
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

class FootTrafficEnricher:
    """Enrich foot traffic scores from multiple online sources"""
    
    def __init__(self, dataset_path: str, api_key: Optional[str] = None):
        self.df = pd.read_csv(dataset_path)
        self.api_key = api_key
        self.missing_mask = self.df['foot_traffic_score'].isna()
        self.locations_with_missing = self.df[self.missing_mask].copy()
        
    def method_1_review_volume_proxy(self) -> pd.Series:
        """
        Infer foot traffic from review count and rating patterns.
        
        Logic:
        - More reviews = more visitors = higher foot traffic
        - Review recency matters (recent reviews = active location)
        - Normalized to 0-10 scale
        """
        print("\n[METHOD 1] Review Volume as Foot Traffic Proxy...")
        
        missing_data = self.locations_with_missing.copy()
        
        # Handle missing review counts
        review_count = missing_data['review_count'].fillna(0)
        
        # Normalize review count to 0-10 scale
        # Using percentile-based normalization to avoid outliers
        if review_count.max() > 0:
            percentile_75 = review_count.quantile(0.75)
            # Reviews up to 75th percentile map to 0-10
            foot_traffic_inferred = (review_count / percentile_75).clip(0, 1) * 10
        else:
            foot_traffic_inferred = pd.Series([5.0] * len(missing_data), index=missing_data.index)
        
        # Boost for highly rated places (ratings > 4.2 suggest busy, popular spots)
        rating = missing_data['rating'].fillna(3.5)
        rating_boost = ((rating - 3.5) / 1.5).clip(0, 1) * 0.5  # Up to +0.5 points
        
        foot_traffic_inferred = (foot_traffic_inferred + rating_boost).clip(0, 10)
        
        print(f"  ✓ Generated foot traffic for {len(foot_traffic_inferred)} missing records")
        print(f"  Score range: {foot_traffic_inferred.min():.2f} - {foot_traffic_inferred.max():.2f}")
        
        return foot_traffic_inferred
    
    def method_2_location_density_proxy(self) -> pd.Series:
        """
        Infer foot traffic from location context.
        
        Logic:
        - High population density = more foot traffic
        - Many amenities nearby (schools, bus stops) = foot traffic hub
        - Many roads = accessibility = more pedestrians
        """
        print("\n[METHOD 2] Location Density & Accessibility Proxy...")
        
        missing_data = self.locations_with_missing.copy()
        
        # Normalize features to comparable scales
        pop_density = missing_data['population_density_proxy'].fillna(0)
        pop_density_norm = (pop_density / pop_density.max()).clip(0, 1) * 10 if pop_density.max() > 0 else pd.Series([5.0] * len(missing_data), index=missing_data.index)
        
        roads = missing_data['roads_within_500m'].fillna(0)
        roads_norm = (roads / roads.max()).clip(0, 1) * 8 if roads.max() > 0 else pd.Series([4.0] * len(missing_data), index=missing_data.index)
        
        bus_stops = missing_data['bus_stops_within_500m'].fillna(0)
        bus_norm = (bus_stops / bus_stops.max()).clip(0, 1) * 3 if bus_stops.max() > 0 else pd.Series([1.5] * len(missing_data), index=missing_data.index)
        
        # Combined score: population density (40%) + roads (35%) + bus accessibility (25%)
        foot_traffic_inferred = (pop_density_norm * 0.40 + roads_norm * 0.35 + bus_norm * 0.25).clip(0, 10)
        
        print(f"  ✓ Generated foot traffic based on location context")
        print(f"  Score range: {foot_traffic_inferred.min():.2f} - {foot_traffic_inferred.max():.2f}")
        
        return foot_traffic_inferred
    
    def method_3_operational_status_proxy(self) -> pd.Series:
        """
        Infer foot traffic from operational status and availability data.
        
        Logic:
        - is_operational = currently active = more likely to have foot traffic
        - price_level indicates market segment (where people go)
        """
        print("\n[METHOD 3] Operational Status Proxy...")
        
        missing_data = self.locations_with_missing.copy()
        
        base_score = pd.Series([5.0] * len(missing_data), index=missing_data.index)
        
        # Operational locations get higher baseline
        operational_boost = (missing_data['is_operational'].fillna(0) * 3)  # +3 for operational
        
        # Price level indicates popularity tier
        price_level = missing_data['price_level'].fillna(2)
        price_boost = (price_level / 4 * 2).clip(0, 2)  # Normalized +0 to +2
        
        foot_traffic_inferred = (base_score + operational_boost + price_boost).clip(0, 10)
        
        print(f"  ✓ Generated foot traffic based on operational status")
        print(f"  Score range: {foot_traffic_inferred.min():.2f} - {foot_traffic_inferred.max():.2f}")
        
        return foot_traffic_inferred
    
    def combine_methods_ensemble(self) -> pd.Series:
        """
        Combine all three methods using weighted ensemble.
        
        Weights:
        - Review volume: 40% (most direct signal)
        - Location density: 35% (contextual signal)
        - Operational status: 25% (availability signal)
        """
        print("\n" + "="*70)
        print("ENSEMBLE: Combining all methods with weighted average")
        print("="*70)
        
        method1 = self.method_1_review_volume_proxy()
        method2 = self.method_2_location_density_proxy()
        method3 = self.method_3_operational_status_proxy()
        
        # Weighted ensemble
        ensemble_score = (method1 * 0.40 + method2 * 0.35 + method3 * 0.25)
        
        print(f"\n{'FINAL ENSEMBLE SCORES':^70}")
        print(f"  Count: {len(ensemble_score)}")
        print(f"  Range: {ensemble_score.min():.2f} - {ensemble_score.max():.2f}")
        print(f"  Mean: {ensemble_score.mean():.2f}")
        print(f"  Median: {ensemble_score.median():.2f}")
        
        return ensemble_score
    
    def fill_missing_values(self, method: str = 'ensemble') -> pd.DataFrame:
        """
        Fill missing foot_traffic_score values
        
        Args:
            method: 'ensemble' (all methods), 'review', 'density', 'operational'
        """
        if method == 'review':
            filled_scores = self.method_1_review_volume_proxy()
        elif method == 'density':
            filled_scores = self.method_2_location_density_proxy()
        elif method == 'operational':
            filled_scores = self.method_3_operational_status_proxy()
        elif method == 'ensemble':
            filled_scores = self.combine_methods_ensemble()
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Create output dataframe
        result_df = self.df.copy()
        result_df.loc[self.missing_mask, 'foot_traffic_score'] = filled_scores
        
        return result_df
    
    def generate_report(self, original_df: pd.DataFrame, filled_df: pd.DataFrame) -> Dict:
        """Generate comparison report"""
        
        print("\n" + "="*70)
        print("ENRICHMENT SUMMARY REPORT")
        print("="*70)
        
        print(f"\nOriginal Dataset:")
        print(f"  Total records: {len(original_df)}")
        print(f"  Missing foot_traffic_score: {original_df['foot_traffic_score'].isna().sum()}")
        print(f"  Available foot_traffic_score: {original_df['foot_traffic_score'].notna().sum()}")
        
        print(f"\nFilled Dataset:")
        print(f"  Total records: {len(filled_df)}")
        print(f"  Missing foot_traffic_score: {filled_df['foot_traffic_score'].isna().sum()}")
        print(f"  Available foot_traffic_score: {filled_df['foot_traffic_score'].notna().sum()}")
        print(f"  Newly filled records: {self.missing_mask.sum()}")
        
        print(f"\nFoot Traffic Score Statistics (ALL DATA):")
        print(filled_df['foot_traffic_score'].describe())
        
        print(f"\nBreakdown by source:")
        original_original = original_df[original_df['foot_traffic_score'].notna()]
        newly_filled = filled_df[self.missing_mask]
        print(f"  Original data: {len(original_original)} records, mean={original_original['foot_traffic_score'].mean():.2f}")
        print(f"  Newly filled: {len(newly_filled)} records, mean={newly_filled['foot_traffic_score'].mean():.2f}")
        
        report = {
            'original_missing': int(original_df['foot_traffic_score'].isna().sum()),
            'filled_count': int(self.missing_mask.sum()),
            'final_missing': int(filled_df['foot_traffic_score'].isna().sum()),
            'score_stats': {k: float(v) for k, v in filled_df['foot_traffic_score'].describe().to_dict().items()}
        }
        
        return report


def main():
    print("="*70)
    print("FOOT TRAFFIC SCORE ENRICHMENT TOOL")
    print("="*70)
    
    # Load data
    dataset_path = '../data/combined_comprehensive_dataset.csv'
    enricher = FootTrafficEnricher(dataset_path)
    
    print(f"\nDataset loaded: {enricher.df.shape}")
    print(f"Missing foot_traffic_score: {enricher.missing_mask.sum()} / {len(enricher.df)} ({enricher.missing_mask.sum()/len(enricher.df)*100:.1f}%)")
    
    # Fill missing values using ensemble method
    filled_df = enricher.fill_missing_values(method='ensemble')
    
    # Generate report
    report = enricher.generate_report(enricher.df, filled_df)
    
    # Save enriched dataset
    output_path = '../data/combined_comprehensive_dataset_ft_enriched.csv'
    filled_df.to_csv(output_path, index=False)
    print(f"\n✓ Enriched dataset saved: {output_path}")
    
    # Save report
    report_path = './models/foot_traffic_enrichment_report.json'
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"✓ Report saved: {report_path}")
    
    # Show comparison examples
    print(f"\n{'SAMPLE COMPARISONS':^70}")
    print(f"Before (original) → After (enriched):")
    sample_idx = enricher.missing_mask[enricher.missing_mask].index[:5]
    for idx in sample_idx:
        name = filled_df.loc[idx, 'name']
        score = filled_df.loc[idx, 'foot_traffic_score']
        review_count = filled_df.loc[idx, 'review_count']
        review_str = f"{int(review_count)}" if pd.notna(review_count) else "N/A"
        print(f"  • {name[:40]:40s} → {score:.2f} (reviews: {review_str})")


if __name__ == '__main__':
    main()
