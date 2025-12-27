import pandas as pd
import numpy as np
import json
import os
from math import radians, cos, sin, asin, sqrt

# ===== LOAD GROWTH SCORES (for inference) =====
def load_growth_scores():
    try:
        if os.path.exists('city_growth_scores.json'):
            with open('city_growth_scores.json', 'r') as f:
                return json.load(f)
        return {}
    except Exception:
        return {}

GROWTH_SCORES = load_growth_scores()

# ===== CBD COORDINATES (City Center) =====
CBD_COORDS = {
    'bangalore': (12.9716, 77.5946),    # MG Road
    'mumbai': (18.9402, 72.8347),       # Fort
    'delhi': (28.6271, 77.2166),        # Connaught Place
    'hyderabad': (17.3850, 78.4867),    # Abids
    'chennai': (13.0827, 80.2707),      # Parrys Corner
    'pune': (18.5204, 73.8567),         # Camp
    'kolkata': (22.5726, 88.3639),      # Park Circus
    'jaipur': (26.9124, 75.7873),       # C Scheme
}

# ===== PROPERTY SUBTYPE MULTIPLIERS (vs 2BHK baseline=1.0) =====
SUBTYPE_MULTIPLIERS = {
    'residential_apartment': {
        'studio_apt': 0.70, '1bhk_apt': 0.90, '2bhk_apt': 1.00, '3bhk_apt': 1.20, '4bhk_apt': 1.50,
        'duplex_apt': 1.40, 'highrise_apt': 1.30, 'service_apt': 1.20, 'affordable_apt': 0.60,
    },
    'independent_house': {
        'independent_floor': 1.20, 'bungalow': 1.30, 'single_storey_house': 0.95, 'multistorey_house': 1.40,
    },
    'villa': {
        'luxury_villa': 2.00, 'duplex_villa': 1.60, 'triplex_villa': 1.80, 'gated_villa': 1.50, 'farm_villa': 1.10,
    },
    'row_house': {
        'townhouse': 1.10, 'cluster_housing': 1.05, 'row_villa': 1.40, 'gated_rowhouse': 1.20,
    },
    'penthouse': {
        'penthouse_single': 1.60, 'penthouse_duplex': 2.00, 'penthouse_terrace': 2.50,
    },
    'residential_plot': {
        'residential_plot': 1.00, 'commercial_plot': 1.50, 'industrial_plot': 0.80, 
        'agricultural_plot': 0.40, 'institutional_plot': 1.20, 'mixed_use_plot': 1.30,
    },
    'commercial_property': {
        'office_space': 1.50, 'it_techpark': 1.80, 'shop_showroom': 1.30,
        'warehouse_godown': 1.20, 'industrial_shed': 1.10, 'hotel_resort': 2.00, 'coworking_space': 1.40,
        'medical_clinic': 1.60, 'educational_institution': 1.70,
    }
}

def haversine(lat1, lon1, lat2, lon2):
    """Calculate distance in km between two lat/lon points."""
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    return 2 * 6371 * asin(sqrt(a))

def compute_distance_to_cbd(df, cbd_coords):
    """Add distance_to_cbd feature."""
    def get_distance(row):
        if row['city'] not in cbd_coords:
            return df['distance_to_cbd'].median() if 'distance_to_cbd' in df.columns else 10
        cbd_lat, cbd_lon = cbd_coords[row['city']]
        return haversine(row['latitude'], row['longitude'], cbd_lat, cbd_lon)
    
    df['distance_to_cbd'] = df.apply(get_distance, axis=1)
    df['distance_to_cbd'] = df['distance_to_cbd'].fillna(df['distance_to_cbd'].median())
    return df

def assign_zone(df):
    """Assign zone: Prime/Urban/Peripheral based on CBD distance."""
    df['zone'] = pd.cut(df['distance_to_cbd'], 
                        bins=[0, 5, 15, np.inf], 
                        labels=['Prime', 'Urban', 'Peripheral'])
    return df

def add_subtype_multiplier(df, subtype_multipliers):
    """Add subtype_multiplier feature."""
    def get_multiplier(row):
        try:
            return subtype_multipliers[row['property_type']][row['property_subtype']]
        except KeyError:
            return 1.0
    
    df['subtype_multiplier'] = df.apply(get_multiplier, axis=1)
    return df

def engineer_features(df, cbd_coords=CBD_COORDS, subtype_multipliers=SUBTYPE_MULTIPLIERS):
    """Apply all feature engineering."""
    df = compute_distance_to_cbd(df, cbd_coords)
    df = assign_zone(df)
    df = add_subtype_multiplier(df, subtype_multipliers)
    
    df = add_subtype_multiplier(df, subtype_multipliers)
    
    # Target: price per sqft (only if price exists, mainly for training)
    if 'price' in df.columns:
        df['price_per_sqft'] = df['price'] / df['area_sqft']
        
    # 7. Historical Growth Score (from 2000-2025 trends)
    if 'city_growth_score' not in df.columns:
        # Inference time: Lookup from JSON
        df['city_growth_score'] = df['city'].map(GROWTH_SCORES)
        df['city_growth_score'] = df['city_growth_score'].fillna(0.2) # Default conservative
    else:
        # Training time: already in CSV, just fill NA
        df['city_growth_score'] = df['city_growth_score'].fillna(0.2)
    
    return df
