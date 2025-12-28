import pandas as pd
import numpy as np
import logging
from math import radians, cos, sin, asin, sqrt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===== COLUMN MAPPING =====
COLUMN_MAPPING = {
    'City': 'city',
    'Locality': 'locality',
    'PropertyType': 'property_type',
    'BHK': 'bedrooms',
    'Bathrooms': 'bathrooms',
    'Balconies': 'balconies',
    'Furnishing': 'furnishing',
    'Parking': 'parking',
    'BuildingType': 'building_type',
    'AmenitiesCount': 'amenities_count',
    'IsRERARegistered': 'is_rera_registered',
    'Floor': 'floor',
    'TotalFloors': 'total_floors',
    'AgeYears': 'age',
    'Latitude': 'latitude',
    'Longitude': 'longitude',
    'Price': 'price',
    'Size_SqFt': 'area_sqft',
    'SuperBuiltUpArea_sqft': 'super_built_up_area',
    'BuiltUpArea_sqft': 'built_up_area',
    'CarpetArea_sqft': 'carpet_area',
}

# ===== MAJOR CITY COORDINATES & PINCODES =====
# Format: (lat, lon, state, default_pincode, radius_km)
CITY_LOCATIONS = {
    'bangalore': (12.9716, 77.5946, 'karnataka', '560001', 50),
    'mumbai': (19.0760, 72.8777, 'maharashtra', '400001', 60),
    'delhi': (28.6139, 77.2090, 'delhi', '110001', 40),
    'chennai': (13.0827, 80.2707, 'tamil nadu', '600001', 50),
    'hyderabad': (17.3850, 78.4867, 'telangana', '500001', 50),
    'pune': (18.5204, 73.8567, 'maharashtra', '411001', 40),
    'kolkata': (22.5726, 88.3639, 'west bengal', '700001', 50),
    'ahmedabad': (23.0225, 72.5714, 'gujarat', '380001', 40),
    'jaipur': (26.9124, 75.7873, 'rajasthan', '302001', 40),
    'gurgaon': (28.4595, 77.0266, 'haryana', '122001', 30),
    'noida': (28.5355, 77.3910, 'uttar pradesh', '201301', 25),
    'ghaziabad': (28.6692, 77.4538, 'uttar pradesh', '201001', 20),
    'thane': (19.2183, 72.9781, 'maharashtra', '400601', 25),
    'navi mumbai': (19.0330, 73.0297, 'maharashtra', '400703', 25),
    'lucknow': (26.8467, 80.9462, 'uttar pradesh', '226001', 35),
    'chandigarh': (30.7333, 76.7794, 'chandigarh', '160001', 20),
    'kochi': (9.9312, 76.2673, 'kerala', '682001', 30),
    'coimbatore': (11.0168, 76.9558, 'tamil nadu', '641001', 30),
    'indore': (22.7196, 75.8577, 'madhya pradesh', '452001', 30),
    'bhopal': (23.2599, 77.4126, 'madhya pradesh', '462001', 30),
    'nagpur': (21.1458, 79.0882, 'maharashtra', '440001', 30),
    'surat': (21.1702, 72.8311, 'gujarat', '395001', 30),
    'vadodara': (22.3072, 73.1812, 'gujarat', '390001', 30),
    'visakhapatnam': (17.6868, 83.2185, 'andhra pradesh', '530001', 30),
    'patna': (25.5941, 85.1376, 'bihar', '800001', 30),
    'mysore': (12.2958, 76.6394, 'karnataka', '570001', 25),
    'mangalore': (12.9141, 74.8560, 'karnataka', '575001', 25),
    'vijayawada': (16.5062, 80.6480, 'andhra pradesh', '520001', 25),
    'bhubaneswar': (20.2961, 85.8245, 'odisha', '751001', 25),
    'ranchi': (23.3441, 85.3096, 'jharkhand', '834001', 25),
    'goa': (15.2993, 74.1240, 'goa', '403001', 40),
    'dehradun': (30.3165, 78.0322, 'uttarakhand', '248001', 25),
    'shimla': (31.1048, 77.1734, 'himachal pradesh', '171001', 20),
    'trivandrum': (8.5241, 76.9366, 'kerala', '695001', 25),
}

# ===== STATE BOUNDING BOXES (lat_min, lat_max, lon_min, lon_max) =====
STATE_BOUNDS = {
    'karnataka': (11.5, 18.5, 74.0, 78.5),
    'maharashtra': (15.6, 22.0, 72.6, 80.9),
    'tamil nadu': (8.0, 13.5, 76.2, 80.4),
    'telangana': (15.8, 19.9, 77.2, 81.3),
    'andhra pradesh': (12.6, 19.1, 76.7, 84.8),
    'kerala': (8.2, 12.8, 74.8, 77.4),
    'delhi': (28.4, 28.9, 76.8, 77.4),
    'uttar pradesh': (23.5, 30.4, 77.1, 84.6),
    'rajasthan': (23.0, 30.2, 69.5, 78.3),
    'gujarat': (20.1, 24.7, 68.2, 74.5),
    'west bengal': (21.5, 27.2, 85.8, 89.9),
    'haryana': (27.6, 30.9, 74.5, 77.6),
    'punjab': (29.5, 32.5, 73.9, 76.9),
    'madhya pradesh': (21.1, 26.9, 74.0, 82.8),
    'bihar': (24.3, 27.5, 83.3, 88.2),
    'jharkhand': (21.9, 25.3, 83.3, 87.9),
    'odisha': (17.8, 22.6, 81.4, 87.5),
    'chhattisgarh': (17.8, 24.1, 80.2, 84.4),
    'goa': (14.9, 15.8, 73.7, 74.5),
    'assam': (24.0, 28.0, 89.7, 96.0),
    'uttarakhand': (28.7, 31.5, 77.6, 81.0),
    'himachal pradesh': (30.4, 33.3, 75.6, 79.0),
    'jammu and kashmir': (32.3, 37.1, 73.3, 80.3),
    'chandigarh': (30.6, 30.8, 76.7, 76.9),
}

def haversine(lat1, lon1, lat2, lon2):
    """Calculate distance in km between two lat/lon points."""
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    return 2 * 6371 * asin(sqrt(a))

def infer_location_from_latlon(lat, lon):
    """
    Infer state and pincode from latitude/longitude.
    Uses nearest city matching and state bounding boxes.
    Returns: (state, pincode)
    """
    if pd.isna(lat) or pd.isna(lon):
        return 'karnataka', '560001'  # Default
    
    # Step 1: Find nearest city within radius
    best_city = None
    best_distance = float('inf')
    
    for city_name, (city_lat, city_lon, state, pincode, radius) in CITY_LOCATIONS.items():
        distance = haversine(lat, lon, city_lat, city_lon)
        if distance < radius and distance < best_distance:
            best_distance = distance
            best_city = (state, pincode)
    
    if best_city:
        return best_city
    
    # Step 2: Fall back to state bounding box
    for state_name, (lat_min, lat_max, lon_min, lon_max) in STATE_BOUNDS.items():
        if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max:
            # Find default pincode for this state (from nearest major city in state)
            for city_name, (city_lat, city_lon, city_state, pincode, _) in CITY_LOCATIONS.items():
                if city_state == state_name:
                    return state_name, pincode
            return state_name, '000001'
    
    # Step 3: Default to Karnataka/Bangalore
    return 'karnataka', '560001'

def infer_property_subtype(row):
    """Infer property_subtype from property_type and bedrooms."""
    ptype = str(row.get('property_type', '')).lower()
    bhk = row.get('bedrooms', 2)
    if pd.isna(bhk):
        bhk = 2
    bhk = int(bhk)
    
    if 'apartment' in ptype or 'flat' in ptype:
        if bhk == 0:
            return 'studio_apt'
        elif bhk == 1:
            return '1bhk_apt'
        elif bhk == 2:
            return '2bhk_apt'
        elif bhk == 3:
            return '3bhk_apt'
        else:
            return '4bhk_apt'
    elif 'villa' in ptype:
        return 'luxury_villa'
    elif 'house' in ptype or 'independent' in ptype:
        return 'independent_floor'
    elif 'penthouse' in ptype:
        return 'penthouse_single'
    elif 'plot' in ptype or 'land' in ptype:
        return 'individual_plot'
    elif 'commercial' in ptype or 'office' in ptype or 'shop' in ptype:
        return 'office_space'
    elif 'row' in ptype or 'townhouse' in ptype:
        return 'townhouse'
    else:
        return '2bhk_apt'

def normalize_property_type(ptype):
    """Normalize property type to one of the 7 main types."""
    ptype = str(ptype).lower().strip()
    
    if 'apartment' in ptype or 'flat' in ptype:
        return 'residential_apartment'
    elif 'villa' in ptype:
        return 'villa'
    elif 'house' in ptype or 'independent' in ptype or 'bungalow' in ptype:
        return 'independent_house'
    elif 'penthouse' in ptype:
        return 'penthouse'
    elif 'plot' in ptype or 'land' in ptype:
        return 'residential_plot'
    elif 'commercial' in ptype or 'office' in ptype or 'shop' in ptype or 'retail' in ptype:
        return 'commercial_property'
    elif 'row' in ptype or 'townhouse' in ptype:
        return 'row_house'
    else:
        return 'residential_apartment'

def load_and_validate(csv_path):
    """
    Load CSV, validate schema, clean outliers, normalize strings.
    Infers state and pincode from latitude/longitude.
    Returns: (cleaned_df, validation_report_dict)
    """
    logger.info(f"Loading: {csv_path}")
    df = pd.read_csv(csv_path, low_memory=False)
    
    initial_count = len(df)
    logger.info(f"Loaded {initial_count} rows")
    
    # ===== RENAME COLUMNS =====
    df = df.rename(columns=COLUMN_MAPPING)
    
    # ===== COALESCE AREA COLUMNS =====
    if 'area_sqft' in df.columns:
        if 'super_built_up_area' in df.columns:
            df['area_sqft'] = df['area_sqft'].fillna(df['super_built_up_area'])
        if 'built_up_area' in df.columns:
            df['area_sqft'] = df['area_sqft'].fillna(df['built_up_area'])
        if 'carpet_area' in df.columns:
            df['area_sqft'] = df['area_sqft'].fillna(df['carpet_area'])
    
    # ===== CONVERT TO NUMERIC FIRST =====
    df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
    df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
    df['area_sqft'] = pd.to_numeric(df['area_sqft'], errors='coerce')
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    df['bedrooms'] = pd.to_numeric(df['bedrooms'], errors='coerce')
    df['bathrooms'] = pd.to_numeric(df['bathrooms'], errors='coerce')
    df['balconies'] = pd.to_numeric(df.get('balconies'), errors='coerce')
    df['floor'] = pd.to_numeric(df.get('floor'), errors='coerce')
    df['total_floors'] = pd.to_numeric(df.get('total_floors'), errors='coerce')
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    df['amenities_count'] = pd.to_numeric(df.get('amenities_count'), errors='coerce')
    
    # ===== IMPUTE EARLY (before filtering) =====
    df['bedrooms'] = df['bedrooms'].fillna(2)
    df['bathrooms'] = df['bathrooms'].fillna(2)
    df['balconies'] = df.get('balconies', pd.Series(dtype='float64')).fillna(1)
    df['floor'] = df.get('floor', pd.Series(dtype='float64')).fillna(3)
    df['total_floors'] = df.get('total_floors', pd.Series(dtype='float64')).fillna(6)
    df['age'] = df['age'].fillna(5)
    df['amenities_count'] = df.get('amenities_count', pd.Series(dtype='float64')).fillna(6)
    df['is_rera_registered'] = df.get('is_rera_registered', pd.Series(dtype='float64')).fillna(0).astype(int)
    
    # Fill categorical
    df['furnishing'] = df.get('furnishing', pd.Series(dtype='object')).fillna('Semi-Furnished')
    df['parking'] = df.get('parking', pd.Series(dtype='object')).fillna('None')
    df['building_type'] = df.get('building_type', pd.Series(dtype='object')).fillna('Gated Community')
    
    # ===== DROP ROWS WITH MISSING CRITICAL VALUES =====
    before_drop = len(df)
    df = df.dropna(subset=['latitude', 'longitude', 'area_sqft', 'price', 'city'])
    logger.info(f"Dropped {before_drop - len(df)} rows with missing lat/lon/area/price/city")
    
    # ===== INFER STATE AND PINCODE FROM LAT/LON =====
    logger.info("Inferring state and pincode from latitude/longitude...")
    location_info = df.apply(
        lambda row: infer_location_from_latlon(row['latitude'], row['longitude']), 
        axis=1
    )
    df['state'] = location_info.apply(lambda x: x[0])
    df['pincode'] = location_info.apply(lambda x: x[1])
    logger.info(f"Inferred {df['state'].nunique()} unique states from coordinates")
    
    # ===== NORMALIZE PROPERTY TYPE =====
    df['property_type'] = df['property_type'].apply(normalize_property_type)
    
    # ===== INFER PROPERTY SUBTYPE =====
    df['property_subtype'] = df.apply(infer_property_subtype, axis=1)
    
    # ===== BOUNDS & SANITY CHECKS =====
    before_bounds = len(df)
    
    # India geographic bounds
    df = df[(df['latitude'].between(6, 37)) & (df['longitude'].between(68, 98))]
    logger.info(f"Geo filter: {before_bounds} -> {len(df)}")
    
    # Realistic property bounds
    df = df[df['area_sqft'] > 100]
    df = df[df['price'] > 100000]
    df = df[df['bedrooms'].between(0, 10)]
    df = df[df['bathrooms'].between(0, 10)]
    df = df[df['age'].between(-1, 100)]
    
    # ===== NORMALIZATION =====
    df['city'] = df['city'].str.lower().str.strip()
    df['state'] = df['state'].str.lower().str.strip()
    df['pincode'] = df['pincode'].astype(str).str.zfill(6)
    df['property_type'] = df['property_type'].str.lower().str.strip()
    df['property_subtype'] = df['property_subtype'].str.lower().str.strip()
    
    final_count = len(df)
    
    # ===== REPORT =====
    report = {
        'initial_rows': initial_count,
        'final_rows': final_count,
        'rows_dropped': initial_count - final_count,
        'pct_retained': (final_count / initial_count * 100) if initial_count > 0 else 0,
        'mean_price': df['price'].mean() if len(df) > 0 else 0,
        'median_price': df['price'].median() if len(df) > 0 else 0,
        'mean_area': df['area_sqft'].mean() if len(df) > 0 else 0,
        'unique_cities': df['city'].nunique(),
        'unique_states': df['state'].nunique(),
        'unique_types': df['property_type'].nunique(),
        'unique_subtypes': df['property_subtype'].nunique(),
    }
    
    logger.info(f"✅ Data validation complete:")
    logger.info(f"   Initial: {report['initial_rows']} → Final: {report['final_rows']} ({report['pct_retained']:.1f}%)")
    if final_count > 0:
        logger.info(f"   Mean price: ₹{report['mean_price']:,.0f}")
        logger.info(f"   States: {report['unique_states']}, Cities: {report['unique_cities']}")
    
    return df, report

if __name__ == '__main__':
    df, report = load_and_validate('consolidated_properties.csv')
    print(f"\nValidation Report:\n{report}")
    if len(df) > 0:
        print(f"\nState distribution:")
        print(df['state'].value_counts().head(10))
        print(f"\nSample data:")
        print(df[['latitude', 'longitude', 'city', 'state', 'pincode', 'property_type', 'area_sqft', 'price']].head())
