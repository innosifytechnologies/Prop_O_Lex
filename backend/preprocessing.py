from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def build_preprocessing_pipeline():
    """
    Build reusable ColumnTransformer + feature lists.
    Returns: (preprocessor, num_features, cat_features)
    """
    
    numerical_features = [
        'latitude', 'longitude', 'distance_to_cbd',
        'area_sqft', 'bedrooms', 'bathrooms', 'balconies', 'age', 
        'subtype_multiplier', 'city_growth_score', 'floor', 
        'total_floors', 'floor_ratio', 'amenities_count', 'is_new_construction',
        'is_rera_registered'
    ]
    
    categorical_features = [
        'city', 'state', 'pincode', 'property_type', 'property_subtype', 
        'zone', 'furnishing', 'parking', 'building_type'
    ]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ]
    )
    
    return preprocessor, numerical_features, categorical_features
