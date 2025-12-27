import joblib
import json
import pandas as pd
import numpy as np
import logging

from feature_engineering import engineer_features, CBD_COORDS, SUBTYPE_MULTIPLIERS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HybridInferenceEngine:
    """Load models, apply hybrid logic, confidence scoring."""
    
    def __init__(self, lr_model_path, xgb_model_path, metadata_path):
        logger.info("Loading models...")
        self.lr_pipeline = joblib.load(lr_model_path)
        self.xgb_pipeline = joblib.load(xgb_model_path)
        
        with open(metadata_path) as f:
            self.metadata = json.load(f)
        
        logger.info("✅ Engine ready")
    
    def _prepare_input(self, input_dict):
        """Prepare input dictionary with engineered features."""
        # Create a copy to avoid modifying original
        prepared = input_dict.copy()
        
        # Normalize strings
        prepared['city'] = prepared['city'].lower().strip()
        prepared['state'] = prepared['state'].lower().strip()
        prepared['pincode'] = str(prepared['pincode']).zfill(6)
        prepared['property_type'] = prepared['property_type'].lower().strip()
        prepared['property_subtype'] = prepared['property_subtype'].lower().strip()
        
        # Calculate distance_to_cbd
        from math import radians, cos, sin, asin, sqrt
        def haversine(lat1, lon1, lat2, lon2):
            lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
            dlat, dlon = lat2 - lat1, lon2 - lon1
            a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
            return 2 * 6371 * asin(sqrt(a))
        
        if prepared['city'] in CBD_COORDS:
            cbd_lat, cbd_lon = CBD_COORDS[prepared['city']]
            prepared['distance_to_cbd'] = haversine(
                prepared['latitude'], prepared['longitude'], cbd_lat, cbd_lon
            )
        else:
            prepared['distance_to_cbd'] = 10  # Default
        
        # Assign zone
        if prepared['distance_to_cbd'] <= 5:
            prepared['zone'] = 'Prime'
        elif prepared['distance_to_cbd'] <= 15:
            prepared['zone'] = 'Urban'
        else:
            prepared['zone'] = 'Peripheral'
        
        # Add subtype_multiplier
        try:
            prepared['subtype_multiplier'] = SUBTYPE_MULTIPLIERS[prepared['property_type']][prepared['property_subtype']]
        except KeyError:
            prepared['subtype_multiplier'] = 1.0
        
        return prepared
    
    def predict_single(self, input_dict):
        """Predict for single property."""
        
        # Prepare DataFrame from input
        prepared = input_dict.copy()
        prepared['pincode'] = str(prepared['pincode']).zfill(6) # Ensure string
        
        # Text normalization
        for col in ['city', 'state', 'property_type', 'property_subtype']:
             if isinstance(prepared.get(col), str):
                prepared[col] = prepared[col].lower().strip()
                
        df_input = pd.DataFrame([prepared])
        
        # Apply centralized feature engineering (adds distance, zone, multipliers, growth_score)
        df_input = engineer_features(df_input)
        
        # Dual predictions
        try:
            lr_pps = self.lr_pipeline.predict(df_input)[0]  # price per sqft
            xgb_pps = self.xgb_pipeline.predict(df_input)[0]
        except Exception as e:
            logger.error(f"Model prediction failed: {e}")
            # Fallback for debugging - check columns
            logger.error(f"Input columns: {df_input.columns.tolist()}")
            raise e
        
        area = float(prepared['area_sqft'])
        lr_price = lr_pps * area
        xgb_price = xgb_pps * area
        
        # Disagreement (normalized by average)
        avg_price = (lr_price + xgb_price) / 2
        delta = abs(xgb_price - lr_price) / max(avg_price, 1)
        
        # Adaptive weighting
        w_lr = 0.7 if delta > 0.25 else 0.4
        w_xgb = 1 - w_lr
        
        # Hybrid price is the final price
        hybrid_price = w_lr * lr_price + w_xgb * xgb_price
        hybrid_pps = hybrid_price / area
        
        # Confidence scoring
        conf_score, conf_level, price_band = self._compute_confidence(
            prepared['pincode'], delta, prepared['city']
        )
        
        return {
            'linear_price': float(lr_price),
            'xgb_price': float(xgb_price),
            'hybrid_price': float(hybrid_price),
            'final_price': float(hybrid_price),
            'final_price_per_sqft': float(hybrid_pps),
            'subtype_multiplier': float(prepared.get('subtype_multiplier', 1.0)),
            'confidence_score': float(conf_score),
            'confidence_level': conf_level,
            'price_band': price_band,
            'currency': 'INR',
            'market': 'India'
        }
    
    def _compute_confidence(self, pincode, model_disagreement, city):
        """
        Confidence scoring based on:
        1. Data coverage - whether pincode exists in training data (35 points)
        2. Model accuracy - MAE relative to mean price (35 points)
        3. Model agreement - how close LR and XGB predictions are (30 points)
        
        Returns score (0-100), level (High/Medium/Low), and price band.
        """
        mae_by_pincode = self.metadata.get('mae_by_pincode', {})
        mean_pps = self.metadata.get('mean_price_per_sqft', 6000)
        global_mae = self.metadata.get('xgb_mae', 2000)
        
        # Component 1: Data Coverage (35 points max)
        # Check if pincode exists in training data
        pincode_str = str(pincode)
        if pincode_str in mae_by_pincode:
            # Pincode was in training - high confidence
            coverage_score = 35
        else:
            # Check if city is a known major city
            major_cities = ['bangalore', 'mumbai', 'delhi', 'hyderabad', 'chennai', 
                          'pune', 'kolkata', 'jaipur', 'ahmedabad', 'noida', 'gurgaon']
            if city.lower() in major_cities:
                coverage_score = 20  # Major city but unknown pincode
            else:
                coverage_score = 10  # Unknown location
        
        # Component 2: Model Accuracy (35 points max)
        # Lower MAE relative to mean price = higher score
        pincode_mae = mae_by_pincode.get(pincode_str, global_mae * 1.5)
        mae_ratio = pincode_mae / mean_pps  # Typical: 0.1 to 0.5
        
        if mae_ratio < 0.15:
            accuracy_score = 35  # Excellent accuracy
        elif mae_ratio < 0.25:
            accuracy_score = 28  # Good accuracy
        elif mae_ratio < 0.35:
            accuracy_score = 20  # Fair accuracy
        elif mae_ratio < 0.50:
            accuracy_score = 12  # Below average
        else:
            accuracy_score = 5   # Poor accuracy
        
        # Component 3: Model Agreement (30 points max)
        # Low disagreement between models = higher confidence
        if model_disagreement < 0.10:
            agreement_score = 30  # Models strongly agree
        elif model_disagreement < 0.20:
            agreement_score = 24  # Models mostly agree
        elif model_disagreement < 0.35:
            agreement_score = 16  # Some disagreement
        elif model_disagreement < 0.50:
            agreement_score = 8   # Significant disagreement
        else:
            agreement_score = 2   # Models widely disagree
        
        # Total score
        total = coverage_score + accuracy_score + agreement_score
        total = min(100, max(0, total))
        
        # Determine level and band
        if total >= 75:
            level = 'High'
            band = '±5%'
        elif total >= 50:
            level = 'Medium'
            band = '±10%'
        else:
            level = 'Low'
            band = '±20%'
        
        return total, level, band
