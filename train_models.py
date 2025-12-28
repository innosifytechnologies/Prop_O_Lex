import pandas as pd
import numpy as np
import json
import os
import logging
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
import joblib

from backend.data_loading import load_and_validate
from backend.feature_engineering import engineer_features, CBD_COORDS, SUBTYPE_MULTIPLIERS
from backend.preprocessing import build_preprocessing_pipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main(csv_path='data/consolidated_properties.csv', output_dir='./models'):
    """Complete training pipeline."""
    
    # ===== 1. LOAD & VALIDATE =====
    logger.info("Step 1: Loading & validating data...")
    df, report = load_and_validate(csv_path)
    logger.info(f"✅ Validation: {report['final_rows']} rows, {report['unique_cities']} cities")
    
    # ===== 2. FEATURE ENGINEERING =====
    logger.info("Step 2: Engineering features...")
    df = engineer_features(df, CBD_COORDS, SUBTYPE_MULTIPLIERS)
    logger.info(f"✅ Features: distance_to_cbd, zone, subtype_multiplier, price_per_sqft")
    
    # ===== 3. PREPARE X, y =====
    logger.info("Step 3: Preparing features & target...")
    target_col = 'price_per_sqft'
    drop_cols = ['price', 'price_per_sqft']  # Don't leak target
    X = df.drop(columns=drop_cols)
    y = df[target_col]
    
    logger.info(f"✅ X shape: {X.shape}, y shape: {y.shape}")
    logger.info(f"   Target (price_per_sqft) - Mean: ₹{y.mean():.0f}, Std: ₹{y.std():.0f}")
    
    # ===== 4. TRAIN/VAL SPLIT =====
    logger.info("Step 4: Train/val split (80/20)...")
    # Use stratified split if enough samples per city, otherwise random
    try:
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=X['city'])
    except ValueError:
        # Fallback if stratification fails
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    logger.info(f"✅ Train: {len(X_train)}, Val: {len(X_val)}")
    
    # ===== 5. BUILD PIPELINES =====
    logger.info("Step 5: Building preprocessing pipelines...")
    preprocessor, num_feats, cat_feats = build_preprocessing_pipeline()
    logger.info(f"✅ Numerical: {len(num_feats)}, Categorical: {len(cat_feats)}")
    
    # ===== 6. TRAIN GRADIENT BOOSTING REGRESSOR =====
    logger.info("Step 6: Training Gradient Boosting Regressor (v2)...")
    gb_model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        min_samples_split=10,
        min_samples_leaf=5,
        subsample=0.8,
        random_state=42,
        validation_fraction=0.1,
        n_iter_no_change=10,
        verbose=0
    )
    gb_pipeline = Pipeline([('preprocessor', preprocessor), ('model', gb_model)])
    gb_pipeline.fit(X_train, y_train)
    
    gb_r2 = gb_pipeline.score(X_val, y_val)
    gb_pred = gb_pipeline.predict(X_val)
    gb_mae = np.mean(np.abs(gb_pred - y_val))
    logger.info(f"✅ Gradient Boosting - R²: {gb_r2:.4f}, MAE: ₹{gb_mae:.0f}/sqft")
    
    # ===== 7. SAVE MODELS =====
    logger.info(f"Step 7: Saving models to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(gb_pipeline, f'{output_dir}/linear_model.pkl') # Keep filename for compatibility or update engine
    joblib.dump(gb_pipeline, f'{output_dir}/xgb_model.pkl')
    logger.info(f"✅ Saved updated models (GradientBoosting)")
    
    # ===== 9. COMPUTE & SAVE METADATA =====
    logger.info("Step 8: Computing metadata...")
    
    # Per-pincode MAE
    X_val_with_pred = X_val.copy()
    X_val_with_pred['gb_pred'] = gb_pred
    X_val_with_pred['actual'] = y_val.values
    
    mae_by_pincode = {}
    for pincode in X_val_with_pred['pincode'].unique():
        mask = X_val_with_pred['pincode'] == pincode
        if mask.sum() > 0:
            pincode_mae = np.mean(np.abs(X_val_with_pred[mask]['gb_pred'] - X_val_with_pred[mask]['actual']))
            mae_by_pincode[str(pincode)] = float(pincode_mae)
    
    metadata = {
        'lr_r2': float(gb_r2), # Using GB results for metadata slots
        'lr_mae': float(gb_mae),
        'xgb_r2': float(gb_r2),
        'xgb_mae': float(gb_mae),
        'mean_price_per_sqft': float(y_train.mean()),
        'std_price_per_sqft': float(y_train.std()),
        'mae_by_pincode': mae_by_pincode,
        'training_samples': len(X_train),
        'validation_samples': len(X_val),
    }
    
    with open(f'{output_dir}/metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"✅ Saved metadata.json with per-pincode MAE")
    
    logger.info("\n" + "="*60)
    logger.info("✅ TRAINING COMPLETE")
    logger.info("="*60)
    logger.info(f"Models saved: {output_dir}/")
    logger.info(f"  - linear_model.pkl (R²={gb_r2:.3f})")
    logger.info(f"  - xgb_model.pkl (R²={gb_r2:.3f})")
    logger.info(f"  - metadata.json")

if __name__ == '__main__':
    main()
