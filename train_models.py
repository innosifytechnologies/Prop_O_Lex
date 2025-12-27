import pandas as pd
import numpy as np
import json
import os
import logging
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
import joblib

from data_loading import load_and_validate
from feature_engineering import engineer_features, CBD_COORDS, SUBTYPE_MULTIPLIERS
from preprocessing import build_preprocessing_pipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main(csv_path='consolidated_properties.csv', output_dir='./models'):
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
    
    # ===== 6. TRAIN LINEAR REGRESSION =====
    logger.info("Step 6a: Training Linear Regression...")
    lr_model = LinearRegression()
    lr_pipeline = Pipeline([('preprocessor', preprocessor), ('model', lr_model)])
    lr_pipeline.fit(X_train, y_train)
    
    lr_r2 = lr_pipeline.score(X_val, y_val)
    lr_pred = lr_pipeline.predict(X_val)
    lr_mae = np.mean(np.abs(lr_pred - y_val))
    logger.info(f"✅ Linear Regression - R²: {lr_r2:.4f}, MAE: ₹{lr_mae:.0f}/sqft")
    
    # ===== 7. TRAIN XGBOOST =====
    logger.info("Step 6b: Training XGBoost...")
    xgb_model = XGBRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
        verbosity=0
    )
    xgb_pipeline = Pipeline([('preprocessor', preprocessor), ('model', xgb_model)])
    xgb_pipeline.fit(X_train, y_train)
    
    xgb_r2 = xgb_pipeline.score(X_val, y_val)
    xgb_pred = xgb_pipeline.predict(X_val)
    xgb_mae = np.mean(np.abs(xgb_pred - y_val))
    logger.info(f"✅ XGBoost - R²: {xgb_r2:.4f}, MAE: ₹{xgb_mae:.0f}/sqft")
    
    # ===== 8. SAVE MODELS =====
    logger.info(f"Step 7: Saving models to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(lr_pipeline, f'{output_dir}/linear_model.pkl')
    joblib.dump(xgb_pipeline, f'{output_dir}/xgb_model.pkl')
    logger.info(f"✅ Saved: linear_model.pkl, xgb_model.pkl")
    
    # ===== 9. COMPUTE & SAVE METADATA =====
    logger.info("Step 8: Computing metadata...")
    
    # Per-pincode MAE
    X_val_with_pred = X_val.copy()
    X_val_with_pred['lr_pred'] = lr_pred
    X_val_with_pred['actual'] = y_val.values
    
    mae_by_pincode = {}
    for pincode in X_val_with_pred['pincode'].unique():
        mask = X_val_with_pred['pincode'] == pincode
        if mask.sum() > 0:
            pincode_mae = np.mean(np.abs(X_val_with_pred[mask]['lr_pred'] - X_val_with_pred[mask]['actual']))
            mae_by_pincode[str(pincode)] = float(pincode_mae)
    
    metadata = {
        'lr_r2': float(lr_r2),
        'lr_mae': float(lr_mae),
        'xgb_r2': float(xgb_r2),
        'xgb_mae': float(xgb_mae),
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
    logger.info(f"  - linear_model.pkl (R²={lr_r2:.3f})")
    logger.info(f"  - xgb_model.pkl (R²={xgb_r2:.3f})")
    logger.info(f"  - metadata.json")

if __name__ == '__main__':
    main()
