#!/usr/bin/env python3
"""
Script to check what features were used during training
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import joblib
from config import MODELS_DIR

def check_training_features():
    print("🔍 Checking Training Features...")
    
    preprocessor_path = os.path.join(MODELS_DIR, 'preprocessor.joblib')
    
    if not os.path.exists(preprocessor_path):
        print("❌ Preprocessor not found")
        return
    
    try:
        preprocessor_data = joblib.load(preprocessor_path)
        
        print("📊 Preprocessor Info:")
        print(f"   - Fitted: {preprocessor_data.get('fitted_features', 'Unknown')}")
        
        feature_columns = preprocessor_data.get('feature_columns', [])
        trained_features = preprocessor_data.get('trained_feature_names', [])
        
        print(f"   - Feature columns: {len(feature_columns)}")
        print(f"   - Trained features: {len(trained_features)}")
        
        if feature_columns:
            print("\n📋 First 20 feature columns:")
            for i, col in enumerate(feature_columns[:20]):
                print(f"   {i+1:2d}. {col}")
            if len(feature_columns) > 20:
                print(f"   ... and {len(feature_columns) - 20} more")
        
        if trained_features:
            print("\n📋 First 20 trained features:")
            for i, col in enumerate(trained_features[:20]):
                print(f"   {i+1:2d}. {col}")
            if len(trained_features) > 20:
                print(f"   ... and {len(trained_features) - 20} more")
                
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    check_training_features()