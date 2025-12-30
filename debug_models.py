import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.real_time_predictor import RealTimePredictor
import pandas as pd
import numpy as np
import joblib
from config import MODELS_DIR

def debug_models():
    print("🔍 Debugging Model Performance")
    print("=" * 50)
    
    # Load models directly
    rf_path = os.path.join(MODELS_DIR, 'random_forest.joblib')
    iso_path = os.path.join(MODELS_DIR, 'isolation_forest.joblib')
    
    rf_model = joblib.load(rf_path)
    iso_model = joblib.load(iso_path)
    
    print("📊 Random Forest Info:")
    print(f"   - Classes: {rf_model.classes_}")
    if hasattr(rf_model, 'feature_importances_'):
        print(f"   - Feature importance sum: {rf_model.feature_importances_.sum():.3f}")
    
    print("📊 Isolation Forest Info:")
    print(f"   - Contamination: {getattr(iso_model, 'contamination', 'Not set')}")
    print(f"   - Expected outliers: {getattr(iso_model, 'contamination', 'Unknown')}")
    
    # Test with the problematic sample
    test_sample = {
        'Fwd Packet Length Mean': 1000.0,
        'Fwd Packet Length Std': 500.0,
        'Bwd Packet Length Mean': 0.0,
        'Bwd Packet Length Std': 0.0,
        'Flow Bytes/s': 1000000.0,
        'Flow Packets/s': 10000.0,
        'Flow IAT Mean': 0.1,
        'Flow IAT Std': 0.01,
        'Fwd IAT Mean': 0.01,
        'Fwd IAT Std': 0.001,
        'Bwd IAT Mean': 0.0,
        'Bwd IAT Std': 0.0,
        'Fwd Packets/s': 10000.0,
        'Bwd Packets/s': 0.0,
        'Packet Length Mean': 1000.0,
        'Packet Length Std': 500.0,
        'Packet Length Variance': 250000.0,
        'Average Packet Size': 1000.0,
        'Avg Fwd Segment Size': 1000.0,
        'Avg Bwd Segment Size': 0.0,
        'Active Mean': 0.1,
        'Active Std': 0.01,
        'Idle Mean': 0.0,
        'Idle Std': 0.0,
    }
    
    df = pd.DataFrame([test_sample])
    
    # Load preprocessor to process features
    predictor = RealTimePredictor()
    X_processed = predictor.preprocessor.prepare_real_time_features(df)
    
    print(f"\n🔍 Testing PORT_SCAN_SSH sample:")
    print(f"   Processed features shape: {X_processed.shape}")
    
    # Test Random Forest
    rf_pred = rf_model.predict(X_processed)[0]
    rf_proba = rf_model.predict_proba(X_processed)[0]
    print(f"   Random Forest:")
    print(f"      Prediction: {rf_pred} (Class: {'ATTACK' if rf_pred == 1 else 'NORMAL'})")
    print(f"      Probabilities: [NORMAL: {rf_proba[0]:.3f}, ATTACK: {rf_proba[1]:.3f}]")
    
    # Test Isolation Forest
    iso_pred = iso_model.predict(X_processed)[0]
    iso_score = iso_model.score_samples(X_processed)[0]
    print(f"   Isolation Forest:")
    print(f"      Prediction: {iso_pred} ({'ANOMALY' if iso_pred == -1 else 'NORMAL'})")
    print(f"      Score: {iso_score:.3f}")
    print(f"      Decision: {'ATTACK' if iso_pred == -1 else 'NORMAL'}")

if __name__ == "__main__":
    debug_models()