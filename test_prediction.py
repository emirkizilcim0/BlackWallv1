#!/usr/bin/env python3
"""
Test script to verify prediction works with ML models
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import joblib
import numpy as np
from utils.preprocessor import DataPreprocessor
from config import MODELS_DIR

def test_prediction():
    print("🧪 Testing Full Prediction System...")
    
    # Load preprocessor
    preprocessor = DataPreprocessor()
    preprocessor_path = os.path.join(MODELS_DIR, 'preprocessor.joblib')
    
    if not os.path.exists(preprocessor_path):
        print("❌ Preprocessor not found. Train models first.")
        return False
    
    try:
        preprocessor.load_preprocessor(preprocessor_path)
        print("✅ Preprocessor loaded")
    except Exception as e:
        print(f"❌ Error loading preprocessor: {e}")
        return False
    
    # Load ML models
    models = {}
    model_files = {
        'logistic_regression': 'logistic_regression.joblib',
        'random_forest': 'random_forest.joblib',
        'svm': 'svm.joblib',
        'isolation_forest': 'isolation_forest.joblib'
    }
    
    for name, filename in model_files.items():
        model_path = os.path.join(MODELS_DIR, filename)
        if os.path.exists(model_path):
            try:
                models[name] = joblib.load(model_path)
                print(f"✅ Loaded {name}")
            except Exception as e:
                print(f"❌ Error loading {name}: {e}")
        else:
            print(f"⚠️ Model not found: {filename}")
    
    if not models:
        print("❌ No models loaded")
        return False
    
    # Get the features that were used during training
    trained_features = getattr(preprocessor, 'feature_columns', [])
    if not trained_features:
        print("❌ No feature information found in preprocessor")
        return False
    
    print(f"🎯 Using {len(trained_features)} training features")
    
    # Test both normal and attack patterns
    test_cases = [
        {
            'name': 'NORMAL Traffic',
            'data': {
                'Fwd Packet Length Mean': 50.0,
                'Fwd Packet Length Std': 15.0,
                'Bwd Packet Length Mean': 40.0,
                'Bwd Packet Length Std': 12.0,
                'Flow Bytes/s': 1000.0,
                'Flow Packets/s': 20.0,
                'Flow IAT Mean': 25.0,
                'Flow IAT Std': 5.0,
                'Fwd IAT Mean': 25.0,
                'Fwd IAT Std': 5.0,
                'Bwd IAT Mean': 20.0,
                'Bwd IAT Std': 4.0,
                'Fwd Packets/s': 10.0,
                'Bwd Packets/s': 8.0,
                'Packet Length Mean': 45.0,
                'Packet Length Std': 12.0,
                'Packet Length Variance': 144.0,
                'Average Packet Size': 45.0,
                'Avg Fwd Segment Size': 50.0,
                'Avg Bwd Segment Size': 40.0,
                'Active Mean': 25.0,
                'Active Std': 5.0,
                'Idle Mean': 100.0,
                'Idle Std': 20.0
            }
        },
        {
            'name': 'ATTACK Traffic', 
            'data': {
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
                'Idle Std': 0.0
            }
        }
    ]
    
    for test_case in test_cases:
        print(f"\n{'='*50}")
        print(f"Testing: {test_case['name']}")
        print(f"{'='*50}")
        
        # Convert to DataFrame
        test_df = pd.DataFrame([test_case['data']])
        
        try:
            # Prepare features
            features = preprocessor.prepare_real_time_features(test_df)
            print(f"✅ Features prepared: {features.shape}")
            
            # Make predictions with all models
            predictions = {}
            for name, model in models.items():
                try:
                    if hasattr(model, 'predict_proba'):
                        # Classification models
                        proba = model.predict_proba(features)[0]
                        prediction = int(model.predict(features)[0])
                        confidence = float(max(proba))
                        label = 'ATTACK' if prediction == 1 else 'NORMAL'
                        
                        print(f"🎯 {name:.<20} {label:.<10} (confidence: {confidence:.3f})")
                        
                    else:
                        # Anomaly detection models (Isolation Forest)
                        raw_prediction = model.predict(features)[0]
                        is_anomaly = (raw_prediction == -1)
                        prediction = 1 if is_anomaly else 0
                        confidence = 0.9 if is_anomaly else 0.1
                        label = 'ATTACK' if is_anomaly else 'NORMAL'
                        
                        print(f"🎯 {name:.<20} {label:.<10} (anomaly: {is_anomaly})")
                    
                    predictions[name] = {
                        'prediction': prediction,
                        'confidence': confidence,
                        'label': label
                    }
                    
                except Exception as model_error:
                    print(f"❌ {name:.<20} ERROR: {model_error}")
                    predictions[name] = {
                        'prediction': 0,
                        'confidence': 0.0,
                        'label': 'ERROR'
                    }
            
            # Summary
            attack_votes = sum(1 for p in predictions.values() if p['label'] == 'ATTACK')
            total_models = len(predictions)
            print(f"\n📊 Summary: {attack_votes}/{total_models} models detected ATTACK")
            
            if attack_votes > total_models / 2:
                print("🚨 MAJORITY VOTE: ATTACK DETECTED!")
            else:
                print("✅ MAJORITY VOTE: NORMAL TRAFFIC")
                
        except Exception as e:
            print(f"❌ Prediction failed: {e}")
            import traceback
            traceback.print_exc()
    
    return True

if __name__ == "__main__":
    test_prediction()