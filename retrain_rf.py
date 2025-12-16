#!/usr/bin/env python3
"""
Retrain Random Forest with better parameters for attack detection
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.preprocessor import DataPreprocessor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
from config import MODELS_DIR
import numpy as np

def retrain_random_forest():
    print("🔄 Retraining Random Forest with Better Parameters")
    print("=" * 50)
    
    preprocessor = DataPreprocessor()
    
    try:
        # Load data
        print("📁 Loading training data...")
        df = preprocessor.load_cic_2017_data(sample_fraction=0.1)  # Use 10% for quick retraining
        
        if df is None or 'is_attack' not in df.columns:
            print("❌ Could not load proper training data")
            return False
        
        # Prepare features
        print("🔧 Preparing features...")
        X_processed, y_binary, _ = preprocessor.prepare_features(df, use_pca=False)
        
        # Check class balance
        attack_ratio = y_binary.sum() / len(y_binary)
        print(f"📊 Attack ratio in training data: {attack_ratio:.3f}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y_binary, test_size=0.2, random_state=42, stratify=y_binary
        )
        
        print(f"📐 Training set: {X_train.shape[0]} samples")
        print(f"📐 Test set: {X_test.shape[0]} samples")
        
        # Train new Random Forest with better parameters for attack detection
        print("🌲 Training Random Forest...")
        
        rf_model = RandomForestClassifier(
            n_estimators=200,           # More trees for better performance
            max_depth=20,               # Deeper trees to capture complex patterns
            min_samples_split=5,        # Less splitting for finer details
            min_samples_leaf=2,         # Smaller leaves for attack patterns
            class_weight='balanced',    # Handle imbalanced data
            random_state=42,
            n_jobs=-1                  # Use all cores
        )
        
        rf_model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = rf_model.predict(X_test)
        y_pred_proba = rf_model.predict_proba(X_test)
        
        print("\n📊 Model Performance:")
        print(classification_report(y_test, y_pred, target_names=['NORMAL', 'ATTACK']))
        
        # Test on our problematic sample
        print("\n🔍 Testing on PORT_SCAN pattern:")
        test_sample = np.array([[1000.0, 500.0, 0.0, 0.0, 1000000.0, 10000.0, 0.1, 0.01, 
                               0.01, 0.001, 0.0, 0.0, 10000.0, 0.0, 1000.0, 500.0, 
                               250000.0, 1000.0, 1000.0, 0.0, 0.1, 0.01, 0.0, 0.0]])
        
        # Ensure test sample has right shape (might need to adjust based on feature selection)
        if test_sample.shape[1] > X_processed.shape[1]:
            test_sample = test_sample[:, :X_processed.shape[1]]
        elif test_sample.shape[1] < X_processed.shape[1]:
            test_sample = np.pad(test_sample, ((0, 0), (0, X_processed.shape[1] - test_sample.shape[1])))
        
        test_pred = rf_model.predict(test_sample)[0]
        test_proba = rf_model.predict_proba(test_sample)[0]
        
        print(f"   Prediction: {'ATTACK' if test_pred == 1 else 'NORMAL'}")
        print(f"   Probabilities: [NORMAL: {test_proba[0]:.3f}, ATTACK: {test_proba[1]:.3f}]")
        
        if test_pred == 1:
            print("   ✅ New model correctly detects PORT_SCAN as attack!")
        else:
            print("   ❌ New model still misses PORT_SCAN")
        
        # Save the improved model
        rf_path = os.path.join(MODELS_DIR, 'random_forest.joblib')
        joblib.dump(rf_model, rf_path)
        print(f"\n💾 Saved improved Random Forest to {rf_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error retraining: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    if retrain_random_forest():
        print("\n🎉 Random Forest retrained successfully!")
        print("🔄 Now test with: python test_detection_fixed.py")
    else:
        print("\n❌ Retraining failed")