# balanced_trainer.py
import pandas as pd
import numpy as np
from models.trainer import BlackWallTrainer
from utils.preprocessor import DataPreprocessor
from config import CIC_DATA_DIR, MODELS_DIR
import os
import joblib

def train_balanced_models(sample_fraction=0.3, balance_ratio=0.5, strategy='oversample'):
    """
    Train models with balanced class distribution
    
    Args:
        sample_fraction: Fraction of total data to use
        balance_ratio: Target ratio of attack samples (0.5 = 50% attacks)
        strategy: 'oversample', 'undersample', or 'smote'
    """
    print("🎯 Training Models with BALANCED Data...")
    print("=" * 60)
    
    preprocessor = DataPreprocessor()
    trainer = BlackWallTrainer()
    
    # Load data
    df = preprocessor.load_cic_2017_data(sample_fraction=sample_fraction)
    
    if df is None or len(df) == 0:
        print("❌ No data loaded")
        return
    
    print(f"📊 Original dataset: {len(df)} rows")
    print("Class distribution BEFORE balancing:")
    class_dist = df['Label'].value_counts()
    print(class_dist)
    
    # Separate classes
    attack_samples = df[df['Label'] == 1]
    normal_samples = df[df['Label'] == 0]
    
    print(f"   - Attack samples: {len(attack_samples)}")
    print(f"   - Normal samples: {len(normal_samples)}")
    print(f"   - Attack ratio: {len(attack_samples)/len(df):.3f}")
    
    # Apply balancing strategy
    if strategy == 'undersample':
        # Undersample normal traffic
        target_attack_count = len(attack_samples)
        target_normal_count = int(target_attack_count * (1 - balance_ratio) / balance_ratio)
        
        normal_balanced = normal_samples.sample(n=min(target_normal_count, len(normal_samples)), 
                                              random_state=42)
        balanced_df = pd.concat([attack_samples, normal_balanced])
        
    elif strategy == 'oversample':
        # Oversample attack traffic
        target_normal_count = len(normal_samples)
        target_attack_count = int(target_normal_count * balance_ratio / (1 - balance_ratio))
        
        # Oversample attacks if needed
        if len(attack_samples) < target_attack_count:
            oversampled_attacks = attack_samples.sample(n=target_attack_count, 
                                                      replace=True, random_state=42)
            balanced_df = pd.concat([normal_samples, oversampled_attacks])
        else:
            # Undersample attacks if too many
            attack_balanced = attack_samples.sample(n=target_attack_count, random_state=42)
            balanced_df = pd.concat([normal_samples, attack_balanced])
            
    elif strategy == 'smote':
        # You'll need to install: pip install imbalanced-learn
        try:
            from imblearn.over_sampling import SMOTE
            X, y_binary, y_multiclass = preprocessor.prepare_features(df, use_pca=False)
            smote = SMOTE(random_state=42)
            X_balanced, y_balanced = smote.fit_resample(X, y_binary)
            
            # Convert back to DataFrame format
            balanced_features = pd.DataFrame(X_balanced, columns=X.columns)
            balanced_df = balanced_features.copy()
            balanced_df['Label'] = y_balanced
            
        except ImportError:
            print("❌ SMOTE not available. Using oversampling instead.")
            return train_balanced_models(sample_fraction, balance_ratio, 'oversample')
    
    else:
        print("❌ Unknown strategy. Using undersampling.")
        return train_balanced_models(sample_fraction, balance_ratio, 'undersample')
    
    print(f"📊 Balanced dataset: {len(balanced_df)} rows")
    print("Class distribution AFTER balancing:")
    balanced_dist = balanced_df['Label'].value_counts()
    print(balanced_dist)
    print(f"   - New attack ratio: {balanced_dist[1]/len(balanced_df):.3f}")
    
    # Prepare features
    X, y_binary, y_multiclass = preprocessor.prepare_features(balanced_df, use_pca=True)
    
    print(f"🎯 Training on {len(X)} balanced samples")
    print(f"   - Attacks: {len(attack_samples)} → {balanced_dist[1]}")
    print(f"   - Normal: {len(normal_samples)} → {balanced_dist[0]}")
    
    # Train models with balanced class weights
    results = trainer.train_models(X, y_binary, y_multiclass)
    
    # Display results
    print("\n📊 Balanced Model Performance:")
    print("=" * 50)
    
    for model_name, result in results.items():
        print(f"\n{model_name.upper():<20}")
        print(f"  Accuracy:  {result['accuracy']:.4f}")
        print(f"  F1-Score:  {result['f1_score']:.4f}")
        if 'roc_auc' in result:
            print(f"  ROC-AUC:   {result['roc_auc']:.4f}")
    
    # Save models
    print("\n💾 Saving balanced models...")
    trainer.save_models()
    
    # Test on some attack patterns
    test_balanced_models()
    
    return results

def test_balanced_models():
    """Test the newly trained balanced models"""
    print("\n🧪 Testing Balanced Models...")
    print("=" * 50)
    
    from utils.real_time_predictor import RealTimePredictor
    predictor = RealTimePredictor()
    
    if not predictor.models:
        print("❌ No models loaded for testing")
        return
    
    # Test samples
    test_attacks = [
        {
            'name': 'DDoS_Test',
            'data': {
                'Fwd Packet Length Mean': 95.0, 'Fwd Packet Length Std': 120.0,
                'Bwd Packet Length Mean': 0.0, 'Bwd Packet Length Std': 0.0,
                'Flow Bytes/s': 450000.0, 'Flow Packets/s': 3200.0,
                'Flow IAT Mean': 0.2, 'Flow IAT Std': 0.4,
                'Fwd IAT Mean': 0.2, 'Fwd IAT Std': 0.4,
                'Fwd Packets/s': 3200.0, 'Bwd Packets/s': 0.0,
                'Packet Length Mean': 95.0, 'Packet Length Std': 120.0,
            }
        },
        {
            'name': 'PortScan_Test', 
            'data': {
                'Fwd Packet Length Mean': 78.0, 'Fwd Packet Length Std': 45.0,
                'Bwd Packet Length Mean': 52.0, 'Bwd Packet Length Std': 38.0,
                'Flow Bytes/s': 85000.0, 'Flow Packets/s': 850.0,
                'Flow IAT Mean': 0.8, 'Flow IAT Std': 1.2,
                'Fwd Packets/s': 480.0, 'Bwd Packets/s': 370.0,
            }
        }
    ]
    
    for test in test_attacks:
        print(f"\n🔍 Testing: {test['name']}")
        
        df = pd.DataFrame([test['data']])
        X_processed = predictor.preprocessor.prepare_real_time_features(df)
        
        if X_processed is None or len(X_processed) == 0:
            print("   ❌ No features processed")
            continue
        
        # Test each model
        attack_detections = 0
        total_models = len(predictor.models)
        
        for model_name, model in predictor.models.items():
            try:
                if hasattr(model, 'predict_proba'):
                    prediction = model.predict(X_processed)[0]
                    confidence = np.max(model.predict_proba(X_processed)[0])
                    label = 'ATTACK' if prediction == 1 else 'NORMAL'
                else:
                    raw_pred = model.predict(X_processed)[0]
                    is_anomaly = (raw_pred == -1)
                    label = 'ATTACK' if is_anomaly else 'NORMAL'
                    confidence = 0.9 if is_anomaly else 0.1
                
                if label == 'ATTACK':
                    attack_detections += 1
                    print(f"   ✅ {model_name}: THREAT (conf: {confidence:.2f})")
                else:
                    print(f"   ❌ {model_name}: NORMAL (conf: {confidence:.2f})")
                    
            except Exception as e:
                print(f"   💥 {model_name}: ERROR - {e}")
        
        print(f"   📊 Result: {attack_detections}/{total_models} models detected threat")
        
        if attack_detections >= total_models * 0.5:
            print("   🎯 GOOD: Majority detection")
        else:
            print("   ⚠️  POOR: Weak detection")

def compare_balanced_vs_normal():
    """Compare balanced vs normal training"""
    print("🔬 COMPARISON: Balanced vs Normal Training")
    print("=" * 60)
    
    # Test current models (trained on imbalanced data)
    print("\n📊 CURRENT MODELS (Imbalanced Training):")
    test_balanced_models()
    
    # Retrain with balanced data
    print("\n🔄 Retraining with balanced data...")
    balanced_results = train_balanced_models(sample_fraction=0.2, balance_ratio=0.4)
    
    print("\n📊 BALANCED MODELS Performance:")
    test_balanced_models()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Balanced Model Training')
    parser.add_argument('--strategy', type=str, default='oversample',
                       choices=['oversample', 'undersample', 'smote'],
                       help='Balancing strategy')
    parser.add_argument('--ratio', type=float, default=0.4,
                       help='Target attack ratio (0.4 = 40% attacks)')
    parser.add_argument('--sample', type=float, default=0.3,
                       help='Sample fraction of total data')
    parser.add_argument('--compare', action='store_true',
                       help='Compare balanced vs normal training')
    
    args = parser.parse_args()
    
    if args.compare:
        compare_balanced_vs_normal()
    else:
        train_balanced_models(
            sample_fraction=args.sample,
            balance_ratio=args.ratio,
            strategy=args.strategy
        )