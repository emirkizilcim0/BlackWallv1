# balanced_trainer_fixed.py
import pandas as pd
import numpy as np
from models.trainer import BlackWallTrainer
from utils.preprocessor import DataPreprocessor
from config import CIC_DATA_DIR, MODELS_DIR
import os
import joblib

def load_and_fix_data(sample_fraction=0.3):
    """Load data and fix label issues"""
    preprocessor = DataPreprocessor()
    df = preprocessor.load_cic_2017_data(sample_fraction=sample_fraction)
    
    if df is None or len(df) == 0:
        print("❌ No data loaded")
        return None, None
    
    print(f"📊 Original dataset: {len(df)} rows")
    
    # Fix label mapping
    df, label_column = fix_label_mapping(df)
    
    # Check if we have any attacks
    attack_count = df[df[label_column] == 1].shape[0]
    normal_count = df[df[label_column] == 0].shape[0]
    
    print(f"   - Attack samples: {attack_count}")
    print(f"   - Normal samples: {normal_count}")
    print(f"   - Attack ratio: {attack_count/len(df):.3f}")
    
    if attack_count == 0:
        print("❌ NO ATTACK SAMPLES FOUND! Cannot train detection models.")
        print("💡 Try loading more data or check your dataset")
        return None, None
    
    return df, label_column

def fix_label_mapping(df):
    """Fix CIC-IDS-2017 label mapping to binary"""
    print("🔄 Mapping labels to binary...")
    
    # CIC-IDS-2017 specific attack labels
    attack_labels = [
        'DDoS', 'PortScan', 'Bot', 'Brute Force', 'Web Attack', 
        'Infiltration', 'Heartbleed', 'FTP-Patator', 'SSH-Patator',
        'DoS', 'DDOS', 'Portscan', 'BOT', 'Botnet'
    ]
    
    # Create binary label column
    df['Label_Binary'] = 0  # Default to normal
    
    if 'Label' in df.columns:
        if df['Label'].dtype == 'object':
            # String labels - look for attack patterns
            for attack_label in attack_labels:
                mask = df['Label'].str.contains(attack_label, case=False, na=False)
                attack_count = mask.sum()
                if attack_count > 0:
                    print(f"   - Mapped {attack_count} '{attack_label}' as attacks")
                    df.loc[mask, 'Label_Binary'] = 1
        else:
            # Numeric labels - assume 1 is attack, 0 is normal
            df['Label_Binary'] = df['Label'].apply(lambda x: 1 if x == 1 else 0)
    
    # If still no attacks, check for other common columns
    if df['Label_Binary'].sum() == 0:
        print("   ⚠️ No attacks found in 'Label' column, checking alternatives...")
        
        # Check for other possible label columns
        possible_label_cols = ['Attack', 'Malicious', 'Class', 'Result']
        for col in possible_label_cols:
            if col in df.columns:
                print(f"   - Found alternative label column: {col}")
                unique_vals = df[col].unique()
                print(f"     Values: {unique_vals}")
    
    return df, 'Label_Binary'

def train_balanced_models_fixed(sample_fraction=0.3, balance_ratio=0.4, strategy='oversample'):
    """Fixed version that handles label issues"""
    print("🎯 Training Models with BALANCED Data (Fixed)...")
    print("=" * 60)
    
    # Load and fix data
    df, label_column = load_and_fix_data(sample_fraction)
    
    if df is None:
        return
    
    preprocessor = DataPreprocessor()
    trainer = BlackWallTrainer()
    
    # Separate classes using the correct label column
    attack_samples = df[df[label_column] == 1]
    normal_samples = df[df[label_column] == 0]
    
    print(f"\n📊 Before balancing:")
    print(f"   - Attack samples: {len(attack_samples)}")
    print(f"   - Normal samples: {len(normal_samples)}")
    
    if len(attack_samples) == 0:
        print("❌ Still no attack samples after mapping!")
        print("💡 Your dataset might not contain attack traffic")
        return
    
    # Apply balancing
    if strategy == 'oversample':
        # Oversample attacks
        target_attack_count = int(len(normal_samples) * balance_ratio / (1 - balance_ratio))
        
        if len(attack_samples) < target_attack_count:
            # Need to oversample attacks
            oversampled_attacks = attack_samples.sample(
                n=target_attack_count, replace=True, random_state=42
            )
            balanced_df = pd.concat([normal_samples, oversampled_attacks])
            print(f"   - Oversampled attacks: {len(attack_samples)} → {target_attack_count}")
        else:
            # Have enough attacks, just take what we need
            attack_balanced = attack_samples.sample(n=target_attack_count, random_state=42)
            balanced_df = pd.concat([normal_samples, attack_balanced])
            print(f"   - Using {target_attack_count} attack samples")
            
    elif strategy == 'undersample':
        # Undersample normals
        target_normal_count = int(len(attack_samples) * (1 - balance_ratio) / balance_ratio)
        normal_balanced = normal_samples.sample(
            n=min(target_normal_count, len(normal_samples)), random_state=42
        )
        balanced_df = pd.concat([attack_samples, normal_balanced])
        print(f"   - Undersampled normals: {len(normal_samples)} → {len(normal_balanced)}")
    
    print(f"📊 After balancing: {len(balanced_df)} rows")
    balanced_attacks = balanced_df[balanced_df[label_column] == 1].shape[0]
    balanced_normals = balanced_df[balanced_df[label_column] == 0].shape[0]
    print(f"   - Attacks: {balanced_attacks}")
    print(f"   - Normals: {balanced_normals}")
    print(f"   - Attack ratio: {balanced_attacks/len(balanced_df):.3f}")
    
    # Use the binary label column for training
    balanced_df['Label'] = balanced_df[label_column]
    
    # Prepare features and train
    X, y_binary, y_multiclass = preprocessor.prepare_features(balanced_df, use_pca=True)
    
    print(f"\n🎯 Training on {len(X)} samples")
    results = trainer.train_models(X, y_binary, y_multiclass)
    
    # Display results
    print("\n📊 Model Performance:")
    print("=" * 50)
    
    for model_name, result in results.items():
        print(f"{model_name.upper():<20} | Acc: {result['accuracy']:.3f} | F1: {result['f1_score']:.3f}")
    
    # Save models
    print("\n💾 Saving models...")
    trainer.save_models()
    
    # Quick test
    test_quick_detection()
    
    return results

def test_quick_detection():
    """Quick test to see if models can detect attacks now"""
    print("\n🧪 Quick Detection Test...")
    
    from utils.real_time_predictor import RealTimePredictor
    predictor = RealTimePredictor()
    
    if not predictor.models:
        print("❌ No models to test")
        return
    
    # Test with obvious attack pattern
    test_data = {
        'Fwd Packet Length Mean': 95.0, 'Fwd Packet Length Std': 120.0,
        'Bwd Packet Length Mean': 0.0, 'Bwd Packet Length Std': 0.0,
        'Flow Bytes/s': 450000.0, 'Flow Packets/s': 3200.0,
        'Flow IAT Mean': 0.2, 'Flow IAT Std': 0.4,
        'Fwd IAT Mean': 0.2, 'Fwd IAT Std': 0.4,
        'Fwd Packets/s': 3200.0, 'Bwd Packets/s': 0.0,
        'Packet Length Mean': 95.0, 'Packet Length Std': 120.0,
    }
    
    df = pd.DataFrame([test_data])
    X_processed = predictor.preprocessor.prepare_real_time_features(df)
    
    if X_processed is None:
        print("❌ Could not process test features")
        return
    
    attack_count = 0
    for name, model in predictor.models.items():
        try:
            if hasattr(model, 'predict_proba'):
                pred = model.predict(X_processed)[0]
                if pred == 1:
                    attack_count += 1
                    print(f"✅ {name}: DETECTED")
                else:
                    print(f"❌ {name}: missed")
        except Exception as e:
            print(f"💥 {name}: error")
    
    print(f"📊 Result: {attack_count}/{len(predictor.models)} models detected attack")

if __name__ == "__main__":
    # First run diagnostic
    print("🔍 Running data diagnostic...")
    from data_diagnostic import diagnose_data_issues
    diagnose_data_issues()
    
    print("\n" + "="*60)
    
    # Then train with balanced data
    train_balanced_models_fixed(
        sample_fraction=0.3,
        balance_ratio=0.4,  # 40% attacks
        strategy='oversample'
    )