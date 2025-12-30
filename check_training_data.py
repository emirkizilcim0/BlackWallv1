import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.preprocessor import DataPreprocessor
import pandas as pd
from config import CIC_DATA_DIR

def check_training_data():
    print("🔍 Checking Training Data Balance")
    print("=" * 50)
    
    preprocessor = DataPreprocessor()
    
    try:
        # Load a sample of training data
        print("📁 Loading training data sample...")
        df = preprocessor.load_cic_2017_data(sample_fraction=0.01)  # Small sample for quick check
        
        if df is None:
            print("❌ Could not load training data")
            return
        
        # Check label distribution
        if 'is_attack' in df.columns:
            attack_count = df['is_attack'].sum()
            normal_count = len(df) - attack_count
            total = len(df)
            
            print(f"📊 Dataset Balance:")
            print(f"   Normal traffic: {normal_count} ({normal_count/total*100:.1f}%)")
            print(f"   Attack traffic: {attack_count} ({attack_count/total*100:.1f}%)")
            print(f"   Total samples: {total}")
            
            if attack_count / total < 0.1:
                print("⚠️  WARNING: Very low attack samples - model may not learn attack patterns!")
            elif attack_count / total > 0.9:
                print("⚠️  WARNING: Very high attack samples - dataset may be imbalanced!")
            else:
                print("✅ Good balance between normal and attack traffic")
        
        # Check specific attack types
        if 'Label' in df.columns:
            print(f"\n📋 Attack Type Distribution:")
            attack_types = df[df['is_attack'] == 1]['Label'].value_counts()
            for attack_type, count in attack_types.head(10).items():  # Show top 10
                print(f"   {attack_type}: {count} samples")
                
    except Exception as e:
        print(f"❌ Error checking training data: {e}")

def test_model_on_real_attacks():
    """Test the model with known attack patterns from CIC-IDS-2017"""
    print(f"\n🎯 Testing Model on Real Attack Patterns")
    print("=" * 50)
    
    from utils.real_time_predictor import RealTimePredictor
    import joblib
    from config import MODELS_DIR
    
    predictor = RealTimePredictor()
    
    # Common CIC-IDS-2017 attack patterns (simplified)
    real_attack_patterns = [
        {
            'name': 'DDoS_LOIC',
            'data': {
                'Fwd Packet Length Mean': 100.0,
                'Fwd Packet Length Std': 20.0,
                'Bwd Packet Length Mean': 0.0,
                'Bwd Packet Length Std': 0.0,
                'Flow Bytes/s': 5000000.0,
                'Flow Packets/s': 5000.0,
                'Flow IAT Mean': 0.0002,
                'Flow IAT Std': 0.0001,
                'Fwd IAT Mean': 0.0002,
                'Fwd IAT Std': 0.0001,
                'Bwd IAT Mean': 0.0,
                'Bwd IAT Std': 0.0,
                'Fwd Packets/s': 5000.0,
                'Bwd Packets/s': 0.0,
                'Packet Length Mean': 100.0,
                'Packet Length Std': 20.0,
                'Packet Length Variance': 400.0,
                'Average Packet Size': 100.0,
                'Avg Fwd Segment Size': 100.0,
                'Avg Bwd Segment Size': 0.0,
                'Active Mean': 0.0002,
                'Active Std': 0.0001,
                'Idle Mean': 0.0,
                'Idle Std': 0.0,
            },
            'description': 'LOIC DDoS attack - high volume UDP/TCP'
        },
        {
            'name': 'PortScan', 
            'data': {
                'Fwd Packet Length Mean': 60.0,
                'Fwd Packet Length Std': 5.0,
                'Bwd Packet Length Mean': 0.0,
                'Bwd Packet Length Std': 0.0,
                'Flow Bytes/s': 100000.0,
                'Flow Packets/s': 2000.0,
                'Flow IAT Mean': 0.0005,
                'Flow IAT Std': 0.0002,
                'Fwd IAT Mean': 0.0005,
                'Fwd IAT Std': 0.0002,
                'Bwd IAT Mean': 0.0,
                'Bwd IAT Std': 0.0,
                'Fwd Packets/s': 2000.0,
                'Bwd Packets/s': 0.0,
                'Packet Length Mean': 60.0,
                'Packet Length Std': 5.0,
                'Packet Length Variance': 25.0,
                'Average Packet Size': 60.0,
                'Avg Fwd Segment Size': 60.0,
                'Avg Bwd Segment Size': 0.0,
                'Active Mean': 0.0005,
                'Active Std': 0.0002,
                'Idle Mean': 0.0,
                'Idle Std': 0.0,
            },
            'description': 'Port scanning - many SYN packets'
        }
    ]
    
    for pattern in real_attack_patterns:
        print(f"\n🔍 Testing: {pattern['name']}")
        print(f"   Description: {pattern['description']}")
        
        df = pd.DataFrame([pattern['data']])
        
        try:
            X_processed = predictor.preprocessor.prepare_real_time_features(df)
            
            # Test Random Forest
            rf_model = predictor.models['random_forest']
            rf_pred = rf_model.predict(X_processed)[0]
            rf_proba = rf_model.predict_proba(X_processed)[0]
            
            # Test Isolation Forest  
            iso_model = predictor.models['isolation_forest']
            iso_pred = iso_model.predict(X_processed)[0]
            iso_score = iso_model.score_samples(X_processed)[0]
            
            print(f"   Random Forest: {'ATTACK' if rf_pred == 1 else 'NORMAL'} "
                  f"(conf: {max(rf_proba):.3f})")
            print(f"   Isolation Forest: {'ATTACK' if iso_pred == -1 else 'NORMAL'} "
                  f"(score: {iso_score:.3f})")
                  
            if rf_pred == 1:
                print("   ✅ Random Forest correctly detected attack")
            else:
                print("   ❌ Random Forest missed this attack")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")

if __name__ == "__main__":
    check_training_data()
    test_model_on_real_attacks()