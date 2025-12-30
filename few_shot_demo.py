import pandas as pd
import joblib
import numpy as np
import os
from config import MODELS_DIR

# Paths
MODEL_PATH = os.path.join(MODELS_DIR, "few_shot_attack_classifier_10shot.joblib")
LABEL_ENCODER_PATH = os.path.join(MODELS_DIR, "few_shot_label_encoder_10shot.joblib")
PREPROCESSOR_PATH = os.path.join(MODELS_DIR, "few_shot_preprocessor.joblib")

def load_and_inspect_preprocessor():
    """Load and inspect preprocessor contents"""
    try:
        preprocessor_data = joblib.load(PREPROCESSOR_PATH)
        print("🔍 Inspecting preprocessor...")
        print("Keys in preprocessor:", list(preprocessor_data.keys()))
        
        for key, value in preprocessor_data.items():
            if key == 'config':
                print(f"\nConfig: {value}")
            elif key in ['scaler', 'feature_selector', 'pca']:
                if value is not None:
                    print(f"\n{key}: Available")
                    if hasattr(value, 'n_features_in_'):
                        print(f"  n_features_in_: {value.n_features_in_}")
                else:
                    print(f"\n{key}: None")
            elif key in ['original_feature_names', 'selected_feature_names']:
                if isinstance(value, list):
                    print(f"\n{key}: List of {len(value)} items")
                else:
                    print(f"\n{key}: {type(value)}")
            else:
                print(f"\n{key}: {type(value)}")
        
        return preprocessor_data
    except Exception as e:
        print(f"Error inspecting preprocessor: {e}")
        return None

def extract_feature_names(preprocessor_data, use_original=True):
    """
    Extract feature names from preprocessor data
    use_original=True: Use original 70 features (for scaler)
    use_original=False: Use selected 50 features (for feature selector)
    """
    if use_original:
        # For scaler, we need ALL original features
        if 'original_feature_names' in preprocessor_data and preprocessor_data['original_feature_names']:
            feature_names = preprocessor_data['original_feature_names']
            print(f"✅ Using original_feature_names: {len(feature_names)} features (for scaler)")
            return feature_names
    else:
        # For model input, we use selected features
        if 'selected_feature_names' in preprocessor_data and preprocessor_data['selected_feature_names']:
            feature_names = preprocessor_data['selected_feature_names']
            print(f"✅ Using selected_feature_names: {len(feature_names)} features (for model)")
            return feature_names
    
    # Fallback
    print("❌ No feature names found in preprocessor")
    return []

def load_models():
    """Load few-shot models and preprocessor"""
    print("📦 Loading few-shot models...")
    
    try:
        model = joblib.load(MODEL_PATH)
        print(f"✅ Loaded ensemble model: {MODEL_PATH}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None, None, None
    
    try:
        label_encoder = joblib.load(LABEL_ENCODER_PATH)
        print(f"✅ Loaded label encoder: {LABEL_ENCODER_PATH}")
        print(f"   Classes: {list(label_encoder.classes_)}")
    except Exception as e:
        print(f"❌ Error loading label encoder: {e}")
        return None, None, None
    
    try:
        preprocessor_data = joblib.load(PREPROCESSOR_PATH)
        print(f"✅ Loaded preprocessor: {PREPROCESSOR_PATH}")
        
        # Store both sets of feature names
        preprocessor_data['original_features_for_scaler'] = extract_feature_names(preprocessor_data, use_original=True)
        preprocessor_data['selected_features_for_model'] = extract_feature_names(preprocessor_data, use_original=False)
        
        if not preprocessor_data['original_features_for_scaler']:
            print("❌ No original feature names could be extracted")
            return None, None, None
        
        print(f"   Original features: {len(preprocessor_data['original_features_for_scaler'])}")
        print(f"   Selected features: {len(preprocessor_data['selected_features_for_model'])}")
        
    except Exception as e:
        print(f"❌ Error loading preprocessor: {e}")
        return None, None, None
    
    return model, label_encoder, preprocessor_data

def prepare_features_for_prediction(sample_dict, preprocessor_data):
    """Prepare features for prediction - FIXED VERSION"""
    # Get feature names for scaler (ALL original features)
    original_feature_names = preprocessor_data.get('original_features_for_scaler', [])
    selected_feature_names = preprocessor_data.get('selected_features_for_model', [])
    
    if not original_feature_names:
        print("❌ No original feature names available for scaler")
        return None
    
    # Get preprocessing components
    scaler = preprocessor_data.get('scaler')
    feature_selector = preprocessor_data.get('feature_selector')
    pca = preprocessor_data.get('pca')
    
    print(f"📋 Need ALL {len(original_feature_names)} original features for scaler")
    
    # Create aligned DataFrame with ALL original features
    X_aligned = pd.DataFrame(columns=original_feature_names)
    
    # Fill with zeros first
    for col in original_feature_names:
        X_aligned[col] = [0.0]
    
    # Override with sample values where available
    features_found = 0
    features_with_values = []
    
    for key, value in sample_dict.items():
        if key == 'Label':
            continue  # Skip label column
            
        # Try exact match first (case-sensitive)
        if key in original_feature_names:
            try:
                X_aligned[key] = [float(value)]
                features_found += 1
                features_with_values.append(key)
            except:
                print(f"⚠️ Could not convert {key} to float, using 0.0")
        
        # Try case-insensitive match if exact match fails
        else:
            # Find matching column (case-insensitive, remove spaces/dashes)
            key_clean = key.lower().replace(' ', '').replace('-', '').replace('_', '')
            matched = False
            for col in original_feature_names:
                col_clean = col.lower().replace(' ', '').replace('-', '').replace('_', '')
                if key_clean == col_clean:
                    try:
                        X_aligned[col] = [float(value)]
                        features_found += 1
                        features_with_values.append(col)
                        print(f"   Matched '{key}' to '{col}'")
                        matched = True
                        break
                    except:
                        print(f"⚠️ Could not convert {key} to float for {col}")
            
            if not matched:
                print(f"⚠️ No match found for '{key}' (not in original features)")
    
    print(f"📊 Matched {features_found} out of {len(original_feature_names)} original features")
    
    # Show which features we have values for
    if features_with_values:
        print("📋 Features with values (first 10):")
        for col in features_with_values[:10]:
            print(f"   {col}: {X_aligned[col].iloc[0]}")
        if len(features_with_values) > 10:
            print(f"   ... and {len(features_with_values) - 10} more")
    
    # Apply preprocessing pipeline
    try:
        # Scale - use ALL original features
        if scaler:
            print(f"🔧 Scaling {len(original_feature_names)} original features...")
            X_scaled = scaler.transform(X_aligned)
            print(f"✅ Scaled features: {X_scaled.shape}")
        else:
            X_scaled = X_aligned.values
            print(f"⚠️ No scaler found, using raw features")
        
        # Feature selection - this will select the 50 features
        if feature_selector:
            X_selected = feature_selector.transform(X_scaled)
            print(f"✅ Selected {X_selected.shape[1]} features (from {X_scaled.shape[1]})")
        else:
            X_selected = X_scaled
            print(f"⚠️ No feature selector, using all {X_scaled.shape[1]} features")
        
        # PCA
        if pca:
            X_processed = pca.transform(X_selected)
            print(f"✅ Applied PCA: {X_processed.shape[1]} components")
        else:
            X_processed = X_selected
            print(f"✅ Final features for model: {X_processed.shape}")
        
        return X_processed
        
    except Exception as e:
        print(f"❌ Error in preprocessing: {e}")
        import traceback
        traceback.print_exc()
        return None

def predict_attack(sample_dict, model=None, label_encoder=None, preprocessor_data=None):
    """Predict attack type for a single sample"""
    # Load models if not provided
    if model is None or label_encoder is None or preprocessor_data is None:
        model, label_encoder, preprocessor_data = load_models()
    
    if model is None or label_encoder is None or preprocessor_data is None:
        return {
            "prediction": "Error loading models",
            "confidence": 0.0,
            "top_predictions": []
        }
    
    # Prepare features
    X_processed = prepare_features_for_prediction(sample_dict, preprocessor_data)
    
    if X_processed is None:
        return {
            "prediction": "Error preparing features",
            "confidence": 0.0,
            "top_predictions": []
        }
    
    # Predict
    try:
        # Get prediction
        pred_encoded = model.predict(X_processed)[0]
        pred_label = label_encoder.inverse_transform([pred_encoded])[0]
        
        # Get confidence and probabilities
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X_processed)[0]
            confidence = float(proba[pred_encoded])
        else:
            proba = None
            confidence = 1.0
        
        # Get top predictions
        top_predictions = []
        if proba is not None:
            # Get top 3 predictions
            top_indices = np.argsort(proba)[-3:][::-1]
            for idx in top_indices:
                label = label_encoder.inverse_transform([idx])[0]
                top_predictions.append({
                    "label": label,
                    "confidence": float(proba[idx])
                })
        
        # Check if we have the true label in the sample
        true_label = sample_dict.get('Label', 'Unknown')
        
        # Print results
        print(f"\n" + "="*50)
        print(f"🚨 PREDICTION RESULT")
        print("="*50)
        print(f"Predicted Attack Type: {pred_label}")
        print(f"Confidence: {confidence:.3f}")
        
        if true_label != 'Unknown':
            print(f"True Label: {true_label}")
            if pred_label == true_label:
                print(f"✅ CORRECT PREDICTION!")
            else:
                print(f"❌ MISMATCH - Expected: {true_label}")
        
        if top_predictions:
            print(f"\n🔝 Top predictions:")
            for i, pred in enumerate(top_predictions, 1):
                print(f"   {i}. {pred['label']}: {pred['confidence']:.3f}")
        
        print("="*50)
        
        return {
            "prediction": pred_label,
            "confidence": confidence,
            "top_predictions": top_predictions,
            "true_label": true_label,
            "is_correct": pred_label == true_label if true_label != 'Unknown' else None
        }
        
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        import traceback
        traceback.print_exc()
        return {
            "prediction": f"Prediction error: {str(e)}",
            "confidence": 0.0,
            "top_predictions": []
        }

# ----------------------------
# TEST FUNCTIONS WITH CORRECT FEATURE HANDLING
# ----------------------------
def test_attack_type_detection():
    """Test attack type detection with correct feature handling"""
    print("🚀 FEW-SHOT ATTACK TYPE DETECTION DEMO")
    print("="*60)
    
    # First inspect the preprocessor to understand feature structure
    preprocessor_data = load_and_inspect_preprocessor()
    if not preprocessor_data:
        print("❌ Failed to load preprocessor")
        return
    
    # Load models
    print("\n📦 Loading models...")
    model, label_encoder, preprocessor_data = load_models()
    
    if model is None:
        print("❌ Failed to load models. Exiting.")
        return
    
    print(f"\n📊 Available attack types: {list(label_encoder.classes_)}")
    
    # Get original feature names (for scaler)
    original_features = preprocessor_data.get('original_features_for_scaler', [])
    if original_features:
        print(f"\n📋 First 20 ORIGINAL features (need all {len(original_features)} for scaler):")
        for i, name in enumerate(original_features[:20], 1):
            print(f"   {i:2d}. {name}")
        print(f"   ... and {len(original_features) - 20} more")
    
    # Create test cases using features that are in the ORIGINAL feature set
    # Based on the inspection, we need to include common features
    test_cases = [
        {
            "name": "DDoS Attack",
            "features": {
                # Common features that are likely in original set
                "Destination Port": 80.0,
                "Flow Duration": 1000.0,
                "Total Fwd Packets": 10000.0,
                "Total Backward Packets": 100.0,
                "Total Length of Fwd Packets": 10000000.0,
                "Total Length of Bwd Packets": 5000.0,
                "Fwd Packet Length Max": 1500.0,
                "Fwd Packet Length Min": 100.0,
                "Fwd Packet Length Mean": 1400.0,
                "Fwd Packet Length Std": 150.0,
                "Bwd Packet Length Max": 100.0,
                "Bwd Packet Length Min": 10.0,
                "Bwd Packet Length Mean": 60.0,
                "Bwd Packet Length Std": 20.0,
                "Flow Packets/s": 1200.0,
                "Flow IAT Std": 0.0003,
                "Flow IAT Max": 0.001,
                "Flow IAT Min": 0.0001,
                "Fwd IAT Total": 0.8,
                "Fwd IAT Std": 0.0002,
                "Fwd IAT Max": 0.0005,
                "Bwd IAT Total": 0.1,
                "Fwd Header Length": 40.0,
                "Bwd Header Length": 0.0,
                "Fwd Packets/s": 800.0,
                "Bwd Packets/s": 400.0,
                "Min Packet Length": 100.0,
                "Max Packet Length": 1500.0,
                "Packet Length Mean": 1200.0,
                "Packet Length Std": 200.0,
                "Packet Length Variance": 40000.0,
                "PSH Flag Count": 0.0,
                "ACK Flag Count": 1.0,
                "Down/Up Ratio": 0.1,
                "Average Packet Size": 1200.0,
                "Avg Fwd Segment Size": 1400.0,
                "Avg Bwd Segment Size": 60.0,
                "Fwd Header Length.1": 40.0,
                "Subflow Fwd Packets": 10000.0,
                "Subflow Bwd Packets": 100.0,
                "Subflow Bwd Bytes": 5000.0,
                "Init_Win_bytes_forward": 65535.0,
                "Init_Win_bytes_backward": 0.0,
                "act_data_pkt_fwd": 0.0,
                "min_seg_size_forward": 40.0,
                "Idle Mean": 0.0,
                "Idle Max": 0.0,
                "Idle Min": 0.0,
                "bytes_per_packet": 1000.0,
                "mean_packet_length_diff": 1340.0,
                "iat_cov": 0.5,
                "Label": "DDoS"
            }
        },
        {
            "name": "PortScan Attack",
            "features": {
                "Destination Port": 22.0,
                "Flow Duration": 100.0,
                "Total Fwd Packets": 50.0,
                "Total Backward Packets": 0.0,
                "Total Length of Fwd Packets": 3000.0,
                "Total Length of Bwd Packets": 0.0,
                "Fwd Packet Length Max": 80.0,
                "Fwd Packet Length Min": 40.0,
                "Fwd Packet Length Mean": 60.0,
                "Fwd Packet Length Std": 10.0,
                "Bwd Packet Length Max": 0.0,
                "Bwd Packet Length Min": 0.0,
                "Bwd Packet Length Mean": 0.0,
                "Bwd Packet Length Std": 0.0,
                "Flow Packets/s": 500.0,
                "Flow IAT Std": 0.02,
                "Flow IAT Max": 0.05,
                "Flow IAT Min": 0.01,
                "Fwd IAT Total": 2.0,
                "Fwd IAT Std": 0.02,
                "Fwd IAT Max": 0.05,
                "Bwd IAT Total": 0.0,
                "Fwd Header Length": 40.0,
                "Bwd Header Length": 0.0,
                "Fwd Packets/s": 500.0,
                "Bwd Packets/s": 0.0,
                "Min Packet Length": 40.0,
                "Max Packet Length": 80.0,
                "Packet Length Mean": 60.0,
                "Packet Length Std": 10.0,
                "Packet Length Variance": 100.0,
                "PSH Flag Count": 0.0,
                "ACK Flag Count": 0.0,
                "Down/Up Ratio": 0.0,
                "Average Packet Size": 60.0,
                "Avg Fwd Segment Size": 60.0,
                "Avg Bwd Segment Size": 0.0,
                "Fwd Header Length.1": 40.0,
                "Subflow Fwd Packets": 50.0,
                "Subflow Bwd Packets": 0.0,
                "Subflow Bwd Bytes": 0.0,
                "Init_Win_bytes_forward": 65535.0,
                "Init_Win_bytes_backward": 0.0,
                "act_data_pkt_fwd": 0.0,
                "min_seg_size_forward": 40.0,
                "Idle Mean": 0.0,
                "Idle Max": 0.0,
                "Idle Min": 0.0,
                "bytes_per_packet": 60.0,
                "mean_packet_length_diff": 60.0,
                "iat_cov": 0.8,
                "Label": "PortScan"
            }
        }
    ]
    
    # Run all test cases
    results = []
    for test_case in test_cases:
        print(f"\n" + "="*60)
        print(f"🔍 TEST: {test_case['name']}")
        print("="*60)
        
        result = predict_attack(
            test_case['features'], 
            model, 
            label_encoder, 
            preprocessor_data
        )
        results.append(result)
    
    # Summary
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    
    correct_predictions = 0
    total_predictions = 0
    
    for i, (test_case, result) in enumerate(zip(test_cases, results), 1):
        true_label = test_case['features'].get('Label', 'Unknown')
        pred_label = result.get('prediction', 'Error')
        confidence = result.get('confidence', 0.0)
        
        is_correct = pred_label == true_label if true_label != 'Unknown' else None
        
        print(f"\nTest {i}: {test_case['name']}")
        print(f"  True: {true_label}")
        print(f"  Predicted: {pred_label}")
        print(f"  Confidence: {confidence:.3f}")
        
        if is_correct is not None:
            if is_correct:
                print(f"  ✅ CORRECT")
                correct_predictions += 1
            else:
                print(f"  ❌ WRONG")
            total_predictions += 1
    
    if total_predictions > 0:
        accuracy = correct_predictions / total_predictions
        print(f"\n🎯 Overall Accuracy: {correct_predictions}/{total_predictions} = {accuracy:.2%}")

def quick_test():
    """Quick test with a few key features"""
    print("🚀 QUICK TEST")
    
    # Test with a subset of features (all will default to 0 for missing ones)
    sample = {
        "Destination Port": 80.0,
        "Flow Duration": 1000.0,
        "Total Fwd Packets": 10000.0,
        "Total Backward Packets": 100.0,
        "bytes_per_packet": 1000.0,
        "Label": "DDoS"
    }
    
    result = predict_attack(sample)
    print(f"\nResult: {result}")

def show_all_features():
    """Show all original features to help create test cases"""
    preprocessor_data = load_and_inspect_preprocessor()
    if not preprocessor_data:
        return
    
    if 'original_feature_names' in preprocessor_data:
        features = preprocessor_data['original_feature_names']
        print(f"\n📋 ALL {len(features)} ORIGINAL FEATURES:")
        print("="*80)
        for i, feature in enumerate(features, 1):
            print(f"{i:3d}. {feature}")
        print("="*80)
        print(f"\n💡 Tip: Use these exact feature names in your test cases")
        print(f"   Missing features will be set to 0.0 automatically")

# ----------------------------
# MAIN EXECUTION
# ----------------------------
if __name__ == "__main__":
    print("Select mode:")
    print("1. Test attack detection (with comprehensive sample data)")
    print("2. Show all original features (for creating test cases)")
    print("3. Quick test (minimal features)")
    print("4. Interactive test")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == "1":
        test_attack_type_detection()
    elif choice == "2":
        show_all_features()
    elif choice == "3":
        quick_test()
    elif choice == "4":
        print("\nComing soon - use option 1 or 3 for now")
        quick_test()
    else:
        # Default to showing features
        show_all_features()