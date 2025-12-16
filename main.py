from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import os
import threading
import time
import queue

from utils.preprocessor import DataPreprocessor
from utils.live_capture import LiveTrafficCapture
from utils.real_time_predictor import RealTimePredictor
from config import MODELS_DIR

app = Flask(__name__)

# Global variables
models = {}
preprocessor = DataPreprocessor()
live_capture = LiveTrafficCapture()
real_time_predictor = RealTimePredictor()
is_real_time_running = False
realtime_logs = queue.Queue()
attack_clf = None
attack_label_encoder = None

def inspect_attack_preprocessor():
    """Inspect the attack preprocessor dictionary"""
    try:
        attack_preprocessor_path = os.path.join(MODELS_DIR, "attack_preprocessor.joblib")
        if os.path.exists(attack_preprocessor_path):
            preprocessor_dict = joblib.load(attack_preprocessor_path)
            
            print("\n" + "="*60)
            print("INSPECTING ATTACK PREPROCESSOR DICTIONARY")
            print("="*60)
            
            for key, value in preprocessor_dict.items():
                if value is None:
                    print(f"{key}: None")
                else:
                    print(f"{key}: {type(value)}")
                    if hasattr(value, 'n_features_in_'):
                        print(f"  → n_features_in_: {value.n_features_in_}")
                    if hasattr(value, 'n_components'):
                        print(f"  → n_components: {value.n_components}")
                    if hasattr(value, 'n_features_out'):
                        print(f"  → n_features_out: {getattr(value, 'n_features_out', 'N/A')}")
            
            # Check feature counts
            if 'scaler' in preprocessor_dict and preprocessor_dict['scaler'] is not None:
                print(f"\nScaler input features: {preprocessor_dict['scaler'].n_features_in_}")
            
            if 'pca' in preprocessor_dict and preprocessor_dict['pca'] is not None:
                print(f"PCA components: {preprocessor_dict['pca'].n_components_}")
                print(f"PCA input features: {preprocessor_dict['pca'].n_features_in_}")
                
            print("="*60 + "\n")
            
        else:
            print(f"❌ Preprocessor not found at {attack_preprocessor_path}")
            
    except Exception as e:
        print(f"❌ Inspection error: {e}")

def debug_preprocessor():
    """Debug the attack type preprocessor"""
    try:
        attack_preprocessor_path = os.path.join(MODELS_DIR, "attack_preprocessor.joblib")
        if os.path.exists(attack_preprocessor_path):
            preprocessor = joblib.load(attack_preprocessor_path)
            
            print("\n" + "="*60)
            print("DEBUG: Attack Preprocessor Info")
            print("="*60)
            
            # Check what type of object it is
            print(f"Preprocessor type: {type(preprocessor)}")
            
            # Check if it's a pipeline
            if hasattr(preprocessor, 'steps'):
                print(f"Pipeline steps: {preprocessor.steps}")
                
            # Check expected input features
            if hasattr(preprocessor, 'n_features_in_'):
                print(f"Expected input features: {preprocessor.n_features_in_}")
            
            # Check if it has a scaler
            if hasattr(preprocessor, 'named_steps'):
                for name, step in preprocessor.named_steps.items():
                    print(f"Step '{name}': {type(step)}")
            
            print("="*60 + "\n")
            
        else:
            print(f"❌ Preprocessor not found at {attack_preprocessor_path}")
            
    except Exception as e:
        print(f"❌ Debug error: {e}")


def detect_and_classify(traffic_features, preprocessor, models, attack_clf=None, attack_label_encoder=None):
    """
    Detect attack and classify type if attack is detected
    Returns: (is_attack, attack_type, confidence)
    """
    try:
        # Prepare features for binary detection
        features_df = pd.DataFrame([traffic_features])
        X_processed = preprocessor.prepare_real_time_features(features_df)
        
        if X_processed is None:
            return False, "Unknown", 0.0
        
        # Use Random Forest for detection (or your preferred model)
        model = models.get('random_forest')
        if model is None:
            return False, "Unknown", 0.0
        
        # Detect if it's an attack
        is_attack = model.predict(X_processed)[0] == 1
        confidence = np.max(model.predict_proba(X_processed)[0])
        
        attack_type = "BENIGN"
        attack_type_confidence = 0.0
        
        # If attack detected, try to classify type
        if is_attack and attack_clf is not None and attack_label_encoder is not None:
            try:
                # Load attack type preprocessor (FIXED PATH)
                attack_preprocessor_path = os.path.join(MODELS_DIR, "attack_preprocessor.joblib")
                if os.path.exists(attack_preprocessor_path):
                    attack_preprocessor_dict = joblib.load(attack_preprocessor_path)  # This is a DICT
                    
                    # Load the features used for attack type classification
                    attack_features_path = os.path.join(MODELS_DIR, "attack_type_features.joblib")
                    if os.path.exists(attack_features_path):
                        attack_features = joblib.load(attack_features_path)
                        
                        # Prepare features for attack type classification
                        # Get only the features that the attack type classifier expects
                        available_features = [f for f in attack_features if f in features_df.columns]
                        
                        if len(available_features) == 0:
                            attack_type = "Attack_Generic"
                            print(f"⚠️ No matching features for attack type classification")
                            return is_attack, attack_type, confidence
                        
                        # Prepare features for attack type classifier
                        X_attack = features_df[available_features].copy()
                        
                        # Fill missing features
                        missing_features = set(attack_features) - set(available_features)
                        for feature in missing_features:
                            X_attack[feature] = 0.0
                        
                        # Reorder columns to match training order
                        X_attack = X_attack[attack_features]
                        
                        # DEBUG: Check what's in the preprocessor dict
                        print(f"DEBUG: Preprocessor dict keys: {list(attack_preprocessor_dict.keys())}")
                        
                        # Apply preprocessing steps from the dictionary
                        X_attack_processed = X_attack.values  # Start with raw values
                        
                        # Step 1: Apply scaler if exists
                        if 'scaler' in attack_preprocessor_dict and attack_preprocessor_dict['scaler'] is not None:
                            X_attack_scaled = attack_preprocessor_dict['scaler'].transform(X_attack_processed)
                            X_attack_processed = X_attack_scaled
                            print(f"DEBUG: After scaling: {X_attack_processed.shape}")
                        
                        # Step 2: Apply feature selector if exists
                        if ('feature_selector' in attack_preprocessor_dict and 
                            attack_preprocessor_dict['feature_selector'] is not None):
                            X_attack_selected = attack_preprocessor_dict['feature_selector'].transform(X_attack_processed)
                            X_attack_processed = X_attack_selected
                            print(f"DEBUG: After feature selection: {X_attack_processed.shape}")
                        
                        # Step 3: Apply PCA if exists
                        if 'pca' in attack_preprocessor_dict and attack_preprocessor_dict['pca'] is not None:
                            X_attack_pca = attack_preprocessor_dict['pca'].transform(X_attack_processed)
                            X_attack_processed = X_attack_pca
                            print(f"DEBUG: After PCA: {X_attack_processed.shape}")
                        
                        # Final shape should be 15 features
                        print(f"DEBUG: Final processed shape: {X_attack_processed.shape}")
                        
                        # Check if dimensions match the classifier
                        if hasattr(attack_clf, 'n_features_in_'):
                            expected_features = attack_clf.n_features_in_
                            actual_features = X_attack_processed.shape[1]
                            print(f"DEBUG: Classifier expects {expected_features}, got {actual_features}")
                            
                            if actual_features != expected_features:
                                print(f"❌ Feature mismatch: Classifier expects {expected_features}, got {actual_features}")
                                # Try to reshape if needed
                                if actual_features > expected_features:
                                    X_attack_processed = X_attack_processed[:, :expected_features]
                                    print(f"DEBUG: Truncated to first {expected_features} features")
                                elif actual_features < expected_features:
                                    # Pad with zeros
                                    padding = np.zeros((X_attack_processed.shape[0], expected_features - actual_features))
                                    X_attack_processed = np.hstack([X_attack_processed, padding])
                                    print(f"DEBUG: Padded with zeros to {expected_features} features")
                        
                        # Predict attack type
                        attack_pred = attack_clf.predict(X_attack_processed)[0]
                        attack_type = attack_label_encoder.inverse_transform([attack_pred])[0]
                        attack_type_confidence = np.max(attack_clf.predict_proba(X_attack_processed)[0])
                        
                        # Print to terminal in a nice format
                        print("\n" + "="*60)
                        print("🔴 ATTACK DETECTED!")
                        print("="*60)
                        print(f"   Attack Type: {attack_type}")
                        print(f"   Type Confidence: {attack_type_confidence:.2%}")
                        print(f"   Binary Confidence: {confidence:.2%}")
                        print(f"   Timestamp: {time.strftime('%H:%M:%S')}")
                        print("="*60)
                        
                    else:
                        attack_type = "Attack_Generic"
                        print(f"⚠️ Attack type features file not found")
                else:
                    attack_type = "Attack_Generic"
                    print(f"⚠️ Attack preprocessor not found at {attack_preprocessor_path}")
                
            except Exception as e:
                print(f"⚠️ Could not classify attack type: {e}")
                import traceback
                traceback.print_exc()
                attack_type = "Attack_Generic"
                print(f"🔴 Generic Attack Detected (confidence: {confidence:.2%})")
        
        return is_attack, attack_type, confidence
        
    except Exception as e:
        print(f"❌ Detection error: {e}")
        import traceback
        traceback.print_exc()
        return False, "Unknown", 0.0
def load_saved_models():
    """Load all pre-trained models"""
    global models, preprocessor, attack_clf, attack_label_encoder
    
    try:
        # Load preprocessor
        inspect_attack_preprocessor()
        debug_preprocessor()
        preprocessor_path = os.path.join(MODELS_DIR, 'preprocessor.joblib')
        if os.path.exists(preprocessor_path):
            preprocessor.load_preprocessor(preprocessor_path)
        
        # Load binary attack detection models
        model_files = {
            'logistic_regression': 'logistic_regression.joblib',
            'decision_tree': 'decision_tree.joblib',
            'random_forest': 'random_forest.joblib',
            'svm': 'svm.joblib',
            'gaussian_nb': 'gaussian_nb.joblib',
            'knn': 'knn.joblib',
            'isolation_forest': 'isolation_forest.joblib',
            'xgboost': 'xgboost.joblib'
        }
        
        loaded_count = 0
        for name, filename in model_files.items():
            model_path = os.path.join(MODELS_DIR, filename)
            if os.path.exists(model_path):
                try:
                    models[name] = joblib.load(model_path)
                    print(f"✅ Loaded {name}")
                    loaded_count += 1
                except Exception as e:
                    print(f"❌ Error loading {name}: {e}")
            else:
                print(f"⚠️ Model file not found: {filename}")
        
        # Load attack type classifier
        attack_clf = None
        attack_label_encoder = None
        attack_type_path = os.path.join(MODELS_DIR, "attack_type_classifier.joblib")
        attack_label_path = os.path.join(MODELS_DIR, "attack_label_encoder.joblib")
        
        if os.path.exists(attack_type_path) and os.path.exists(attack_label_path):
            try:
                attack_clf = joblib.load(attack_type_path)
                attack_label_encoder = joblib.load(attack_label_path)
                print(f"✅ Loaded Attack Type Classifier")
                print(f"📋 Attack classes: {list(attack_label_encoder.classes_)}")
            except Exception as e:
                print(f"❌ Error loading attack type classifier: {e}")
        else:
            print(f"⚠️ Attack type classifier files not found")
        
        print(f"📊 Total binary models loaded: {loaded_count}/{len(model_files)}")
        return loaded_count > 0
        
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        return False
# Update the real_time_predictor initialization
real_time_predictor = RealTimePredictor()
def add_realtime_log(message, log_type="system"):
    """Add log message to real-time queue"""
    timestamp = time.strftime("%H:%M:%S")
    log_entry = {
        'timestamp': timestamp,
        'message': message,
        'type': log_type
    }
    realtime_logs.put(log_entry)

def realtime_monitoring_loop():
    """Main real-time monitoring loop"""
    global is_real_time_running
    
    try:
        add_realtime_log("Starting live network capture...", "system")
        real_time_predictor.capture.start_capture()
        
        while is_real_time_running:
            try:
                # Process captured packets
                packets = real_time_predictor.capture.get_captured_packets(max_packets=50)
                
                if packets:
                    add_realtime_log(f"Captured {len(packets)} packets for analysis", "system")
                    
                    # Convert to DataFrame and process
                    df = pd.DataFrame(packets)
                    
                    try:
                        # Prepare features for prediction
                        X_processed = real_time_predictor.preprocessor.prepare_real_time_features(df)
                        
                        # Make predictions and store them
                        predictions = real_time_predictor._make_predictions(df, X_processed)
                        
                        # Check each prediction for attacks and classify them
                        for i, row in df.iterrows():
                            # Create feature dictionary from the row
                            features = row.to_dict()
                            
                            # Detect and classify attack
                            is_attack, attack_type, confidence = detect_and_classify(
                                features, 
                                real_time_predictor.preprocessor,
                                real_time_predictor.models,
                                attack_clf,
                                attack_label_encoder
                            )
                            
                            # Log to real-time queue
                            if is_attack:
                                add_realtime_log(
                                    f"🚨 Attack detected: {attack_type} (confidence: {confidence:.2%})",
                                    "threat"
                                )
                        
                        add_realtime_log(f"Analyzed {len(df)} packets (live)", "system")
                    
                    except Exception as e:
                        add_realtime_log(f"Analysis error: {str(e)}", "error")
                
                # Brief sleep to prevent CPU overload
                time.sleep(2)
                
            except Exception as e:
                add_realtime_log(f"Monitoring error: {str(e)}", "error")
                time.sleep(5)
                
    except Exception as e:
        add_realtime_log(f"Real-time monitoring failed: {str(e)}", "error")
    finally:
        real_time_predictor.capture.stop_capture()
        add_realtime_log("Live monitoring stopped", "system")
        is_real_time_running = False

@app.route('/')
def dashboard():
    """Main dashboard page"""
    return render_template('dashboard.html')

@app.route('/enable_threat_mode', methods=['POST'])
def enable_threat_mode():
    """Enable threat simulation"""
    try:
        real_time_predictor.enable_threat_mode(True)
        return jsonify({'status': 'success', 'message': 'Threat simulation enabled'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/disable_threat_mode', methods=['POST'])
def disable_threat_mode():
    """Disable threat simulation"""
    try:
        real_time_predictor.enable_threat_mode(False)
        return jsonify({'status': 'success', 'message': 'Threat simulation disabled'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/inject_threat/<threat_type>', methods=['POST'])
def inject_threat(threat_type):
    """Inject specific threat type"""
    try:
        result = real_time_predictor.inject_manual_threat(threat_type)
        return jsonify({'status': 'success', 'message': result})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/threat_types')
def get_threat_types():
    """Get available threat types"""
    threat_types = [
        {'id': 'port_scan', 'name': 'Port Scanning', 'description': 'Network reconnaissance'},
        {'id': 'syn_flood', 'name': 'SYN Flood', 'description': 'DDoS attack simulation'},
        {'id': 'ddos', 'name': 'DDoS Pattern', 'description': 'Distributed denial of service'},
        {'id': 'malformed', 'name': 'Malformed Packets', 'description': 'Protocol violations'},
        {'id': 'suspicious', 'name': 'Suspicious Payloads', 'description': 'Malicious content'}
    ]
    return jsonify({'threat_types': threat_types})

@app.route('/threat_status')
def get_threat_status():
    """Get current threat simulation status"""
    return jsonify({
        'threat_mode_enabled': real_time_predictor.threat_mode,
        'message': 'Threat simulation active' if real_time_predictor.threat_mode else 'Threat simulation inactive'
    })


@app.route('/start_realtime', methods=['POST'])
def start_realtime():
    """Start real-time protection"""
    global is_real_time_running
    
    if is_real_time_running:
        return jsonify({
            'status': 'error', 
            'message': 'Real-time protection already running'
        })
    
    try:
        # Load models if not already loaded
        if not real_time_predictor.models:
            real_time_predictor.load_models()
        
        # Clear any stale realtime predictions before starting
        real_time_predictor.realtime_predictions.clear()
        
        is_real_time_running = True
        
        # Start real-time monitoring in a separate thread
        monitor_thread = threading.Thread(target=realtime_monitoring_loop, daemon=True)
        monitor_thread.start()
        
        return jsonify({
            'status': 'success', 
            'message': 'Real-time protection activated'
        })
        
    except Exception as e:
        is_real_time_running = False
        return jsonify({
            'status': 'error', 
            'message': f'Failed to start real-time: {str(e)}'
        })

@app.route('/stop_realtime', methods=['POST'])
def stop_realtime():
    """Stop real-time protection"""
    global is_real_time_running
    is_real_time_running = False
    return jsonify({
        'status': 'success', 
        'message': 'Real-time protection stopped'
    })

@app.route('/get_realtime_logs')
def get_realtime_logs():
    """Get real-time log messages"""
    logs = []
    while not realtime_logs.empty():
        try:
            logs.append(realtime_logs.get_nowait())
        except queue.Empty:
            break
    return jsonify({'logs': logs})

@app.route('/get_realtime_predictions')
def get_realtime_predictions():
    """Return latest real-time prediction summaries"""
    preds = list(real_time_predictor.realtime_predictions)
    return jsonify({'predictions': preds})

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for prediction with all models"""
    try:
        # Check if preprocessor is loaded and fitted
        if not hasattr(preprocessor, 'scaler') or not hasattr(preprocessor.scaler, 'mean_'):
            return jsonify({
                'status': 'error',
                'message': 'Preprocessor not fitted. Please train models first.'
            })
        
        data = request.get_json()
        
        # Convert input to DataFrame
        input_df = pd.DataFrame([data])
        
        # Preprocess input using real-time method
        try:
            X_processed = preprocessor.prepare_real_time_features(input_df)
        except ValueError as e:
            return jsonify({
                'status': 'error', 
                'message': f'Preprocessing error: {str(e)}'
            })
        
        # Get predictions from all models
        predictions = {}
        model_performance = {}
        
        # Also do attack type classification if available
        attack_type_info = None
        
        for name, model in models.items():
            try:
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(X_processed)[0]
                    raw_pred = model.predict(X_processed)[0]

                    if isinstance(raw_pred, str):
                        is_benign = raw_pred.strip().upper() == 'BENIGN'
                        prediction = 0 if is_benign else 1
                        if hasattr(model, 'classes_') and 'BENIGN' in model.classes_:
                            benign_idx = list(model.classes_).index('BENIGN')
                            benign_prob = proba[benign_idx]
                            attack_conf = 1.0 - benign_prob
                            confidence = float(max(attack_conf, benign_prob))
                        else:
                            confidence = float(max(proba))
                        label = 'ATTACK' if prediction == 1 else 'NORMAL'
                    else:
                        prediction = int(raw_pred)
                        confidence = float(max(proba))
                        label = 'ATTACK' if prediction == 1 else 'NORMAL'
                    
                    model_performance[name] = {
                        'type': 'classification',
                        'class_probabilities': {str(cls): float(p) for cls, p in zip(model.classes_, proba)}
                    }
                    
                    # Print to terminal if it's an attack
                    if label == 'ATTACK':
                        print(f"\n🔍 {name} detected ATTACK (confidence: {confidence:.2%})")
                        
                        # Try to classify attack type if classifier exists
                       # Try to classify attack type if classifier exists
# Try to classify attack type if classifier exists
                        # Try to classify attack type if classifier exists
                        if attack_clf is not None and attack_label_encoder is not None:
                            try:
                                # Load attack type preprocessor
                                attack_preprocessor_path = os.path.join(MODELS_DIR, "attack_preprocessor.joblib")
                                if os.path.exists(attack_preprocessor_path):
                                    attack_preprocessor_dict = joblib.load(attack_preprocessor_path)  # This is a DICT

                                    # Load attack features
                                    attack_features_path = os.path.join(MODELS_DIR, "attack_type_features.joblib")
                                    if os.path.exists(attack_features_path):
                                        attack_features = joblib.load(attack_features_path)

                                        # Prepare features for attack type classification
                                        available_features = [f for f in attack_features if f in input_df.columns]

                                        if len(available_features) > 0:
                                            X_attack = input_df[available_features].copy()

                                            # Fill missing features
                                            missing_features = set(attack_features) - set(available_features)
                                            for feature in missing_features:
                                                X_attack[feature] = 0.0

                                            # Reorder columns to match training order
                                            X_attack = X_attack[attack_features]

                                            # DEBUG: Check what's in the preprocessor dict
                                            print(f"DEBUG [/predict]: Preprocessor dict keys: {list(attack_preprocessor_dict.keys())}")

                                            # Apply preprocessing steps sequentially
                                            X_attack_processed = X_attack.values

                                            # 1. Apply scaler
                                            if 'scaler' in attack_preprocessor_dict and attack_preprocessor_dict['scaler'] is not None:
                                                X_attack_processed = attack_preprocessor_dict['scaler'].transform(X_attack_processed)

                                            # 2. Apply feature selector (if exists)
                                            if ('feature_selector' in attack_preprocessor_dict and 
                                                attack_preprocessor_dict['feature_selector'] is not None):
                                                X_attack_processed = attack_preprocessor_dict['feature_selector'].transform(X_attack_processed)

                                            # 3. Apply PCA (if exists) - THIS IS LIKELY THE STEP THAT REDUCES TO 15 FEATURES
                                            if 'pca' in attack_preprocessor_dict and attack_preprocessor_dict['pca'] is not None:
                                                X_attack_processed = attack_preprocessor_dict['pca'].transform(X_attack_processed)

                                            print(f"DEBUG [/predict]: Final processed shape: {X_attack_processed.shape}")

                                            # Check dimensions
                                            if hasattr(attack_clf, 'n_features_in_'):
                                                expected = attack_clf.n_features_in_
                                                actual = X_attack_processed.shape[1]

                                                if actual != expected:
                                                    print(f"⚠️ Feature count mismatch: Classifier expects {expected}, got {actual}")
                                                    # Fix the mismatch
                                                    if actual > expected:
                                                        X_attack_processed = X_attack_processed[:, :expected]
                                                    elif actual < expected:
                                                        padding = np.zeros((X_attack_processed.shape[0], expected - actual))
                                                        X_attack_processed = np.hstack([X_attack_processed, padding])

                                            # Predict attack type
                                            attack_pred = attack_clf.predict(X_attack_processed)[0]
                                            attack_type = attack_label_encoder.inverse_transform([attack_pred])[0]
                                            attack_type_confidence = np.max(attack_clf.predict_proba(X_attack_processed)[0])

                                            print(f"   Attack Type: {attack_type}")
                                            print(f"   Type Confidence: {attack_type_confidence:.2%}")

                                            # Store attack type info for API response
                                            if attack_type_info is None:
                                                attack_type_info = {
                                                    'attack_type': attack_type,
                                                    'confidence': attack_type_confidence,
                                                    'timestamp': time.strftime('%H:%M:%S')
                                                }

                            except Exception as e:
                                print(f"   Could not classify attack type: {e}")
                                import traceback
                                traceback.print_exc()
                
                else:
                    # Anomaly detection models (Isolation Forest)
                    raw_prediction = model.predict(X_processed)[0]
                    is_anomaly = (raw_prediction == -1)
                    prediction = 1 if is_anomaly else 0
                    confidence = 0.9 if is_anomaly else 0.1
                    label = 'ATTACK' if is_anomaly else 'NORMAL'
                    
                    model_performance[name] = {
                        'type': 'anomaly_detection',
                        'raw_score': float(raw_prediction)
                    }
                    
                    # Print to terminal if it's an attack
                    if label == 'ATTACK':
                        print(f"\n🔍 {name} detected ANOMALY")
                
                predictions[name] = {
                    'prediction': prediction,
                    'confidence': confidence,
                    'label': label
                }
                
            except Exception as model_error:
                print(f"❌ Error with model {name}: {model_error}")
                predictions[name] = {
                    'prediction': 0,
                    'confidence': 0.0,
                    'label': 'ERROR',
                    'error': str(model_error)
                }
        
        # Calculate ensemble decision
        attack_votes = sum(1 for p in predictions.values() if p.get('label') == 'ATTACK')
        total_models = len(predictions)
        ensemble_confidence = attack_votes / total_models
        
        # Print summary to terminal
        if attack_votes > 0:
            print(f"\n🎯 Ensemble: {attack_votes}/{total_models} models detected threats")
            if attack_type_info:
                print(f"📊 Most Likely Attack Type: {attack_type_info['attack_type']} "
                      f"(confidence: {attack_type_info['confidence']:.2%})")
        
        # Prepare response
        response = {
            'status': 'success',
            'predictions': predictions,
            'ensemble': {
                'attack_votes': attack_votes,
                'total_models': total_models,
                'consensus': 'ATTACK' if attack_votes > total_models / 2 else 'NORMAL',
                'confidence': ensemble_confidence
            },
            'model_performance': model_performance
        }
        
        # Add attack type info if available
        if attack_type_info:
            response['attack_type'] = attack_type_info
        
        return jsonify(response)
    
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        })

@app.route('/system_status')
def system_status():
    """Get system status"""
    models_loaded = len(models) > 0
    preprocessor_fitted = hasattr(preprocessor, 'scaler') and hasattr(preprocessor.scaler, 'mean_')
    
    status_info = {
        'models_loaded': models_loaded,
        'models_count': len(models),
        'preprocessor_fitted': preprocessor_fitted,
        'system_status': 'OPERATIONAL' if (models_loaded and preprocessor_fitted) else 'SETUP_REQUIRED',
        'realtime_running': is_real_time_running
    }
    
    return jsonify(status_info)

if __name__ == '__main__':
    print("🚀 Starting BlackWall Cyber Defense System...")
    
    # Load models at startup
    if load_saved_models():
        print("✅ All models loaded successfully")
        print("🌐 Starting Flask server on http://localhost:5001")
        app.run(debug=True, host='0.0.0.0', port=5001)
    else:
        print("❌ Failed to load models. Please train models first.")
        print("💡 Run: python train_model.py")
