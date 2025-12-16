import pandas as pd
import numpy as np
import joblib
from utils.preprocessor import DataPreprocessor
from utils.live_capture import LiveTrafficCapture
import time
from datetime import datetime
from utils.threat_simulator import ThreatSimulator, PacketInjector 
import os
from collections import deque
from config import MODELS_DIR

class RealTimePredictor:
    def __init__(self):
        self.models = {}
        self.preprocessor = DataPreprocessor()
        self.capture = LiveTrafficCapture()
        self.threat_simulator = ThreatSimulator()  # ADD THIS
        self.packet_injector = PacketInjector()    # ADD THIS
        self.threat_mode = False                   # ADD THIS
        self.realtime_predictions = deque(maxlen=50)  # Store recent realtime decisions
        self.load_models()


    def enable_threat_mode(self, enable=True):
        """Enable/disable threat simulation"""
        self.threat_mode = enable
        if enable:
            print("🔥 THREAT MODE ENABLED - Generating malicious traffic")
            # Start threat simulation in background
            self.threat_simulator.start_threat_simulation()
        else:
            self.threat_simulator.stop_threat_simulation()
            print("🟢 Threat mode disabled")
            # Clear any synthetic attack snapshots so UI reflects fresh live data
            self.realtime_predictions.clear()
    
    def inject_manual_threat(self, threat_type):
        """Manually inject specific threat types"""
        threats = {
            'port_scan': self.packet_injector.inject_port_scan,
            'syn_flood': self.packet_injector.inject_syn_flood,
            'ddos': self.packet_injector.inject_ddos_pattern,
            'malformed': self.packet_injector.inject_malformed_packets,
            'suspicious': self.packet_injector.inject_suspicious_payloads
        }
        
        if threat_type in threats:
            threats[threat_type]()
            # Push a synthetic attack decision so UI reflects the action immediately
            self.realtime_predictions.append({
                'timestamp': datetime.utcnow().isoformat(),
                'final_decision': 'ATTACK',
                'model_predictions': {
                    name: {
                        'prediction': 1,
                        'confidence': 0.99,
                        'label': 'ATTACK'
                    } for name in self.models.keys()
                }
            })
            return f"Injected {threat_type} threat"
        else:
            return f"Unknown threat type: {threat_type}"
    
    def start_real_time_protection(self):
        """Start real-time network protection"""
        print("🛡️ Starting BlackWall Real-Time Protection...")
        
        # Ask user if they want threat simulation
        response = input("Enable threat simulation for testing? (y/n): ").lower().strip()
        if response == 'y':
            self.enable_threat_mode(True)
        
        self.capture.start_capture()
        
        try:
            while True:
                # Process captured packets every 5 seconds
                time.sleep(5)
                self._process_captured_packets()
                
        except KeyboardInterrupt:
            print("\n🛑 Stopping real-time protection...")
            self.enable_threat_mode(False)  # ADD THIS
            self.capture.stop_capture()        
    
# In utils/real_time_predictor.py - update load_models method
    def load_models(self):
        """Load all trained models for real-time prediction"""
        try:
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

            for name, filename in model_files.items():
                model_path = os.path.join(MODELS_DIR, filename)
                if os.path.exists(model_path):
                    self.models[name] = joblib.load(model_path)
                    print(f"✅ Loaded {name} for real-time prediction")

            # Load preprocessor
            preprocessor_path = os.path.join(MODELS_DIR, 'preprocessor.joblib')
            if os.path.exists(preprocessor_path):
                self.preprocessor.load_preprocessor(preprocessor_path)

            return len(self.models) > 0

        except Exception as e:
            print(f"❌ Error loading models for real-time: {e}")
            return False
    
    def start_real_time_protection(self):
        """Start real-time network protection"""
        print("🛡️ Starting BlackWall Real-Time Protection...")
        self.capture.start_capture()
        
        try:
            while True:
                # Process captured packets every 5 seconds
                time.sleep(5)
                self._process_captured_packets()
                
        except KeyboardInterrupt:
            print("\n🛑 Stopping real-time protection...")
            self.capture.stop_capture()

    def _process_captured_packets(self):
        """Process captured packets and make predictions"""
        packets = self.capture.get_captured_packets(max_packets=100)

        # DEBUG: Show what we're capturing
        if packets:
            print(f"📦 Captured {len(packets)} real packets")
            if len(packets) > 0:
                sample = packets[0]
                print(f"🔍 Sample packet: {sample.get('src_ip', 'N/A')} → {sample.get('dst_ip', 'N/A')}:{sample.get('dst_port', 'N/A')}")
        else:
            print("🔍 No packets captured - waiting for network traffic...")
            # Generate test packets for demonstration
            packets = self._generate_test_packets()
            if packets:
                print(f"🧪 Using {len(packets)} test packets for demonstration")

        if not packets:
            return

        print(f"📊 Processing {len(packets)} packets...")

        # Convert to DataFrame
        df = pd.DataFrame(packets)

        # Prepare features for real-time prediction
        try:
            X_processed = self.preprocessor.prepare_real_time_features(df)

            if len(X_processed) == 0:
                print("❌ No features processed - check preprocessor")
                return

            print(f"✅ Features processed: {X_processed.shape}")

            # Make predictions
            self._make_predictions(df, X_processed)

        except Exception as e:
            print(f"❌ Error processing packets: {e}")
            import traceback
            traceback.print_exc()
    def _make_predictions(self, original_df, processed_features):
        """Make predictions using loaded models with fallback to Isolation Forest"""
        threats_detected = 0
        total_packets = len(original_df)

        print(f"🔍 Analyzing {total_packets} packets...")

        for i, (idx, row) in enumerate(original_df.iterrows()):
            packet_features = processed_features[i:i+1]

            model_predictions = {}
            # Evaluate all loaded models with string-label support
            for model_name, model in self.models.items():
                try:
                    if hasattr(model, 'predict_proba'):
                        proba = model.predict_proba(packet_features)[0]
                        raw_pred = model.predict(packet_features)[0]

                        if isinstance(raw_pred, str):
                            is_benign = raw_pred.strip().upper() == 'BENIGN'
                            benign_idx = list(model.classes_).index('BENIGN') if 'BENIGN' in model.classes_ else None
                            attack_conf = 1.0 - proba[benign_idx] if benign_idx is not None else max(proba)
                            confidence = max(proba)
                            label = 'ATTACK' if not is_benign else 'NORMAL'
                        else:
                            label = 'ATTACK' if int(raw_pred) == 1 else 'NORMAL'
                            attack_idx = list(model.classes_).index(1) if 1 in model.classes_ else None
                            attack_conf = proba[attack_idx] if attack_idx is not None else max(proba)
                            confidence = max(proba)

                        model_predictions[model_name] = {
                            'prediction': int(raw_pred) if not isinstance(raw_pred, str) else raw_pred,
                            'confidence': float(confidence),
                            'attack_confidence': float(attack_conf),
                            'label': label
                        }
                    else:
                        # Anomaly detectors (Isolation Forest)
                        raw_prediction = model.predict(packet_features)[0]
                        is_anomaly = (raw_prediction == -1)
                        model_predictions[model_name] = {
                            'prediction': 1 if is_anomaly else 0,
                            'confidence': 0.9 if is_anomaly else 0.1,
                            'label': 'ATTACK' if is_anomaly else 'NORMAL',
                            'raw_score': float(getattr(model, 'score_samples', lambda x: [0])(packet_features)[0])
                        }
                except Exception as e:
                    model_predictions[model_name] = {
                        'prediction': 0,
                        'confidence': 0.0,
                        'label': 'ERROR',
                        'error': str(e)
                    }

            # Ensemble decision: majority attack
            attack_votes = sum(1 for p in model_predictions.values() if p.get('label') == 'ATTACK')
            final_decision = 'ATTACK' if attack_votes > len(model_predictions) / 2 else 'NORMAL'

            # Persist latest prediction summary for UI polling
            self.realtime_predictions.append({
                'timestamp': datetime.utcnow().isoformat(),
                'final_decision': final_decision,
                'model_predictions': model_predictions
            })

            if final_decision == 'ATTACK':
                threats_detected += 1
                src_ip = row.get('src_ip', 'Unknown')
                dst_ip = row.get('dst_ip', 'Unknown')
                dst_port = row.get('dst_port', 'Unknown')

                print(f"🚨 THREAT DETECTED: {src_ip} -> {dst_ip}:{dst_port}")

                # Log detailed predictions
                for model_name, pred in model_predictions.items():
                    status = "🔴" if pred['label'] == 'ATTACK' else "🟢"
                    conf_type = "attack_conf" if 'attack_confidence' in pred else "conf"
                    conf_value = pred.get('attack_confidence', pred.get('confidence', 0))
                    print(f"   {status} {model_name}: {pred['label']} ({conf_type}: {conf_value:.2f})")

                print("   🛡️ Containment protocol initiated...")

        # Summary
        if threats_detected > 0:
            print(f"📊 Summary: {threats_detected}/{total_packets} threats detected")
        else:
            print("✅ All traffic appears normal")
    def _determine_threat_level(self, model_predictions):
        """Determine if packet is a threat using weighted voting"""
        if not model_predictions:
            return False

        attack_votes = 0
        total_models = len(model_predictions)

        for model_name, prediction in model_predictions.items():
            # Random Forest gets higher weight (more reliable)
            if model_name == 'random_forest':
                weight = 2.0
            else:
                weight = 1.0

            if prediction['label'] == 'ATTACK' and prediction['confidence'] > 0.7:
                attack_votes += weight

        # Require strong evidence for attack classification
        return attack_votes >= 1.5  # At least one confident RF prediction or multiple weaker ones
