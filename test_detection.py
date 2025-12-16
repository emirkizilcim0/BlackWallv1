#!/usr/bin/env python3
"""
Test script to verify threat detection functionality
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.real_time_predictor import RealTimePredictor
import pandas as pd
import numpy as np
from datetime import datetime

def test_detection():
    print("🧪 Testing BlackWall Threat Detection System")
    print("=" * 50)
    
    # Initialize predictor
    predictor = RealTimePredictor()
    
    if not predictor.models:
        print("❌ No models loaded - please train models first")
        return False
    
    print("✅ Models loaded successfully")
    print(f"📊 Loaded models: {list(predictor.models.keys())}")
    
    # Test samples - these should trigger threat detection
    test_samples = [
        {
            'name': 'PORT_SCAN_SSH',
            'data': {
                'src_ip': '10.0.0.5', 'dst_ip': '192.168.1.10',
                'src_port': 1234, 'dst_port': 22,  # SSH port
                'protocol': 6, 'packet_size': 60, 'flags': 2,  # SYN flag
                'duration': 0.1, 'flow_duration': 1000,
                'flow_bytes_s': 1000000, 'flow_packets_s': 10000,
                'fwd_packet_length_mean': 1000.0,
                'fwd_packet_length_std': 500.0,
                'bwd_packet_length_mean': 0.0,
                'bwd_packet_length_std': 0.0,
                'flow_iat_mean': 0.1,
                'flow_iat_std': 0.01,
                'fwd_iat_mean': 0.01,
                'fwd_iat_std': 0.001,
                'fwd_packets_s': 10000.0,
                'bwd_packets_s': 0.0,
                'packet_length_mean': 1000.0,
                'packet_length_std': 500.0,
                'packet_length_variance': 250000.0,
                'average_packet_size': 1000.0,
                'avg_fwd_segment_size': 1000.0,
                'avg_bwd_segment_size': 0.0,
                'active_mean': 0.1,
                'active_std': 0.01,
                'idle_mean': 0.0,
                'idle_std': 0.0
            },
            'expected': 'ATTACK'
        },
        {
            'name': 'DDoS_ATTACK',
            'data': {
                'src_ip': '172.16.0.8', 'dst_ip': '192.168.1.15', 
                'src_port': 54321, 'dst_port': 80,  # HTTP flood
                'protocol': 6, 'packet_size': 1500, 'flags': 24,  # PSH-ACK
                'duration': 0.01, 'flow_duration': 500,
                'flow_bytes_s': 5000000, 'flow_packets_s': 5000,
                'fwd_packet_length_mean': 1500.0,
                'fwd_packet_length_std': 100.0,
                'bwd_packet_length_mean': 0.0,
                'bwd_packet_length_std': 0.0,
                'flow_iat_mean': 0.001,
                'flow_iat_std': 0.0001,
                'fwd_iat_mean': 0.001,
                'fwd_iat_std': 0.0001,
                'fwd_packets_s': 5000.0,
                'bwd_packets_s': 0.0,
                'packet_length_mean': 1500.0,
                'packet_length_std': 100.0,
                'packet_length_variance': 10000.0,
                'average_packet_size': 1500.0,
                'avg_fwd_segment_size': 1500.0,
                'avg_bwd_segment_size': 0.0,
                'active_mean': 0.001,
                'active_std': 0.0001,
                'idle_mean': 0.0,
                'idle_std': 0.0
            },
            'expected': 'ATTACK'
        },
        {
            'name': 'NORMAL_WEB_TRAFFIC',
            'data': {
                'src_ip': '192.168.1.100', 'dst_ip': '8.8.8.8',
                'src_port': 54321, 'dst_port': 443,  # HTTPS
                'protocol': 6, 'packet_size': 800, 'flags': 24,
                'duration': 0.5, 'flow_duration': 2000,
                'flow_bytes_s': 50000, 'flow_packets_s': 50,
                'fwd_packet_length_mean': 500.0,
                'fwd_packet_length_std': 100.0,
                'bwd_packet_length_mean': 400.0,
                'bwd_packet_length_std': 80.0,
                'flow_iat_mean': 20.0,
                'flow_iat_std': 5.0,
                'fwd_iat_mean': 25.0,
                'fwd_iat_std': 5.0,
                'fwd_packets_s': 25.0,
                'bwd_packets_s': 20.0,
                'packet_length_mean': 450.0,
                'packet_length_std': 90.0,
                'packet_length_variance': 8100.0,
                'average_packet_size': 450.0,
                'avg_fwd_segment_size': 500.0,
                'avg_bwd_segment_size': 400.0,
                'active_mean': 25.0,
                'active_std': 5.0,
                'idle_mean': 100.0,
                'idle_std': 20.0
            },
            'expected': 'NORMAL'
        },
        {
            'name': 'SUSPICIOUS_BEHAVIOR',
            'data': {
                'src_ip': '10.1.1.50', 'dst_ip': '192.168.1.20',
                'src_port': 9999, 'dst_port': 3389,  # RDP - often targeted
                'protocol': 6, 'packet_size': 200, 'flags': 2,  # SYN
                'duration': 0.05, 'flow_duration': 100,
                'flow_bytes_s': 2000000, 'flow_packets_s': 2000,
                'fwd_packet_length_mean': 800.0,
                'fwd_packet_length_std': 300.0,
                'bwd_packet_length_mean': 50.0,
                'bwd_packet_length_std': 10.0,
                'flow_iat_mean': 0.5,
                'flow_iat_std': 0.1,
                'fwd_iat_mean': 0.5,
                'fwd_iat_std': 0.1,
                'fwd_packets_s': 1000.0,
                'bwd_packets_s': 50.0,
                'packet_length_mean': 600.0,
                'packet_length_std': 250.0,
                'packet_length_variance': 62500.0,
                'average_packet_size': 600.0,
                'avg_fwd_segment_size': 800.0,
                'avg_bwd_segment_size': 50.0,
                'active_mean': 0.5,
                'active_std': 0.1,
                'idle_mean': 10.0,
                'idle_std': 2.0
            },
            'expected': 'ATTACK'
        }
    ]
    
    results = []
    
    for test in test_samples:
        print(f"\n🔍 Testing: {test['name']}")
        print(f"   Source: {test['data']['src_ip']}:{test['data']['src_port']} → {test['data']['dst_ip']}:{test['data']['dst_port']}")
        
        try:
            # Create DataFrame
            df = pd.DataFrame([test['data']])
            
            # Prepare features
            X_processed = predictor.preprocessor.prepare_real_time_features(df)
            
            if len(X_processed) == 0:
                print("   ❌ No features processed")
                results.append((test['name'], 'ERROR', 'No features'))
                continue
            
            print(f"   ✅ Features processed: {X_processed.shape}")
            
            # Test each model
            model_results = {}
            for model_name, model in predictor.models.items():
                try:
                    if hasattr(model, 'predict_proba'):
                        # Classification models
                        prediction = model.predict(X_processed)[0]
                        proba = model.predict_proba(X_processed)[0]
                        confidence = max(proba)
                        label = 'ATTACK' if prediction == 1 else 'NORMAL'
                    else:
                        # Anomaly detection models
                        raw_pred = model.predict(X_processed)[0]
                        is_anomaly = (raw_pred == -1)
                        label = 'ATTACK' if is_anomaly else 'NORMAL'
                        confidence = 0.9 if is_anomaly else 0.1
                    
                    model_results[model_name] = {
                        'label': label,
                        'confidence': confidence,
                        'correct': label == test['expected']
                    }
                    
                    status_icon = "✅" if label == test['expected'] else "❌"
                    print(f"   {status_icon} {model_name}: {label} (conf: {confidence:.2f})")
                    
                except Exception as e:
                    print(f"   ❌ {model_name} error: {e}")
                    model_results[model_name] = {'label': 'ERROR', 'confidence': 0, 'correct': False}
            
            # Determine overall result
            attack_votes = sum(1 for r in model_results.values() if r['label'] == 'ATTACK')
            total_models = len(model_results)
            
            overall_label = 'ATTACK' if attack_votes > total_models / 2 else 'NORMAL'
            overall_correct = overall_label == test['expected']
            
            results.append((test['name'], overall_label, 'PASS' if overall_correct else 'FAIL', model_results))
            
        except Exception as e:
            print(f"   ❌ Test failed: {e}")
            import traceback
            traceback.print_exc()
            results.append((test['name'], 'ERROR', 'FAIL', {}))
    
    # Print summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    
    passed = 0
    for test_name, result, status, model_results in results:
        status_icon = "✅" if status == 'PASS' else "❌"
        print(f"{status_icon} {test_name}: {result} ({status})")
        if status == 'PASS':
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All tests passed! Threat detection is working correctly.")
        return True
    else:
        print("⚠️  Some tests failed. Check your models and preprocessor.")
        return False

def test_preprocessor():
    """Test if preprocessor is working correctly"""
    print("\n" + "=" * 50)
    print("🔧 Testing Preprocessor")
    print("=" * 50)
    
    from utils.preprocessor import DataPreprocessor
    from config import MODELS_DIR
    import joblib
    import os
    
    preprocessor = DataPreprocessor()
    
    # Try to load preprocessor
    preprocessor_path = os.path.join(MODELS_DIR, 'preprocessor.joblib')
    if os.path.exists(preprocessor_path):
        try:
            preprocessor.load_preprocessor(preprocessor_path)
            print("✅ Preprocessor loaded successfully")
            
            # Test with sample data
            test_data = {
                'src_ip': '192.168.1.1',
                'dst_ip': '8.8.8.8', 
                'src_port': 12345,
                'dst_port': 80,
                'protocol': 6,
                'packet_size': 500
            }
            
            df = pd.DataFrame([test_data])
            features = preprocessor.prepare_real_time_features(df)
            
            if len(features) > 0:
                print(f"✅ Feature preparation working: {features.shape}")
            else:
                print("❌ No features generated")
                
        except Exception as e:
            print(f"❌ Preprocessor error: {e}")
    else:
        print("❌ Preprocessor file not found")

if __name__ == "__main__":
    print("🚀 BlackWall Detection System Test")
    print("This test verifies that your ML models can detect threats.")
    print("If tests fail, you may need to retrain your models.\n")
    
    # Test preprocessor first
    test_preprocessor()
    
    # Run main detection test
    success = test_detection()
    
    if success:
        print("\n🎉 SYSTEM READY: Your IDS can detect threats!")
        print("   Next: Start the web interface with 'python main.py'")
    else:
        print("\n⚠️  SYSTEM ISSUES: Some components need attention")
        print("   Check: 1) Models are trained 2) Preprocessor is fitted 3) Feature names match")
    
    sys.exit(0 if success else 1)