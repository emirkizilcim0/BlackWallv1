#!/usr/bin/env python3
"""
Enhanced test script with model performance analysis and overfitting detection
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.real_time_predictor import RealTimePredictor
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

class ModelAnalyzer:
    def __init__(self):
        self.analysis_results = {}
        
    def analyze_model_performance(self, model, model_name, X_test, y_test, X_train=None, y_train=None):
        """Comprehensive model analysis for overfitting/underfitting detection"""
        
        analysis = {
            'name': model_name,
            'test_accuracy': 0,
            'test_f1': 0,
            'train_accuracy': 0,
            'train_f1': 0,
            'overfitting_score': 0,
            'underfitting_score': 0,
            'confidence_analysis': {},
            'confusion_matrix': None,
            'status': 'UNKNOWN',
            'issues': []
        }
        
        try:
            # Test set predictions
            if hasattr(model, 'predict_proba'):
                y_pred_test = model.predict(X_test)
                y_pred_proba_test = model.predict_proba(X_test)[:, 1]
            else:
                # For anomaly detection models
                y_pred_raw = model.predict(X_test)
                y_pred_test = (y_pred_raw == -1).astype(int)
                y_pred_proba_test = np.where(y_pred_test == 1, 0.9, 0.1)
            
            # Calculate test metrics
            analysis['test_accuracy'] = accuracy_score(y_test, y_pred_test)
            analysis['test_f1'] = f1_score(y_test, y_pred_test)
            analysis['confusion_matrix'] = confusion_matrix(y_test, y_pred_test)
            
            # Calculate confidence statistics
            analysis['confidence_analysis'] = {
                'mean_confidence': np.mean(y_pred_proba_test),
                'std_confidence': np.std(y_pred_proba_test),
                'max_confidence': np.max(y_pred_proba_test),
                'min_confidence': np.min(y_pred_proba_test),
                'high_confidence_rate': np.mean(y_pred_proba_test > 0.9),
                'low_confidence_rate': np.mean(y_pred_proba_test < 0.1)
            }
            
            # Training set analysis (if available)
            if X_train is not None and y_train is not None:
                if hasattr(model, 'predict_proba'):
                    y_pred_train = model.predict(X_train)
                    y_pred_proba_train = model.predict_proba(X_train)[:, 1]
                else:
                    y_pred_raw_train = model.predict(X_train)
                    y_pred_train = (y_pred_raw_train == -1).astype(int)
                    y_pred_proba_train = np.where(y_pred_train == 1, 0.9, 0.1)
                
                analysis['train_accuracy'] = accuracy_score(y_train, y_pred_train)
                analysis['train_f1'] = f1_score(y_train, y_pred_train)
                
                # Overfitting/Underfitting analysis
                accuracy_gap = analysis['train_accuracy'] - analysis['test_accuracy']
                f1_gap = analysis['train_f1'] - analysis['test_f1']
                
                analysis['overfitting_score'] = max(accuracy_gap, f1_gap)
                analysis['underfitting_score'] = max(1 - analysis['train_accuracy'], 1 - analysis['test_accuracy'])
                
                # Determine model status
                if analysis['overfitting_score'] > 0.15:
                    analysis['status'] = 'OVERFITTING'
                    analysis['issues'].append(f"High overfitting (gap: {analysis['overfitting_score']:.3f})")
                elif analysis['underfitting_score'] > 0.4:
                    analysis['status'] = 'UNDERFITTING'
                    analysis['issues'].append(f"High underfitting (score: {analysis['underfitting_score']:.3f})")
                elif analysis['test_accuracy'] > 0.95:
                    analysis['status'] = 'EXCELLENT'
                elif analysis['test_accuracy'] > 0.85:
                    analysis['status'] = 'GOOD'
                elif analysis['test_accuracy'] > 0.70:
                    analysis['status'] = 'FAIR'
                else:
                    analysis['status'] = 'POOR'
                    analysis['issues'].append(f"Low test accuracy: {analysis['test_accuracy']:.3f}")
            
            # Confidence-based issues
            conf_analysis = analysis['confidence_analysis']
            if conf_analysis['high_confidence_rate'] > 0.8:
                analysis['issues'].append("Over-confident predictions")
            if conf_analysis['low_confidence_rate'] > 0.8:
                analysis['issues'].append("Under-confident predictions")
            if conf_analysis['std_confidence'] < 0.1:
                analysis['issues'].append("Low confidence variance")
                
        except Exception as e:
            analysis['status'] = 'ERROR'
            analysis['issues'].append(f"Analysis error: {str(e)}")
        
        return analysis

def test_detection_with_analysis():
    print("🧪 Testing BlackWall Threat Detection System with Model Analysis")
    print("=" * 60)
    
    # Initialize predictor and analyzer
    predictor = RealTimePredictor()
    analyzer = ModelAnalyzer()
    
    if not predictor.models:
        print("❌ No models loaded - please train models first")
        return False
    
    print("✅ Models loaded successfully")
    print(f"📊 Loaded models: {list(predictor.models.keys())}")
    
    # Test samples
    test_samples = [
        {
            'name': 'CIC_PortScan',
            'data': {
                'Fwd Packet Length Mean': 78.0, 'Fwd Packet Length Std': 45.0,
                'Bwd Packet Length Mean': 52.0, 'Bwd Packet Length Std': 38.0,
                'Flow Bytes/s': 85000.0, 'Flow Packets/s': 850.0,
                'Flow IAT Mean': 0.8, 'Flow IAT Std': 1.2,
                'Fwd IAT Mean': 0.9, 'Fwd IAT Std': 1.3,
                'Bwd IAT Mean': 0.7, 'Bwd IAT Std': 1.1,
                'Fwd Packets/s': 480.0, 'Bwd Packets/s': 370.0,
                'Packet Length Mean': 65.0, 'Packet Length Std': 42.0,
                'Packet Length Variance': 1764.0, 'Average Packet Size': 65.0,
                'Avg Fwd Segment Size': 78.0, 'Avg Bwd Segment Size': 52.0,
                'Active Mean': 0.8, 'Active Std': 1.1,
                'Idle Mean': 2.5, 'Idle Std': 3.8,
                'src_ip': '10.0.0.5', 'dst_ip': '192.168.1.10',
                'src_port': 1234, 'dst_port': 22, 'protocol': 6, 'packet_size': 78
            },
            'expected': 1  # ATTACK
        },
        {
            'name': 'CIC_DDoS',
            'data': {
                'Fwd Packet Length Mean': 95.0, 'Fwd Packet Length Std': 120.0,
                'Bwd Packet Length Mean': 0.0, 'Bwd Packet Length Std': 0.0,
                'Flow Bytes/s': 450000.0, 'Flow Packets/s': 3200.0,
                'Flow IAT Mean': 0.2, 'Flow IAT Std': 0.4,
                'Fwd IAT Mean': 0.2, 'Fwd IAT Std': 0.4,
                'Bwd IAT Mean': 0.0, 'Bwd IAT Std': 0.0,
                'Fwd Packets/s': 3200.0, 'Bwd Packets/s': 0.0,
                'Packet Length Mean': 95.0, 'Packet Length Std': 120.0,
                'Packet Length Variance': 14400.0, 'Average Packet Size': 95.0,
                'Avg Fwd Segment Size': 95.0, 'Avg Bwd Segment Size': 0.0,
                'Active Mean': 0.2, 'Active Std': 0.3,
                'Idle Mean': 0.8, 'Idle Std': 1.2,
                'src_ip': '172.16.0.8', 'dst_ip': '192.168.1.15',
                'src_port': 54321, 'dst_port': 80, 'protocol': 6, 'packet_size': 95
            },
            'expected': 1  # ATTACK
        },
        {
            'name': 'NORMAL_WEB_TRAFFIC',
            'data': {
                'Fwd Packet Length Mean': 450.0, 'Fwd Packet Length Std': 120.0,
                'Bwd Packet Length Mean': 380.0, 'Bwd Packet Length Std': 90.0,
                'Flow Bytes/s': 25000.0, 'Flow Packets/s': 45.0,
                'Flow IAT Mean': 15.0, 'Flow IAT Std': 8.0,
                'Fwd IAT Mean': 18.0, 'Fwd IAT Std': 9.0,
                'Bwd IAT Mean': 12.0, 'Bwd IAT Std': 7.0,
                'Fwd Packets/s': 25.0, 'Bwd Packets/s': 20.0,
                'Packet Length Mean': 415.0, 'Packet Length Std': 105.0,
                'Packet Length Variance': 11025.0, 'Average Packet Size': 415.0,
                'Avg Fwd Segment Size': 450.0, 'Avg Bwd Segment Size': 380.0,
                'Active Mean': 15.0, 'Active Std': 8.0,
                'Idle Mean': 30.0, 'Idle Std': 15.0,
                'src_ip': '192.168.1.100', 'dst_ip': '8.8.8.8',
                'src_port': 54321, 'dst_port': 443, 'protocol': 6, 'packet_size': 450
            },
            'expected': 0  # NORMAL
        }
    ]
    
    # Prepare test data - FIXED: Handle numpy array return
    test_data = []
    test_labels = []
    
    for test in test_samples:
        df = pd.DataFrame([test['data']])
        X_processed = predictor.preprocessor.prepare_real_time_features(df)
        
        if X_processed is not None and len(X_processed) > 0:
            # Handle both DataFrame and numpy array returns
            if isinstance(X_processed, pd.DataFrame):
                test_data.append(X_processed.iloc[0].values)
            else:  # numpy array
                test_data.append(X_processed[0])
            test_labels.append(test['expected'])
        else:
            print(f"❌ No features processed for {test['name']}")
    
    if not test_data:
        print("❌ No test data could be processed")
        return False
    
    X_test = np.array(test_data)
    y_test = np.array(test_labels)
    
    print(f"📊 Test dataset: {X_test.shape[0]} samples, {X_test.shape[1]} features")
    
    # Analyze each model
    print("\n" + "=" * 60)
    print("🔍 MODEL PERFORMANCE ANALYSIS")
    print("=" * 60)
    
    all_analyses = {}
    
    for model_name, model in predictor.models.items():
        print(f"\n📈 Analyzing {model_name}...")
        
        # For this demo, we don't have training data, so we'll skip train/test comparison
        analysis = analyzer.analyze_model_performance(
            model, model_name, X_test, y_test
        )
        
        all_analyses[model_name] = analysis
        
        # Display analysis results
        status_icon = {
            'EXCELLENT': '🏆', 'GOOD': '✅', 'FAIR': '⚠️', 
            'POOR': '❌', 'OVERFITTING': '🎯', 'UNDERFITTING': '📉', 'ERROR': '💥'
        }.get(analysis['status'], '❓')
        
        print(f"   {status_icon} Status: {analysis['status']}")
        print(f"   📊 Test Accuracy: {analysis['test_accuracy']:.3f}")
        print(f"   ⚡ Test F1-Score: {analysis['test_f1']:.3f}")
        
        if analysis['issues']:
            print(f"   🚨 Issues: {', '.join(analysis['issues'])}")
        
        # Confidence analysis
        conf = analysis['confidence_analysis']
        print(f"   🎯 Confidence - Mean: {conf['mean_confidence']:.3f}, "
              f"Std: {conf['std_confidence']:.3f}")
    
    # Overall assessment
    print("\n" + "=" * 60)
    print("🏆 OVERALL MODEL ASSESSMENT")
    print("=" * 60)
    
    # Count models by status
    status_count = {}
    for analysis in all_analyses.values():
        status = analysis['status']
        status_count[status] = status_count.get(status, 0) + 1
    
    for status, count in status_count.items():
        icon = {'EXCELLENT': '🏆', 'GOOD': '✅', 'FAIR': '⚠️', 'POOR': '❌', 
                'OVERFITTING': '🎯', 'UNDERFITTING': '📉', 'ERROR': '💥'}.get(status, '❓')
        print(f"{icon} {status}: {count} model(s)")
    
    # Individual test results
    print("\n" + "=" * 60)
    print("🧪 INDIVIDUAL TEST RESULTS")
    print("=" * 60)
    
    results = []
    for test in test_samples:
        print(f"\n🔍 Testing: {test['name']}")
        print(f"   Expected: {'ATTACK' if test['expected'] == 1 else 'NORMAL'}")
        
        df = pd.DataFrame([test['data']])
        X_processed = predictor.preprocessor.prepare_real_time_features(df)
        
        if X_processed is None or len(X_processed) == 0:
            print("   ❌ No features processed")
            continue
        
        # Test each model
        model_votes = {'ATTACK': 0, 'NORMAL': 0}
        model_details = {}
        
        for model_name, model in predictor.models.items():
            try:
                if hasattr(model, 'predict_proba'):
                    prediction = model.predict(X_processed)[0]
                    proba = model.predict_proba(X_processed)[0]
                    confidence = max(proba)
                    label = 'ATTACK' if prediction == 1 else 'NORMAL'
                else:
                    raw_pred = model.predict(X_processed)[0]
                    is_anomaly = (raw_pred == -1)
                    label = 'ATTACK' if is_anomaly else 'NORMAL'
                    confidence = 0.9 if is_anomaly else 0.1
                
                model_votes[label] += 1
                model_details[model_name] = {
                    'label': label, 
                    'confidence': confidence,
                    'correct': label == ('ATTACK' if test['expected'] == 1 else 'NORMAL')
                }
                
                status_icon = "✅" if model_details[model_name]['correct'] else "❌"
                print(f"   {status_icon} {model_name}: {label} (conf: {confidence:.2f})")
                
            except Exception as e:
                print(f"   ❌ {model_name} error: {e}")
        
        # Overall result
        overall_label = 'ATTACK' if model_votes['ATTACK'] > model_votes['NORMAL'] else 'NORMAL'
        overall_correct = overall_label == ('ATTACK' if test['expected'] == 1 else 'NORMAL')
        
        results.append((test['name'], overall_label, overall_correct))
        
        status_icon = "✅" if overall_correct else "❌"
        print(f"   {status_icon} OVERALL: {overall_label} "
              f"(Votes: ATTACK {model_votes['ATTACK']}, NORMAL {model_votes['NORMAL']})")
    
    # Final summary
    print("\n" + "=" * 60)
    print("📊 FINAL SUMMARY")
    print("=" * 60)
    
    passed_tests = sum(1 for _, _, correct in results if correct)
    total_tests = len(results)
    
    print(f"🎯 Test Results: {passed_tests}/{total_tests} passed")
    
    # Model health assessment
    excellent_models = sum(1 for a in all_analyses.values() if a['status'] in ['EXCELLENT', 'GOOD'])
    problem_models = sum(1 for a in all_analyses.values() if a['status'] in ['POOR', 'OVERFITTING', 'UNDERFITTING', 'ERROR'])
    
    print(f"🔧 Model Health: {excellent_models} healthy, {problem_models} problematic out of {len(all_analyses)} total")
    
    if passed_tests == total_tests and problem_models == 0:
        print("🎉 EXCELLENT: All tests passed and models are healthy!")
        return True
    elif passed_tests >= total_tests * 0.7 and problem_models <= len(all_analyses) * 0.3:
        print("✅ GOOD: Most tests passed and models are generally healthy")
        return True
    else:
        print("⚠️  NEEDS ATTENTION: Significant issues detected")
        return False

if __name__ == "__main__":
    success = test_detection_with_analysis()
    sys.exit(0 if success else 1)