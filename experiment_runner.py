import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
from data_cutter import DataCutter
from hyperparameter_tuner import HyperparameterTuner
from models.trainer import BlackWallTrainer
from utils.preprocessor import DataPreprocessor
from config import EXPERIMENTS_DIR, MODEL_CONFIG

class ExperimentRunner:
    def __init__(self):
        self.data_cutter = DataCutter()
        self.tuner = HyperparameterTuner()
        self.preprocessor = DataPreprocessor()
        self.trainer = BlackWallTrainer()
        
    def run_comprehensive_experiment(self):
        """Run comprehensive experiments with different cuts and parameters"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_dir = os.path.join(EXPERIMENTS_DIR, f"exp_{timestamp}")
        os.makedirs(experiment_dir, exist_ok=True)
        
        print("🚀 Starting Comprehensive BlackWall Experiments")
        print("=" * 60)
        
        # Experiment with different data cuts
        sample_fractions = [0.01, 0.05, 0.1, 0.2]
        strategies = ['random', 'balanced', 'attack_focused']
        
        all_results = []
        
        for fraction in sample_fractions:
            for strategy in strategies:
                print(f"\n{'='*50}")
                print(f"🔬 Experiment: {strategy} strategy, {fraction*100}% data")
                print(f"{'='*50}")
                
                try:
                    # Cut data
                    cut_data = self.data_cutter.load_and_cut_data(strategy, fraction)
                    
                    # Preprocess
                    X, y_binary, y_multiclass = self.preprocessor.prepare_features(
                        cut_data, use_pca=True
                    )
                    
                    # Train with default parameters
                    default_results = self.trainer.train_models(X, y_binary, y_multiclass)
                    
                    # Train with tuned parameters
                    X_train, X_test, y_train, y_test = self._split_data(X, y_binary)
                    tuned_models = self.tuner.tune_all_models(X_train, y_train)
                    
                    # Evaluate tuned models
                    tuned_results = self._evaluate_tuned_models(tuned_models, X_test, y_test)
                    
                    # Store results
                    experiment_result = {
                        'strategy': strategy,
                        'sample_fraction': fraction,
                        'data_size': len(cut_data),
                        'default_results': default_results,
                        'tuned_results': tuned_results,
                        'best_model': self._find_best_model(tuned_results),
                        'timestamp': timestamp
                    }
                    
                    all_results.append(experiment_result)
                    
                    # Save individual experiment
                    self._save_experiment(experiment_result, experiment_dir, 
                                        f"{strategy}_{fraction}")
                    
                    print(f"✅ Completed experiment: {strategy}, {fraction*100}%")
                    
                except Exception as e:
                    print(f"❌ Experiment failed: {e}")
                    continue
        
        # Generate final report
        self._generate_final_report(all_results, experiment_dir)
        
        return all_results
    
    def _split_data(self, X, y):
        """Split data for training and testing"""
        from sklearn.model_selection import train_test_split
        return train_test_split(
            X, y, 
            test_size=MODEL_CONFIG['test_size'],
            random_state=MODEL_CONFIG['random_state'],
            stratify=y
        )
    
    def _evaluate_tuned_models(self, tuned_models, X_test, y_test):
        """Evaluate tuned models on test set"""
        from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
        
        results = {}
        
        for model_name, model_info in tuned_models.items():
            model = model_info['model']
            
            if hasattr(model, 'predict_proba'):
                y_pred = model.predict(X_test)
                y_proba = model.predict_proba(X_test)[:, 1]
                
                results[model_name] = {
                    'accuracy': accuracy_score(y_test, y_pred),
                    'f1_score': f1_score(y_test, y_pred),
                    'roc_auc': roc_auc_score(y_test, y_proba),
                    'tuning_score': model_info['score'],
                    'parameters': model_info['params']
                }
            else:
                # For Isolation Forest
                y_pred = model.predict(X_test)
                y_pred_binary = (y_pred == -1).astype(int)
                
                results[model_name] = {
                    'accuracy': accuracy_score(y_test, y_pred_binary),
                    'f1_score': f1_score(y_test, y_pred_binary),
                    'roc_auc': 0.5,  # Not applicable
                    'tuning_score': model_info['score'],
                    'parameters': model_info['params']
                }
        
        return results
    
    def _find_best_model(self, results):
        """Find the best performing model"""
        best_model = None
        best_score = 0
        
        for model_name, metrics in results.items():
            if metrics['f1_score'] > best_score:
                best_score = metrics['f1_score']
                best_model = model_name
        
        return {'model': best_model, 'f1_score': best_score}
    
    def _save_experiment(self, result, experiment_dir, name):
        """Save individual experiment results"""
        file_path = os.path.join(experiment_dir, f"{name}.json")
        
        # Convert numpy types to Python types for JSON serialization
        serializable_result = self._make_json_serializable(result)
        
        with open(file_path, 'w') as f:
            json.dump(serializable_result, f, indent=2)
    
    def _make_json_serializable(self, obj):
        """Convert numpy types to Python native types for JSON"""
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, 
                          np.int32, np.int64, np.uint8, np.uint16, 
                          np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(v) for v in obj]
        elif isinstance(obj, tuple):
            return tuple(self._make_json_serializable(v) for v in obj)
        else:
            return obj
    
    def _generate_final_report(self, all_results, experiment_dir):
        """Generate a comprehensive final report"""
        report = {
            'experiment_summary': {
                'total_experiments': len(all_results),
                'timestamp': all_results[0]['timestamp'] if all_results else 'N/A'
            },
            'best_performing_experiments': [],
            'recommendations': []
        }
        
        # Find best experiments
        for result in all_results:
            report['best_performing_experiments'].append({
                'strategy': result['strategy'],
                'sample_fraction': result['sample_fraction'],
                'best_model': result['best_model'],
                'data_size': result['data_size']
            })
        
        # Generate recommendations
        if all_results:
            best_exp = max(all_results, 
                         key=lambda x: x['best_model']['f1_score'])
            
            report['recommendations'] = [
                f"Best strategy: {best_exp['strategy']}",
                f"Optimal data fraction: {best_exp['sample_fraction']*100}%",
                f"Recommended model: {best_exp['best_model']['model']}",
                f"Expected F1-score: {best_exp['best_model']['f1_score']:.4f}"
            ]
        
        # Save report
        report_path = os.path.join(experiment_dir, "final_report.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📊 Final report saved to: {report_path}")
        
        # Print summary to console
        print("\n🎯 EXPERIMENT SUMMARY:")
        print("=" * 50)
        for rec in report['recommendations']:
            print(f"👉 {rec}")