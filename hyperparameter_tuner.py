import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os
from config import TUNING_CONFIG, HYPERPARAMETERS, EXPERIMENTS_DIR

class HyperparameterTuner:
    def __init__(self):
        self.config = TUNING_CONFIG
        self.hyperparameters = HYPERPARAMETERS
        
    def tune_logistic_regression(self, X_train, y_train):
        """Tune Logistic Regression hyperparameters"""
        from sklearn.linear_model import LogisticRegression
        
        print("🎯 Tuning Logistic Regression...")
        lr = LogisticRegression(random_state=42, class_weight='balanced')
        
        search = RandomizedSearchCV(
            lr, self.hyperparameters['logistic_regression'],
            n_iter=self.config['n_iter'],
            cv=self.config['cv_folds'],
            scoring=self.config['scoring'],
            random_state=42,
            n_jobs=-1
        )
        
        search.fit(X_train, y_train)
        print(f"✅ Best parameters: {search.best_params_}")
        print(f"✅ Best score: {search.best_score_:.4f}")
        
        return search.best_estimator_, search.best_params_, search.best_score_
    
    def tune_gaussian_nb(self, X_train, y_train):
        """Tune Gaussian Naive Bayes hyperparameters"""
        from sklearn.naive_bayes import GaussianNB
        
        print("🎯 Tuning Gaussian Naive Bayes...")
        gnb = GaussianNB()
        
        search = RandomizedSearchCV(
            gnb, self.hyperparameters['gaussian_nb'],
            n_iter=self.config['n_iter'],
            cv=self.config['cv_folds'],
            scoring=self.config['scoring'],
            random_state=42,
            n_jobs=-1
        )
        
        search.fit(X_train, y_train)
        print(f"✅ Best parameters: {search.best_params_}")
        print(f"✅ Best score: {search.best_score_:.4f}")
        
        return search.best_estimator_, search.best_params_, search.best_score_
    
    def tune_random_forest(self, X_train, y_train):
        """Tune Random Forest hyperparameters"""
        from sklearn.ensemble import RandomForestClassifier
        
        print("🎯 Tuning Random Forest...")
        rf = RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=-1)
        
        search = RandomizedSearchCV(
            rf, self.hyperparameters['random_forest'],
            n_iter=self.config['n_iter'],
            cv=self.config['cv_folds'],
            scoring=self.config['scoring'],
            random_state=42,
            n_jobs=-1
        )
        
        search.fit(X_train, y_train)
        print(f"✅ Best parameters: {search.best_params_}")
        print(f"✅ Best score: {search.best_score_:.4f}")
        
        return search.best_estimator_, search.best_params_, search.best_score_
    
    def tune_isolation_forest(self, X_train, y_train):
        """Tune Isolation Forest hyperparameters"""
        from sklearn.ensemble import IsolationForest
        
        print("🎯 Tuning Isolation Forest...")
        iso = IsolationForest(random_state=42, n_jobs=-1)
        
        search = RandomizedSearchCV(
            iso, self.hyperparameters['isolation_forest'],
            n_iter=self.config['n_iter'],
            cv=self.config['cv_folds'],
            scoring=self.config['scoring'],
            random_state=42,
            n_jobs=-1
        )
        
        # Convert labels for anomaly detection (1 for normal, -1 for anomaly)
        y_train_iso = np.where(y_train == 0, 1, -1)
        
        search.fit(X_train, y_train_iso)
        print(f"✅ Best parameters: {search.best_params_}")
        print(f"✅ Best score: {search.best_score_:.4f}")
        
        return search.best_estimator_, search.best_params_, search.best_score_
    
    def tune_all_models(self, X_train, y_train):
        """Tune hyperparameters for all models"""
        tuned_models = {}
        
        print("🚀 Starting hyperparameter tuning for all models...")
        
        # Tune each model
        lr_model, lr_params, lr_score = self.tune_logistic_regression(X_train, y_train)
        tuned_models['logistic_regression'] = {
            'model': lr_model, 'params': lr_params, 'score': lr_score
        }
        
        nb_model, nb_params, nb_score = self.tune_gaussian_nb(X_train, y_train)
        tuned_models['gaussian_nb'] = {
            'model': nb_model, 'params': nb_params, 'score': nb_score
        }
        
        rf_model, rf_params, rf_score = self.tune_random_forest(X_train, y_train)
        tuned_models['random_forest'] = {
            'model': rf_model, 'params': rf_params, 'score': rf_score
        }
        
        iso_model, iso_params, iso_score = self.tune_isolation_forest(X_train, y_train)
        tuned_models['isolation_forest'] = {
            'model': iso_model, 'params': iso_params, 'score': iso_score
        }
        
        # Print summary
        print("\n📊 Tuning Results Summary:")
        print("=" * 50)
        for model_name, results in tuned_models.items():
            print(f"{model_name:.<20} Score: {results['score']:.4f}")
        
        return tuned_models