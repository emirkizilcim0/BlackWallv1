import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
import json
from datetime import datetime
from collections import Counter
import optuna
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


warnings.filterwarnings('ignore')

from utils.preprocessor import DataPreprocessor
from config import MODELS_DIR, RESULTS_DIR

# ==============================
# CONFIG
# ==============================
N_SHOTS = 10                # number of samples per attack class
USE_PCA = False
PCA_COMPONENTS = 30
RANDOM_STATE = 42
N_TRIALS = 30               # Optuna trials for hyperparameter optimization

PREPROCESSOR_NAME = "few_shot_preprocessor.joblib"
MODEL_NAME = f"few_shot_attack_classifier_{N_SHOTS}shot.joblib"
ENCODER_NAME = f"few_shot_label_encoder_{N_SHOTS}shot.joblib"
RESULTS_FILE = f"few_shot_results_{N_SHOTS}shot.json"

# Create directories
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ==============================
# JSON SERIALIZABLE CONVERTER
# ==============================
def convert_to_serializable(obj):
    """
    Convert numpy/pandas objects to JSON serializable types
    """
    if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict()
    elif isinstance(obj, pd.Series):
        return obj.tolist()
    elif isinstance(obj, datetime):
        return obj.isoformat()
    elif hasattr(obj, 'tolist'):  # For numpy scalars
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_serializable(item) for item in obj)
    else:
        return obj

# ==============================
# ENHANCED FEW-SHOT SAMPLING
# ==============================
def few_shot_sample(df, label_col="Label", n_shots=10, balance_strategy='oversample'):
    """
    Enhanced few-shot sampling with balancing strategies
    """
    print(f"\n🎯 Creating {n_shots}-shot dataset...")
    
    # Identify available classes
    class_counts = df[label_col].value_counts()
    valid_classes = class_counts[class_counts >= n_shots].index.tolist()
    
    if len(valid_classes) < 2:
        print(f"❌ Only {len(valid_classes)} class(es) with ≥{n_shots} samples. Need at least 2.")
        return None
    
    print(f"📊 Available classes with ≥{n_shots} samples: {len(valid_classes)}")
    
    sampled_dfs = []
    
    for label in valid_classes:
        group = df[df[label_col] == label]
        
        if len(group) >= n_shots:
            # Simple random sampling
            sampled = group.sample(n=n_shots, random_state=RANDOM_STATE)
        else:
            # Apply balancing strategy
            if balance_strategy == 'oversample':
                # Oversample minority class
                sampled = group.sample(n=n_shots, replace=True, random_state=RANDOM_STATE)
                print(f"⚠️ {label}: Oversampled from {len(group)} to {n_shots} samples")
            elif balance_strategy == 'undersample':
                # Undersample - use all available
                sampled = group
                print(f"⚠️ {label}: Using all {len(group)} samples (undersampling)")
            else:
                # Skip classes with insufficient samples
                print(f"⚠️ Skipping class '{label}' (only {len(group)} samples)")
                continue
        
        sampled_dfs.append(sampled)
        print(f"✅ {label}: {len(sampled)} samples")
    
    if not sampled_dfs:
        print("❌ No classes had enough samples!")
        return None
    
    few_shot_df = pd.concat(sampled_dfs, ignore_index=True)
    
    print(f"\n📊 Few-shot dataset statistics:")
    print(f"   Total samples: {len(few_shot_df)}")
    print(f"   Classes: {few_shot_df[label_col].nunique()}")
    print(f"   Class distribution:")
    for label, count in few_shot_df[label_col].value_counts().items():
        print(f"     {label}: {count} samples ({count/len(few_shot_df)*100:.1f}%)")
    
    return few_shot_df

# ==============================
# ADVANCED FEATURE ENGINEERING
# ==============================
def create_advanced_features(df, preprocessor):
    """
    Create advanced features for better few-shot learning
    """
    print("\n🔧 Creating advanced features...")
    
    # Basic numerical features
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove label columns
    exclude_cols = ['Label', 'label', 'is_attack', 'attack_type', 'Attack Type']
    feature_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    # Create interaction features
    interaction_features = []
    
    # Common network traffic interactions
    if 'Total Length of Fwd Packets' in feature_cols and 'Total Length of Bwd Packets' in feature_cols:
        df['fwd_bwd_length_ratio'] = df['Total Length of Fwd Packets'] / \
                                     (df['Total Length of Bwd Packets'] + 1e-10)
        interaction_features.append('fwd_bwd_length_ratio')
    
    if 'Flow Bytes/s' in feature_cols and 'Flow Packets/s' in feature_cols:
        df['bytes_per_packet'] = df['Flow Bytes/s'] / (df['Flow Packets/s'] + 1e-10)
        interaction_features.append('bytes_per_packet')
    
    if 'Fwd Packet Length Mean' in feature_cols and 'Bwd Packet Length Mean' in feature_cols:
        df['mean_packet_length_diff'] = df['Fwd Packet Length Mean'] - df['Bwd Packet Length Mean']
        interaction_features.append('mean_packet_length_diff')
    
    # Statistical features
    if 'Flow IAT Mean' in feature_cols and 'Flow IAT Std' in feature_cols:
        df['iat_cov'] = df['Flow IAT Std'] / (df['Flow IAT Mean'] + 1e-10)
        interaction_features.append('iat_cov')
    
    print(f"✅ Created {len(interaction_features)} interaction features")
    
    # Update feature columns
    feature_cols = [col for col in df.columns 
                   if col not in exclude_cols 
                   and np.issubdtype(df[col].dtype, np.number)]
    
    return df, feature_cols

# ==============================
# HYPERPARAMETER OPTIMIZATION WITH OPTUNA
# ==============================
def optimize_hyperparameters(X_train, y_train, model_type='rf', n_trials=30):
    """
    Hyperparameter optimization using Optuna
    """
    print(f"\n🎯 Optimizing {model_type.upper()} hyperparameters with Optuna...")
    
    def objective(trial):
        if model_type == 'rf':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 5, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
                'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
            }
            # Don't include class_weight in params, add it separately
            model = RandomForestClassifier(
                **params,
                class_weight='balanced',
                random_state=RANDOM_STATE,
                n_jobs=-1
            )
        
        elif model_type == 'xgb':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
                'gamma': trial.suggest_float('gamma', 0, 0.5),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),
            }
            model = XGBClassifier(
                **params,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                verbosity=0,
                eval_metric='mlogloss'
            )
        
        elif model_type == 'lgbm':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 20, 100),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),
            }
            model = LGBMClassifier(
                **params,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                verbose=-1
            )
        
        # Cross-validation
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
        scores = []
        
        for train_idx, val_idx in cv.split(X_train, y_train):
            X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
            y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]
            
            model.fit(X_train_fold, y_train_fold)
            y_pred = model.predict(X_val_fold)
            scores.append(accuracy_score(y_val_fold, y_pred))
        
        return np.mean(scores)
    
    study = optuna.create_study(
        direction='maximize', 
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE)
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    print(f"✅ Best trial:")
    print(f"   Score: {study.best_value:.4f}")
    print(f"   Params: {study.best_params}")
    
    # Convert params to serializable format
    serializable_params = convert_to_serializable(study.best_params)
    
    return serializable_params

# ==============================
# ENSEMBLE MODEL TRAINING (FIXED)
# ==============================
def train_ensemble_model(X_train, y_train, best_params=None):
    """
    Train an ensemble model with multiple classifiers
    """
    print("\n🏗️ Training ensemble model...")
    
    # Base models with default parameters
    rf_params = best_params.get('rf', {}) if best_params else {}
    xgb_params = best_params.get('xgb', {}) if best_params else {}
    lgbm_params = best_params.get('lgbm', {}) if best_params else {}
    
    # Remove class_weight from params if present (for RandomForest)
    rf_params = {k: v for k, v in rf_params.items() if k != 'class_weight'}
    
    # Define individual models with fixed parameters
    rf_model = RandomForestClassifier(
        **rf_params,
        n_estimators=rf_params.get('n_estimators', 200),
        max_depth=rf_params.get('max_depth', 15),
        min_samples_split=rf_params.get('min_samples_split', 5),
        min_samples_leaf=rf_params.get('min_samples_leaf', 2),
        class_weight='balanced',
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    
    xgb_model = XGBClassifier(
        **xgb_params,
        n_estimators=xgb_params.get('n_estimators', 100),
        max_depth=xgb_params.get('max_depth', 7),
        learning_rate=xgb_params.get('learning_rate', 0.1),
        random_state=RANDOM_STATE,
        use_label_encoder=False,
        eval_metric='mlogloss',
        n_jobs=-1
    )
    
    lgbm_model = LGBMClassifier(
        **lgbm_params,
        n_estimators=lgbm_params.get('n_estimators', 100),
        max_depth=lgbm_params.get('max_depth', 7),
        learning_rate=lgbm_params.get('learning_rate', 0.1),
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1
    )
    
    # Soft voting ensemble
    ensemble = VotingClassifier(
        estimators=[
            ('rf', rf_model),
            ('xgb', xgb_model),
            ('lgbm', lgbm_model)
        ],
        voting='soft',
        weights=[1, 1, 1]
    )
    
    # Train ensemble
    print("   Training ensemble...")
    ensemble.fit(X_train, y_train)
    
    # Also train individual models for analysis
    individual_models = {}
    
    print("   Training RandomForest...")
    rf_model.fit(X_train, y_train)
    individual_models['rf'] = rf_model
    
    print("   Training XGBoost...")
    xgb_model.fit(X_train, y_train)
    individual_models['xgb'] = xgb_model
    
    print("   Training LightGBM...")
    lgbm_model.fit(X_train, y_train)
    individual_models['lgbm'] = lgbm_model
    
    print("✅ Ensemble model trained")
    
    return ensemble, individual_models

# ==============================
# SIMPLIFIED ENSEMBLE (Alternative if still having issues)
# ==============================
def train_simple_ensemble(X_train, y_train):
    """
    Simplified ensemble without hyperparameter optimization
    """
    print("\n🏗️ Training simple ensemble model...")
    
    # Define individual models with simple parameters
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    
    xgb_model = XGBClassifier(
        n_estimators=100,
        max_depth=7,
        learning_rate=0.1,
        random_state=RANDOM_STATE,
        use_label_encoder=False,
        eval_metric='mlogloss',
        n_jobs=-1
    )
    
    lgbm_model = LGBMClassifier(
        n_estimators=100,
        max_depth=7,
        learning_rate=0.1,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1
    )
    
    # Soft voting ensemble
    ensemble = VotingClassifier(
        estimators=[
            ('rf', rf_model),
            ('xgb', xgb_model),
            ('lgbm', lgbm_model)
        ],
        voting='soft',
        weights=[1, 1, 1]
    )
    
    # Train ensemble
    print("   Training ensemble...")
    ensemble.fit(X_train, y_train)
    
    # Also train individual models
    individual_models = {
        'rf': rf_model,
        'xgb': xgb_model,
        'lgbm': lgbm_model
    }
    
    for name, model in individual_models.items():
        print(f"   Training {name}...")
        model.fit(X_train, y_train)
    
    print("✅ Simple ensemble model trained")
    
    return ensemble, individual_models

# ==============================
# ADVANCED EVALUATION
# ==============================
# ==============================
# ADVANCED EVALUATION (FIXED FEATURE IMPORTANCE)
# ==============================
def evaluate_model(model, X_test, y_test, label_encoder, individual_models=None, feature_names=None):
    """
    Comprehensive model evaluation with fixed feature importance visualization
    """
    print("\n" + "="*70)
    print("📊 COMPREHENSIVE EVALUATION")
    print("="*70)
    
    # Convert y_test to regular Python list if it's numpy
    if isinstance(y_test, np.ndarray):
        y_test_conv = y_test.tolist()
    else:
        y_test_conv = list(y_test)
    
    # Ensemble predictions
    y_pred = model.predict(X_test)
    
    # Basic metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average='weighted'
    )
    
    print(f"📈 Ensemble Model Performance:")
    print(f"   Accuracy:  {accuracy:.4f}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall:    {recall:.4f}")
    print(f"   F1-Score:  {f1:.4f}")
    
    # Individual model performance
    if individual_models:
        print(f"\n🔍 Individual Model Performance:")
        for name, ind_model in individual_models.items():
            y_pred_ind = ind_model.predict(X_test)
            acc_ind = accuracy_score(y_test, y_pred_ind)
            print(f"   {name.upper()}: {acc_ind:.4f}")
    
    # Detailed classification report
    print(f"\n📋 Detailed Classification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=label_encoder.classes_,
            digits=3
        )
    )
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=label_encoder.classes_,
        yticklabels=label_encoder.classes_,
        annot_kws={"size": 10}
    )
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.title(f"Few-Shot Attack Type Confusion Matrix ({N_SHOTS}-shot)", fontsize=14)
    plt.tight_layout()
    
    cm_path = os.path.join(RESULTS_DIR, f"few_shot_confusion_matrix_{N_SHOTS}shot.png")
    plt.savefig(cm_path, dpi=150, bbox_inches='tight')
    print(f"\n💾 Saved confusion matrix: {cm_path}")
    plt.close()
    
    # Feature importance for RandomForest - FIXED VERSION
    if individual_models and 'rf' in individual_models:
        rf_model = individual_models['rf']
        if hasattr(rf_model, 'feature_importances_'):
            # Create feature names if not provided
            if feature_names is None:
                feature_names = [f"Feature {i}" for i in range(X_test.shape[1])]
                print(f"⚠️ Using generic feature names for feature importance")
            elif len(feature_names) != X_test.shape[1]:
                print(f"⚠️ Feature names count ({len(feature_names)}) doesn't match feature count ({X_test.shape[1]})")
                feature_names = [f"Feature {i}" for i in range(X_test.shape[1])]
            
            # Create feature importance DataFrame
            feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': rf_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            # Take top 20 features
            top_features = feature_importance.head(20)
            
            # Create the plot with better formatting
            plt.figure(figsize=(14, 10))
            
            # Create horizontal bar chart (easier to read feature names)
            bars = plt.barh(range(len(top_features)), top_features['importance'])
            plt.yticks(range(len(top_features)), top_features['feature'], fontsize=10)
            plt.xlabel('Feature Importance', fontsize=12)
            plt.title(f'Top 20 Feature Importances - RandomForest ({N_SHOTS}-shot)', fontsize=14, pad=20)
            
            # Add value labels on bars
            for i, (bar, importance) in enumerate(zip(bars, top_features['importance'])):
                width = bar.get_width()
                plt.text(width + 0.001, bar.get_y() + bar.get_height()/2, 
                        f'{importance:.4f}', 
                        ha='left', va='center', fontsize=9)
            
            # Invert y-axis so highest importance is at top
            plt.gca().invert_yaxis()
            
            # Add grid for better readability
            plt.grid(axis='x', alpha=0.3, linestyle='--')
            
            plt.tight_layout()
            
            fi_path = os.path.join(RESULTS_DIR, f"feature_importance_{N_SHOTS}shot.png")
            plt.savefig(fi_path, dpi=150, bbox_inches='tight')
            print(f"💾 Saved feature importance: {fi_path}")
            
            # Also create a text file with all feature importances
            txt_path = os.path.join(RESULTS_DIR, f"feature_importance_{N_SHOTS}shot.txt")
            with open(txt_path, 'w') as f:
                f.write(f"Feature Importances - RandomForest ({N_SHOTS}-shot)\n")
                f.write("="*60 + "\n")
                for idx, row in feature_importance.iterrows():
                    f.write(f"{row['feature']}: {row['importance']:.6f}\n")
            print(f"💾 Saved detailed feature importance text: {txt_path}")
            
            # Print top 10 features in console
            print(f"\n🔝 Top 10 Most Important Features:")
            for i, (_, row) in enumerate(feature_importance.head(10).iterrows(), 1):
                print(f"   {i:2d}. {row['feature']:30s}: {row['importance']:.6f}")
            
            plt.close()
    
    # Convert metrics to serializable types
    accuracy = float(accuracy)
    precision = float(precision)
    recall = float(recall)
    f1 = float(f1)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm.tolist()  # Convert numpy array to list
    }
# ==============================
# DATA PREPARATION PIPELINE
# ==============================
def prepare_data_pipeline(df, preprocessor):
    """
    Complete data preparation pipeline
    """
    print("\n📊 Data preparation pipeline...")
    
    # Only attack samples for attack type classification
    attack_df = df[df["is_attack"] == 1].copy()
    print(f"⚔️ Attack samples for training: {len(attack_df)}")
    print(f"⚔️ Unique attack types: {attack_df['Label'].nunique()}")
    
    # Few-shot sampling with augmentation
    few_shot_df = few_shot_sample(
        attack_df,
        label_col="Label",
        n_shots=N_SHOTS,
        balance_strategy='oversample'
    )
    
    if few_shot_df is None:
        print("❌ Few-shot sampling failed")
        return None, None, None, None, None
    
    # Advanced feature engineering
    few_shot_df, feature_cols = create_advanced_features(few_shot_df, preprocessor)
    
    # Prepare features and labels
    print("\n🔧 Preparing features and labels...")
    
    # Filter only relevant columns
    X = few_shot_df[feature_cols].copy()
    
    # Handle missing values
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())
    
    # Remove constant columns
    constant_cols = [col for col in X.columns if X[col].nunique() <= 1]
    if constant_cols:
        print(f"🗑️ Removing {len(constant_cols)} constant columns")
        X = X.drop(columns=constant_cols)
    
    # Encode labels
    y = few_shot_df['Label'].copy()
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    print(f"✅ Feature shape: {X.shape}")
    print(f"✅ Classes: {len(label_encoder.classes_)}")
    
    # Convert class distribution to serializable format
    class_dist = dict(Counter(y_encoded))
    serializable_dist = {str(k): int(v) for k, v in class_dist.items()}
    print(f"✅ Class distribution: {serializable_dist}")
    
    # Feature scaling
    print("\n⚖️ Scaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Feature selection
    print("\n🎯 Selecting best features...")
    
    # Initial feature selection with ANOVA F-value
    selector = SelectKBest(score_func=f_classif, k=min(50, X_scaled.shape[1]))
    X_selected = selector.fit_transform(X_scaled, y_encoded)
    
    print(f"✅ Selected {X_selected.shape[1]} features using ANOVA F-test")
    
    # Apply PCA if needed
    if USE_PCA:
        print("\n🌀 Applying PCA...")
        n_components = min(PCA_COMPONENTS, X_selected.shape[1])
        pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
        X_processed = pca.fit_transform(X_selected)
        explained_var = pca.explained_variance_ratio_.sum()
        print(f"✅ PCA: {n_components} components ({explained_var:.3f} variance explained)")
    else:
        X_processed = X_selected
        pca = None
    
    return X_processed, y_encoded, label_encoder, scaler, selector, pca

# ==============================
# MAIN TRAINING PIPELINE
# ==============================
# ==============================
# MAIN TRAINING PIPELINE (FIXED)
# ==============================
def main():
    print("🚀 Starting Enhanced Few-Shot Attack Type Training")
    print("="*70)
    print(f"Configuration: {N_SHOTS}-shot learning")
    print(f"Random State: {RANDOM_STATE}")
    print(f"Models Directory: {MODELS_DIR}")
    print(f"Results Directory: {RESULTS_DIR}")
    print("="*70)
    
    # ------------------------------
    # Load data
    # ------------------------------
    preprocessor = DataPreprocessor()
    
    print("📂 Loading CIC-IDS-2017 data...")
    df = preprocessor.load_cic_2017_data(sample_fraction=0.3)
    
    if df is None or df.empty:
        print("❌ Failed to load data")
        return
    
    print(f"📊 Dataset statistics:")
    print(f"   Total samples: {len(df):,}")
    attack_count = int((df['is_attack'] == 1).sum())
    normal_count = int((df['is_attack'] == 0).sum())
    print(f"   Attack samples: {attack_count:,}")
    print(f"   Normal samples: {normal_count:,}")
    
    # ------------------------------
    # Prepare data
    # ------------------------------
    # MODIFIED: Get feature names BEFORE processing
    print("\n📊 Data preparation pipeline...")
    
    # Only attack samples for attack type classification
    attack_df = df[df["is_attack"] == 1].copy()
    print(f"⚔️ Attack samples for training: {len(attack_df)}")
    print(f"⚔️ Unique attack types: {attack_df['Label'].nunique()}")
    
    # Few-shot sampling with augmentation
    few_shot_df = few_shot_sample(
        attack_df,
        label_col="Label",
        n_shots=N_SHOTS,
        balance_strategy='oversample'
    )
    
    if few_shot_df is None:
        print("❌ Few-shot sampling failed")
        return
    
    # Advanced feature engineering
    few_shot_df, feature_cols = create_advanced_features(few_shot_df, preprocessor)
    
    # Prepare features and labels
    print("\n🔧 Preparing features and labels...")
    
    # Filter only relevant columns
    X = few_shot_df[feature_cols].copy()
    
    # Store feature names BEFORE processing
    original_feature_names = X.columns.tolist()
    print(f"📋 Original feature names: {len(original_feature_names)}")
    print(f"📋 First 5 features: {original_feature_names[:5]}")
    
    # Handle missing values
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())
    
    # Remove constant columns
    constant_cols = [col for col in X.columns if X[col].nunique() <= 1]
    if constant_cols:
        print(f"🗑️ Removing {len(constant_cols)} constant columns")
        X = X.drop(columns=constant_cols)
        # Update feature names
        original_feature_names = [col for col in original_feature_names if col not in constant_cols]
    
    # Encode labels
    y = few_shot_df['Label'].copy()
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    print(f"✅ Feature shape: {X.shape}")
    print(f"✅ Classes: {len(label_encoder.classes_)}")
    
    # Convert class distribution to serializable format
    class_dist = dict(Counter(y_encoded))
    serializable_dist = {str(k): int(v) for k, v in class_dist.items()}
    print(f"✅ Class distribution: {serializable_dist}")
    
    # Feature scaling
    print("\n⚖️ Scaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Feature selection
    print("\n🎯 Selecting best features...")
    
    # Initial feature selection with ANOVA F-value
    selector = SelectKBest(score_func=f_classif, k=min(50, X_scaled.shape[1]))
    X_selected = selector.fit_transform(X_scaled, y_encoded)
    
    # Get selected feature indices and names
    if hasattr(selector, 'get_support'):
        selected_indices = selector.get_support(indices=True)
        selected_feature_names = [original_feature_names[i] for i in selected_indices]
        print(f"✅ Selected {len(selected_feature_names)} features using ANOVA F-test")
        print(f"📋 Selected features (first 10): {selected_feature_names[:10]}")
    else:
        selected_feature_names = []
        print(f"✅ Selected {X_selected.shape[1]} features using ANOVA F-test")
    
    # Apply PCA if needed
    if USE_PCA:
        print("\n🌀 Applying PCA...")
        n_components = min(PCA_COMPONENTS, X_selected.shape[1])
        pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
        X_processed = pca.fit_transform(X_selected)
        explained_var = pca.explained_variance_ratio_.sum()
        print(f"✅ PCA: {n_components} components ({explained_var:.3f} variance explained)")
    else:
        X_processed = X_selected
        pca = None
    
    # ------------------------------
    # Train/test split
    # ------------------------------
    print("\n📊 Creating train/test split...")
    
    # Ensure we have at least 2 samples per class in both sets
    min_samples = 2
    valid_classes = []
    
    for class_idx in np.unique(y_encoded):
        class_count = np.sum(y_encoded == class_idx)
        if class_count >= min_samples * 2:
            valid_classes.append(class_idx)
    
    # Filter data
    mask = np.isin(y_encoded, valid_classes)
    X_filtered = X_processed[mask]
    y_filtered = y_encoded[mask]
    
    print(f"✅ Using {len(valid_classes)} classes with ≥{min_samples*2} samples")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_filtered, y_filtered,
        test_size=0.3,
        random_state=RANDOM_STATE,
        stratify=y_filtered
    )
    
    print(f"📊 Dataset split:")
    print(f"   Training samples: {len(X_train)}")
    print(f"   Testing samples: {len(X_test)}")
    print(f"   Training classes: {len(np.unique(y_train))}")
    print(f"   Testing classes: {len(np.unique(y_test))}")
    
    # ------------------------------
    # Handle class imbalance
    # ------------------------------
    print("\n⚖️ Handling class imbalance...")
    
    # Check if we have enough samples for SMOTE
    if len(np.unique(y_train)) > 1 and len(y_train) > 10:
        try:
            # Use different k_neighbors based on sample size
            k_neighbors = min(3, len(y_train) - 1)
            smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=k_neighbors)
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
            
            print(f"✅ After SMOTE:")
            print(f"   Training samples: {len(X_train_resampled)}")
            
            # Convert class distribution to serializable format
            class_dist = dict(Counter(y_train_resampled))
            serializable_dist = {str(k): int(v) for k, v in class_dist.items()}
            print(f"   Class distribution: {serializable_dist}")
            
            X_train, y_train = X_train_resampled, y_train_resampled
        except Exception as e:
            print(f"⚠️ SMOTE failed: {e}. Using original data.")
    
    # ------------------------------
    # Hyperparameter optimization (Optional)
    # ------------------------------
    use_hyperopt = False  # Set to True to enable hyperparameter optimization
    
    best_params = {}
    
    if use_hyperopt and len(X_train) > 20:
        print("\n🎯 Starting hyperparameter optimization...")
        
        try:
            # Optimize RandomForest
            rf_best = optimize_hyperparameters(
                X_train, y_train, 
                model_type='rf', 
                n_trials=min(N_TRIALS, 20)
            )
            best_params['rf'] = rf_best
        except Exception as e:
            print(f"⚠️ RF optimization failed: {e}")
            best_params['rf'] = {}
        
        # Only optimize if we have enough samples
        if len(X_train) > 50:
            try:
                # Optimize XGBoost
                xgb_best = optimize_hyperparameters(
                    X_train, y_train,
                    model_type='xgb',
                    n_trials=min(N_TRIALS, 15)
                )
                best_params['xgb'] = xgb_best
            except Exception as e:
                print(f"⚠️ XGB optimization failed: {e}")
                best_params['xgb'] = {}
            
            try:
                # Optimize LightGBM
                lgbm_best = optimize_hyperparameters(
                    X_train, y_train,
                    model_type='lgbm',
                    n_trials=min(N_TRIALS, 15)
                )
                best_params['lgbm'] = lgbm_best
            except Exception as e:
                print(f"⚠️ LGBM optimization failed: {e}")
                best_params['lgbm'] = {}
    
    # ------------------------------
    # Train model
    # ------------------------------
    print("\n🏗️ Training model...")
    
    # Choose training method based on whether we have optimized parameters
    if use_hyperopt and best_params:
        try:
            ensemble_model, individual_models = train_ensemble_model(
                X_train, y_train, best_params
            )
        except Exception as e:
            print(f"⚠️ Ensemble training failed: {e}")
            print("   Falling back to simple ensemble...")
            ensemble_model, individual_models = train_simple_ensemble(X_train, y_train)
    else:
        # Use simple ensemble
        ensemble_model, individual_models = train_simple_ensemble(X_train, y_train)
    
    # ------------------------------
    # Evaluate model
    # ------------------------------
    # Use selected feature names for feature importance visualization
    evaluation_results = evaluate_model(
        ensemble_model, 
        X_test, 
        y_test, 
        label_encoder, 
        individual_models, 
        selected_feature_names  # Pass the selected feature names
    )
    
    # ------------------------------
    # Save everything
    # ------------------------------
    print("\n💾 Saving models and results...")
    
    # Save ensemble model
    model_path = os.path.join(MODELS_DIR, MODEL_NAME)
    joblib.dump(ensemble_model, model_path)
    print(f"✅ Ensemble model saved: {model_path}")
    
    # Save label encoder
    encoder_path = os.path.join(MODELS_DIR, ENCODER_NAME)
    joblib.dump(label_encoder, encoder_path)
    print(f"✅ Label encoder saved: {encoder_path}")
    
    # Save preprocessor with feature names
    preprocessor_data = {
        'scaler': scaler,
        'feature_selector': selector,
        'pca': pca,
        'original_feature_names': original_feature_names,  # All original features
        'selected_feature_names': selected_feature_names,  # Features after selection
        'timestamp': datetime.now().isoformat(),
        'config': {
            'n_shots': N_SHOTS,
            'use_pca': USE_PCA,
            'pca_components': PCA_COMPONENTS,
            'random_state': RANDOM_STATE
        }
    }
    
    preprocessor_path = os.path.join(MODELS_DIR, PREPROCESSOR_NAME)
    joblib.dump(preprocessor_data, preprocessor_path)
    print(f"✅ Preprocessor saved: {preprocessor_path}")
    print(f"   Original features: {len(original_feature_names)}")
    print(f"   Selected features: {len(selected_feature_names)}")
    
    # Save individual models
    for name, model in individual_models.items():
        ind_path = os.path.join(MODELS_DIR, f"few_shot_{name}_{N_SHOTS}shot.joblib")
        joblib.dump(model, ind_path)
        print(f"✅ {name.upper()} model saved: {ind_path}")
    
    # ------------------------------
    # Prepare results data
    # ------------------------------
    # Convert all data to serializable format
    results_data = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'n_shots': N_SHOTS,
            'use_pca': USE_PCA,
            'pca_components': PCA_COMPONENTS,
            'random_state': RANDOM_STATE,
            'use_hyperopt': use_hyperopt
        },
        'data_statistics': {
            'total_samples': int(len(df)),
            'attack_samples': int((df['is_attack'] == 1).sum()),
            'train_samples': int(len(X_train)),
            'test_samples': int(len(X_test)),
            'n_classes': int(len(label_encoder.classes_)),
            'n_features': int(X_test.shape[1]),
            'n_selected_features': int(len(selected_feature_names))
        },
        'performance': evaluation_results,
        'best_params': convert_to_serializable(best_params),
        'classes': label_encoder.classes_.tolist(),
        'top_features': selected_feature_names[:10]  # Top 10 feature names
    }
    
    # Convert to serializable format
    results_data = convert_to_serializable(results_data)
    
    # Save results
    results_path = os.path.join(RESULTS_DIR, RESULTS_FILE)
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)
    print(f"✅ Results saved: {results_path}")
    
    # ------------------------------
    # Test predictions on sample data
    # ------------------------------
    print("\n" + "="*60)
    print("🧪 TESTING WITH SAMPLE ATTACKS")
    print("="*60)
    
    # Create sample predictions (optional)
    try:
        # Get feature names for reference
        if selected_feature_names:
            print(f"📋 Using {len(selected_feature_names)} selected features")
        
        # Test a few predictions
        sample_idx = min(3, len(X_test) - 1)
        if sample_idx > 0:
            for i in range(sample_idx):
                sample = X_test[i].reshape(1, -1)
                true_label = label_encoder.inverse_transform([y_test[i]])[0]
                pred_label = label_encoder.inverse_transform(ensemble_model.predict(sample))[0]
                
                if hasattr(ensemble_model, 'predict_proba'):
                    proba = ensemble_model.predict_proba(sample)[0]
                    confidence = float(max(proba))
                else:
                    confidence = 1.0
                
                if pred_label == true_label:
                    print(f"✅ Sample {i+1}: CORRECT - {true_label} (confidence: {confidence:.3f})")
                else:
                    print(f"❌ Sample {i+1}: WRONG - Predicted {pred_label}, Expected {true_label} (confidence: {confidence:.3f})")
    except Exception as e:
        print(f"⚠️ Sample testing failed: {e}")
    
    # ------------------------------
    # Final summary
    # ------------------------------
    print("\n" + "="*70)
    print("🎉 FEW-SHOT TRAINING COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"📁 Models saved in: {MODELS_DIR}")
    print(f"📊 Results saved in: {RESULTS_DIR}")
    print(f"🎯 Configuration: {N_SHOTS}-shot learning")
    print(f"📈 Best Accuracy: {evaluation_results['accuracy']:.4f}")
    print(f"📊 F1-Score: {evaluation_results['f1']:.4f}")
    if selected_feature_names:
        print(f"🔝 Top 5 features: {selected_feature_names[:5]}")
    print("="*70)

# ==============================
# ENTRY POINT
# ==============================
if __name__ == "__main__":
    main()