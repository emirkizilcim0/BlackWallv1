import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score, 
                           confusion_matrix, classification_report,
                           roc_curve)
from sklearn.model_selection import train_test_split
from utils.preprocessor import DataPreprocessor
from config import CIC_DATA_DIR, MODELS_DIR, RESULTS_DIR, MODEL_CONFIG
import warnings
warnings.filterwarnings('ignore')
# Import models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier


# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def create_balanced_attack_dataset():
    """Create a dataset with clear but realistic separation between normal and attacks"""
    print("🛠️ Creating BALANCED attack dataset...")
    
    np.random.seed(42)
    synthetic_data = []
    
    # NORMAL TRAFFIC PATTERNS (Class 0) - More realistic and varied
    print("📊 Generating realistic normal traffic patterns...")
    for i in range(6000):  # More normal samples
        normal_sample = {
            'Fwd Packet Length Mean': max(1, np.random.normal(450, 150)),
            'Fwd Packet Length Std': max(1, np.random.normal(180, 80)),
            'Bwd Packet Length Mean': max(1, np.random.normal(350, 120)),
            'Bwd Packet Length Std': max(1, np.random.normal(150, 60)),
            'Flow Bytes/s': max(1000, np.random.normal(120000, 80000)),
            'Flow Packets/s': max(1, np.random.normal(150, 80)),
            'Flow IAT Mean': max(0.001, np.random.normal(1.2, 0.8)),
            'Flow IAT Std': max(0.001, np.random.normal(1.8, 1.0)),
            'Fwd IAT Mean': max(0.001, np.random.normal(1.0, 0.6)),
            'Fwd IAT Std': max(0.001, np.random.normal(1.5, 0.8)),
            'Bwd IAT Mean': max(0.001, np.random.normal(1.4, 0.9)),
            'Bwd IAT Std': max(0.001, np.random.normal(2.0, 1.2)),
            'Fwd Packets/s': max(1, np.random.normal(80, 40)),
            'Bwd Packets/s': max(1, np.random.normal(70, 35)),
            'Packet Length Mean': max(1, np.random.normal(400, 120)),
            'Packet Length Std': max(1, np.random.normal(170, 70)),
            'Packet Length Variance': max(1, np.random.normal(30000, 15000)),
            'Average Packet Size': max(1, np.random.normal(410, 100)),
            'Avg Fwd Segment Size': max(1, np.random.normal(450, 150)),
            'Avg Bwd Segment Size': max(1, np.random.normal(350, 120)),
            'Active Mean': max(0.001, np.random.normal(0.8, 0.5)),
            'Active Std': max(0.001, np.random.normal(1.2, 0.7)),
            'Idle Mean': max(0.001, np.random.normal(1.8, 1.0)),
            'Idle Std': max(0.001, np.random.normal(2.5, 1.5)),
            'Label': 'BENIGN',
            'Label_Binary': 0
        }
        synthetic_data.append(normal_sample)
    
    # DDoS ATTACK PATTERNS (Class 1)
    print("🔥 Generating DDoS attack patterns...")
    for i in range(2000):
        ddos_sample = {
            'Fwd Packet Length Mean': max(1, np.random.normal(1500, 200)),
            'Fwd Packet Length Std': max(1, np.random.normal(50, 15)),
            'Bwd Packet Length Mean': max(1, np.random.normal(10, 5)),
            'Bwd Packet Length Std': max(1, np.random.normal(5, 2)),
            'Flow Bytes/s': max(1000, np.random.normal(2000000, 500000)),
            'Flow Packets/s': max(1, np.random.normal(5000, 1000)),
            'Flow IAT Mean': max(0.001, np.random.normal(0.001, 0.0005)),
            'Flow IAT Std': max(0.001, np.random.normal(0.0005, 0.0002)),
            'Fwd IAT Mean': max(0.001, np.random.normal(0.001, 0.0005)),
            'Fwd IAT Std': max(0.001, np.random.normal(0.0005, 0.0002)),
            'Bwd IAT Mean': max(0.001, np.random.normal(0.0, 0.0)),
            'Bwd IAT Std': max(0.001, np.random.normal(0.0, 0.0)),
            'Fwd Packets/s': max(1, np.random.normal(5000, 1000)),
            'Bwd Packets/s': max(1, np.random.normal(5, 3)),
            'Packet Length Mean': max(1, np.random.normal(1400, 200)),
            'Packet Length Std': max(1, np.random.normal(50, 15)),
            'Packet Length Variance': max(1, np.random.normal(2500, 500)),
            'Average Packet Size': max(1, np.random.normal(1400, 200)),
            'Avg Fwd Segment Size': max(1, np.random.normal(1500, 200)),
            'Avg Bwd Segment Size': max(1, np.random.normal(10, 5)),
            'Active Mean': max(0.001, np.random.normal(0.001, 0.0005)),
            'Active Std': max(0.001, np.random.normal(0.0005, 0.0002)),
            'Idle Mean': max(0.001, np.random.normal(0.0, 0.0)),
            'Idle Std': max(0.001, np.random.normal(0.0, 0.0)),
            'Label': 'DDoS',
            'Label_Binary': 1
        }
        synthetic_data.append(ddos_sample)
    
    # PORTSCAN ATTACK PATTERNS (Class 1)
    print("🔍 Generating PortScan attack patterns...")
    for i in range(2000):
        portscan_sample = {
            'Fwd Packet Length Mean': max(1, np.random.normal(60, 10)),
            'Fwd Packet Length Std': max(1, np.random.normal(5, 2)),
            'Bwd Packet Length Mean': max(1, np.random.normal(0, 0)),
            'Bwd Packet Length Std': max(1, np.random.normal(0, 0)),
            'Flow Bytes/s': max(1000, np.random.normal(50000, 20000)),
            'Flow Packets/s': max(1, np.random.normal(1500, 400)),
            'Flow IAT Mean': max(0.001, np.random.normal(0.005, 0.002)),
            'Flow IAT Std': max(0.001, np.random.normal(0.002, 0.001)),
            'Fwd IAT Mean': max(0.001, np.random.normal(0.005, 0.002)),
            'Fwd IAT Std': max(0.001, np.random.normal(0.002, 0.001)),
            'Bwd IAT Mean': max(0.001, np.random.normal(0.0, 0.0)),
            'Bwd IAT Std': max(0.001, np.random.normal(0.0, 0.0)),
            'Fwd Packets/s': max(1, np.random.normal(1500, 400)),
            'Bwd Packets/s': max(1, np.random.normal(0, 0)),
            'Packet Length Mean': max(1, np.random.normal(60, 10)),
            'Packet Length Std': max(1, np.random.normal(5, 2)),
            'Packet Length Variance': max(1, np.random.normal(100, 25)),
            'Average Packet Size': max(1, np.random.normal(60, 10)),
            'Avg Fwd Segment Size': max(1, np.random.normal(60, 10)),
            'Avg Bwd Segment Size': max(1, np.random.normal(0, 0)),
            'Active Mean': max(0.001, np.random.normal(0.005, 0.002)),
            'Active Std': max(0.001, np.random.normal(0.002, 0.001)),
            'Idle Mean': max(0.001, np.random.normal(0.0, 0.0)),
            'Idle Std': max(0.001, np.random.normal(0.0, 0.0)),
            'Label': 'PortScan',
            'Label_Binary': 1
        }
        synthetic_data.append(portscan_sample)
    
    # BRUTEFORCE ATTACK PATTERNS (Class 1)
    print("💥 Generating BruteForce attack patterns...")
    for i in range(2000):
        brute_sample = {
            'Fwd Packet Length Mean': max(1, np.random.normal(70, 10)),
            'Fwd Packet Length Std': max(1, np.random.normal(3, 1)),
            'Bwd Packet Length Mean': max(1, np.random.normal(60, 8)),
            'Bwd Packet Length Std': max(1, np.random.normal(2, 1)),
            'Flow Bytes/s': max(1000, np.random.normal(500000, 200000)),
            'Flow Packets/s': max(1, np.random.normal(3000, 800)),
            'Flow IAT Mean': max(0.001, np.random.normal(0.0001, 0.00005)),
            'Flow IAT Std': max(0.001, np.random.normal(0.00005, 0.00002)),
            'Fwd IAT Mean': max(0.001, np.random.normal(0.0001, 0.00005)),
            'Fwd IAT Std': max(0.001, np.random.normal(0.00005, 0.00002)),
            'Bwd IAT Mean': max(0.001, np.random.normal(0.0001, 0.00005)),
            'Bwd IAT Std': max(0.001, np.random.normal(0.00005, 0.00002)),
            'Fwd Packets/s': max(1, np.random.normal(1500, 400)),
            'Bwd Packets/s': max(1, np.random.normal(1500, 400)),
            'Packet Length Mean': max(1, np.random.normal(65, 8)),
            'Packet Length Std': max(1, np.random.normal(4, 1)),
            'Packet Length Variance': max(1, np.random.normal(16, 4)),
            'Average Packet Size': max(1, np.random.normal(65, 8)),
            'Avg Fwd Segment Size': max(1, np.random.normal(70, 10)),
            'Avg Bwd Segment Size': max(1, np.random.normal(60, 8)),
            'Active Mean': max(0.001, np.random.normal(0.0001, 0.00005)),
            'Active Std': max(0.001, np.random.normal(0.00005, 0.00002)),
            'Idle Mean': max(0.001, np.random.normal(0.0, 0.0)),
            'Idle Std': max(0.001, np.random.normal(0.0, 0.0)),
            'Label': 'BruteForce',
            'Label_Binary': 1
        }
        synthetic_data.append(brute_sample)

    # ADVANCED PERSISTENT THREAT PATTERNS
    print("🕵️ Generating Advanced Persistent Threat patterns...")
    for i in range(1500):
        apt_sample = {
            'Fwd Packet Length Mean': max(1, np.random.normal(245, 40)),
            'Fwd Packet Length Std': max(1, np.random.normal(35, 8)),
            'Bwd Packet Length Mean': max(1, np.random.normal(220, 35)),
            'Bwd Packet Length Std': max(1, np.random.normal(28, 6)),
            'Flow Bytes/s': max(1000, np.random.normal(45000, 15000)),
            'Flow Packets/s': max(1, np.random.normal(85, 25)),
            'Flow IAT Mean': max(0.001, np.random.normal(12.5, 3.0)),
            'Flow IAT Std': max(0.001, np.random.normal(8.2, 2.0)),
            'Fwd IAT Mean': max(0.001, np.random.normal(13.0, 3.2)),
            'Fwd IAT Std': max(0.001, np.random.normal(8.5, 2.1)),
            'Bwd IAT Mean': max(0.001, np.random.normal(12.0, 2.8)),
            'Bwd IAT Std': max(0.001, np.random.normal(7.8, 1.9)),
            'Fwd Packets/s': max(1, np.random.normal(45, 12)),
            'Bwd Packets/s': max(1, np.random.normal(40, 10)),
            'Packet Length Mean': max(1, np.random.normal(232.5, 30)),
            'Packet Length Std': max(1, np.random.normal(32.0, 7)),
            'Packet Length Variance': max(1, np.random.normal(1024, 300)),
            'Average Packet Size': max(1, np.random.normal(232.5, 30)),
            'Avg Fwd Segment Size': max(1, np.random.normal(245, 40)),
            'Avg Bwd Segment Size': max(1, np.random.normal(220, 35)),
            'Active Mean': max(0.001, np.random.normal(12.5, 3.0)),
            'Active Std': max(0.001, np.random.normal(8.0, 2.0)),
            'Idle Mean': max(0.001, np.random.normal(45.0, 15.0)),
            'Idle Std': max(0.001, np.random.normal(25.0, 8.0)),
            'Label': 'APT',
            'Label_Binary': 1
        }
        synthetic_data.append(apt_sample)

    # DATA EXFILTRATION PATTERNS
    print("📤 Generating Data Exfiltration patterns...")
    for i in range(1500):
        exfil_sample = {
            'Fwd Packet Length Mean': max(1, np.random.normal(1450, 100)),
            'Fwd Packet Length Std': max(1, np.random.normal(45, 10)),
            'Bwd Packet Length Mean': max(1, np.random.normal(52, 15)),
            'Bwd Packet Length Std': max(1, np.random.normal(8, 2)),
            'Flow Bytes/s': max(1000, np.random.normal(75000, 20000)),
            'Flow Packets/s': max(1, np.random.normal(52, 15)),
            'Flow IAT Mean': max(0.001, np.random.normal(19.3, 3.0)),
            'Flow IAT Std': max(0.001, np.random.normal(2.1, 0.5)),
            'Fwd IAT Mean': max(0.001, np.random.normal(19.5, 3.1)),
            'Fwd IAT Std': max(0.001, np.random.normal(2.0, 0.4)),
            'Bwd IAT Mean': max(0.001, np.random.normal(19.0, 2.9)),
            'Bwd IAT Std': max(0.001, np.random.normal(2.2, 0.5)),
            'Fwd Packets/s': max(1, np.random.normal(26, 8)),
            'Bwd Packets/s': max(1, np.random.normal(26, 8)),
            'Packet Length Mean': max(1, np.random.normal(751, 200)),
            'Packet Length Std': max(1, np.random.normal(699, 150)),
            'Packet Length Variance': max(1, np.random.normal(488601, 100000)),
            'Average Packet Size': max(1, np.random.normal(751, 200)),
            'Avg Fwd Segment Size': max(1, np.random.normal(1450, 100)),
            'Avg Bwd Segment Size': max(1, np.random.normal(52, 15)),
            'Active Mean': max(0.001, np.random.normal(19.3, 3.0)),
            'Active Std': max(0.001, np.random.normal(2.0, 0.4)),
            'Idle Mean': max(0.001, np.random.normal(60.0, 20.0)),
            'Idle Std': max(0.001, np.random.normal(5.0, 1.5)),
            'Label': 'DataExfiltration',
            'Label_Binary': 1
        }
        synthetic_data.append(exfil_sample)

    # DNS TUNNELING PATTERNS
    print("🌐 Generating DNS Tunneling patterns...")
    for i in range(1500):
        dns_sample = {
            'Fwd Packet Length Mean': max(1, np.random.normal(95, 20)),
            'Fwd Packet Length Std': max(1, np.random.normal(12, 3)),
            'Bwd Packet Length Mean': max(1, np.random.normal(255, 60)),
            'Bwd Packet Length Std': max(1, np.random.normal(45, 12)),
            'Flow Bytes/s': max(1000, np.random.normal(12000, 5000)),
            'Flow Packets/s': max(1, np.random.normal(35, 12)),
            'Flow IAT Mean': max(0.001, np.random.normal(28.5, 8.0)),
            'Flow IAT Std': max(0.001, np.random.normal(15.2, 4.0)),
            'Fwd IAT Mean': max(0.001, np.random.normal(29.0, 8.2)),
            'Fwd IAT Std': max(0.001, np.random.normal(15.5, 4.1)),
            'Bwd IAT Mean': max(0.001, np.random.normal(28.0, 7.8)),
            'Bwd IAT Std': max(0.001, np.random.normal(14.8, 3.9)),
            'Fwd Packets/s': max(1, np.random.normal(18, 6)),
            'Bwd Packets/s': max(1, np.random.normal(17, 6)),
            'Packet Length Mean': max(1, np.random.normal(175, 50)),
            'Packet Length Std': max(1, np.random.normal(80, 20)),
            'Packet Length Variance': max(1, np.random.normal(6400, 2000)),
            'Average Packet Size': max(1, np.random.normal(175, 50)),
            'Avg Fwd Segment Size': max(1, np.random.normal(95, 20)),
            'Avg Bwd Segment Size': max(1, np.random.normal(255, 60)),
            'Active Mean': max(0.001, np.random.normal(28.5, 8.0)),
            'Active Std': max(0.001, np.random.normal(15.0, 4.0)),
            'Idle Mean': max(0.001, np.random.normal(120.0, 40.0)),
            'Idle Std': max(0.001, np.random.normal(45.0, 15.0)),
            'Label': 'DNSTunneling',
            'Label_Binary': 1
        }
        synthetic_data.append(dns_sample)

    df = pd.DataFrame(synthetic_data)

    # Remove any negative values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].clip(lower=0.1)
    
    print(f"🎯 Created BALANCED dataset: {len(df)} total rows")
    print(f"📊 Normal: {6000}, Attacks: {6000}")
    print(f"📈 Balanced ratio: 1:1")
    
    return df

def save_attack_type_metrics(model, X_test, y_test, label_encoder):
    """Save metrics for attack type classifier"""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    y_pred = model.predict(X_test)
    
    # Get actual classes present in y_test
    test_classes = np.unique(y_test)
    test_class_names = [label_encoder.inverse_transform([c])[0] for c in test_classes]
    
    # Save confusion matrix only if we have at least 2 classes
    if len(test_classes) >= 2:
        cm = confusion_matrix(y_test, y_pred, labels=test_classes)
        plt.figure(figsize=(max(8, len(test_classes)*1.5), max(6, len(test_classes)*1.2)))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=test_class_names, 
                    yticklabels=test_class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Attack Type Classification - Confusion Matrix')
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, 'attack_type_confusion_matrix.png'), dpi=150)
        plt.close()
        print("  ✅ Saved confusion matrix")
    else:
        print("  ⚠️ Skipped confusion matrix (not enough classes)")
    
    # Save classification report
    try:
        report = classification_report(
            y_test, 
            y_pred, 
            target_names=test_class_names, 
            digits=3,
            zero_division=0
        )
        with open(os.path.join(RESULTS_DIR, 'attack_type_classification_report.txt'), 'w') as f:
            f.write("Attack Type Classification Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Classes tested: {', '.join(test_class_names)}\n")
            f.write(f"Number of samples: {len(y_test)}\n\n")
            f.write(report)
        print("  ✅ Saved classification report")
    except Exception as e:
        print(f"  ⚠️ Could not save classification report: {e}")
    
    print("📊 Saved attack type classifier metrics")

def train_balanced_models():
    """Train models that balance attack detection with false positive reduction"""
    print("⚖️ Training BALANCED Attack Detection Models...")
    print("=" * 70)
    
    # Load real CIC-IDS-2017 data
    preprocessor = DataPreprocessor()
    sample_fraction = MODEL_CONFIG.get('sample_fraction', 0.1)
    print(f"📥 Loading CIC-IDS-2017 with sample_fraction={sample_fraction} ...")
    df = preprocessor.load_cic_2017_data(sample_fraction=sample_fraction)
    if df is None or df.empty:
        raise RuntimeError("No CIC-IDS-2017 data loaded. Please place CSVs in data/CIC_IDS_2017.")

    # ===== TRAIN ATTACK TYPE CLASSIFIER FIRST =====
    print("\n🎯 Training ATTACK TYPE CLASSIFIER...")
    print("-" * 50)
    
    # First, let's see what attack types we have
    print("📊 Analyzing attack types in dataset...")
    attack_df = df[df['is_attack'] == 1].copy()
    
    # Use the preprocessor's attack mapping to get broader categories
    print("🗺️ Mapping attacks to broader categories...")
    attack_df['Attack_Category'] = attack_df['Label'].map(preprocessor.attack_map)
    
    # Fill any unmapped attacks with 'Other'
    attack_df['Attack_Category'] = attack_df['Attack_Category'].fillna('Other')
    
    # Check category distribution
    category_counts = attack_df['Attack_Category'].value_counts()
    print(f"Total attack samples: {len(attack_df)}")
    print(f"Unique attack categories after mapping: {len(category_counts)}")
    print("Attack category distribution:")
    for category, count in category_counts.items():
        print(f"  {category}: {count}")
    
    # Use ALL attack categories regardless of sample count
    print(f"\n📊 Using ALL {len(category_counts)} attack categories (no filtering)...")
    valid_attack_categories = category_counts.index.tolist()
    
    if len(valid_attack_categories) < 2:
        print(f"⚠️ Not enough attack categories ({len(valid_attack_categories)}). Need at least 2.")
        print("Will train only binary classifiers.")
        attack_clf = None
        attack_label_encoder = None
    else:
        # Use ALL attack samples
        filtered_attack_df = attack_df.copy()  # Use all attacks
        print(f"📊 Using ALL {len(filtered_attack_df)} attack samples for attack type classification")
        print(f"📊 Class distribution (using ALL attacks):")
        for category in valid_attack_categories:
            count = len(filtered_attack_df[filtered_attack_df['Attack_Category'] == category])
            print(f"  {category}: {count} samples")
        
        # Use the preprocessor to prepare features for attack type classification
        print("\n🔧 Preparing features using preprocessor...")
        
        # Get all attack samples with their categories
        attack_samples_df = df[df['is_attack'] == 1].copy()
        attack_samples_df['Attack_Category'] = attack_samples_df['Label'].map(preprocessor.attack_map)
        attack_samples_df['Attack_Category'] = attack_samples_df['Attack_Category'].fillna('Other')
        
        # Create a temporary Label column with the mapped categories
        attack_samples_df = attack_samples_df.copy()
        attack_samples_df['Label_Original'] = attack_samples_df['Label']
        attack_samples_df['Label'] = attack_samples_df['Attack_Category']
        
        try:
            # Prepare features specifically for attack type classification
            # CRITICAL: Use SAME n_components as binary classifier (35)
            X_processed, y_encoded, attack_label_encoder = preprocessor.prepare_attack_type_features(
                attack_samples_df,
                use_pca=True,
                n_components=35,  # MUST match binary classifier's 35 components
                min_samples_per_class=1,  # Include ALL attack types
                simplify_classes=False  # Don't simplify further since we already mapped
            )
            
            if X_processed is None or y_encoded is None:
                print("⚠️ Could not prepare features for attack type classification")
                attack_clf = None
                attack_label_encoder = None
            else:
                print(f"\n📊 Attack Type Classification Dataset:")
                print(f"  Samples: {X_processed.shape[0]}")
                print(f"  Features: {X_processed.shape[1]}")  # This should show 35
                print(f"  Number of classes: {len(np.unique(y_encoded))}")
                print(f"  Classes: {list(attack_label_encoder.classes_)}")
                
                # Check if we have 35 PCA components
                if X_processed.shape[1] != 35:
                    print(f"⚠️ WARNING: Expected 35 PCA components, got {X_processed.shape[1]}")
                    print(f"   This will cause feature mismatch errors!")
                
                # Handle class imbalance
                from sklearn.utils.class_weight import compute_class_weight
                classes = np.unique(y_encoded)
                class_weights = compute_class_weight('balanced', classes=classes, y=y_encoded)
                class_weight_dict = {cls: weight for cls, weight in zip(classes, class_weights)}
                print(f"  Class weights: {class_weight_dict}")
                
                # Split data for attack type classification
                try:
                    X_train_attack, X_test_attack, y_train_attack, y_test_attack = train_test_split(
                        X_processed, y_encoded, 
                        test_size=0.25, 
                        random_state=42,
                        stratify=y_encoded
                    )
                except:
                    print("⚠️ Stratified split failed, using random split")
                    X_train_attack, X_test_attack, y_train_attack, y_test_attack = train_test_split(
                        X_processed, y_encoded, 
                        test_size=0.25, 
                        random_state=42
                    )
                
                print(f"  Training samples: {X_train_attack.shape[0]}")
                print(f"  Testing samples: {X_test_attack.shape[0]}")
                print(f"  Training features: {X_train_attack.shape[1]}")  # Should be 35
                
                # Use Random Forest for attack type classification
                print("\n🔧 Training Random Forest for attack type classification...")
                attack_clf = RandomForestClassifier(
                    n_estimators=200,
                    max_depth=25,
                    min_samples_split=2,
                    min_samples_leaf=1,
                    class_weight='balanced_subsample',
                    random_state=42,
                    n_jobs=-1,
                    bootstrap=True,
                    oob_score=True
                )
                
                attack_clf.fit(X_train_attack, y_train_attack)
                
                # Evaluate attack type classifier
                y_pred_attack = attack_clf.predict(X_test_attack)
                
                # Get the actual classes present in y_test_attack
                test_classes = np.unique(y_test_attack)
                test_class_names = [attack_label_encoder.inverse_transform([c])[0] for c in test_classes]
                
                print("\n📊 ATTACK TYPE CLASSIFICATION REPORT")
                try:
                    report = classification_report(
                        y_test_attack,
                        y_pred_attack,
                        target_names=test_class_names,
                        digits=3,
                        zero_division=0
                    )
                    print(report)
                except Exception as e:
                    print(f"⚠️ Could not generate full classification report: {e}")
                    accuracy = accuracy_score(y_test_attack, y_pred_attack)
                    print(f"📊 Simple Accuracy: {accuracy:.3f}")
                    print(f"📊 Tested on {len(test_classes)} classes: {test_class_names}")
                
                accuracy = accuracy_score(y_test_attack, y_pred_attack)
                print(f"📊 Overall Accuracy: {accuracy:.3f}")
                
                # Save attack type classifier and related files
                import joblib
                joblib.dump(attack_clf, os.path.join(MODELS_DIR, "attack_type_classifier.joblib"))
                joblib.dump(attack_label_encoder, os.path.join(MODELS_DIR, "attack_label_encoder.joblib"))
                
                # Save the features used for attack type classification
                if hasattr(preprocessor, 'attack_feature_columns'):
                    joblib.dump(preprocessor.attack_feature_columns, os.path.join(MODELS_DIR, "attack_type_features.joblib"))
                    print(f"💾 Saved {len(preprocessor.attack_feature_columns)} attack feature columns")
                
                # ==== CRITICAL: Save attack type preprocessor ====
                print("💾 Saving attack type preprocessor...")
                attack_preprocessor_data = {
                    'scaler': preprocessor.scaler,
                    'feature_selector': preprocessor.feature_selector,
                    'pca': preprocessor.pca,
                    'feature_columns': preprocessor.attack_feature_columns if hasattr(preprocessor, 'attack_feature_columns') else [],
                    'feature_defaults': preprocessor.feature_defaults if hasattr(preprocessor, 'feature_defaults') else {},
                    '_fitted': preprocessor._fitted,
                    'n_components': 35  # CRITICAL: Save the number of PCA components
                }
                joblib.dump(attack_preprocessor_data, os.path.join(MODELS_DIR, "attack_preprocessor.joblib"))
                print("✅ Attack type preprocessor saved with 35 PCA components")
                # ===== END CRITICAL SECTION =====
                
                # Save metrics and graphs for attack type classifier
                save_attack_type_metrics(attack_clf, X_test_attack, y_test_attack, attack_label_encoder)
                print("✅ Attack type classifier saved with metrics")
                
        except Exception as e:
            print(f"⚠️ Error training attack type classifier: {e}")
            import traceback
            traceback.print_exc()
            attack_clf = None
            attack_label_encoder = None

    # ===== TRAIN BINARY ATTACK DETECTION MODELS =====
    print("\n" + "="*70)
    print("🎯 Training BINARY ATTACK DETECTION MODELS...")
    print("="*70)
    
    # IMPORTANT: Create a NEW preprocessor for binary classification
    # because the previous one was modified for attack type classification
    print("🔄 Creating fresh preprocessor for binary classification...")
    binary_preprocessor = DataPreprocessor()
    
    # Prepare features for binary classification
    X_processed, y_binary, _ = binary_preprocessor.prepare_features(df, use_pca=True, n_components=35)
    print(f"📐 Final feature shape: {X_processed.shape}")
    
    # Split data for binary classification
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y_binary.astype(int), 
        test_size=0.3,
        random_state=42,
        stratify=y_binary
    )
    print(f"\n📊 Dataset Info:")
    print(f"   Training samples: {X_train.shape[0]}")
    print(f"   Testing samples: {X_test.shape[0]}")
    print(f"   Features: {X_train.shape[1]}")
    print(f"   Class balance: {y_train.mean():.3f} attack ratio")
    
    models = {}
    results = {}
    
    print("\n🎯 Training BALANCED models...")
    
    # More aggressive class weights for better attack detection
    class_weight = {0: 1, 1: 3}
    
    # 1. Logistic Regression
    print("1. Training Logistic Regression (Attack-Sensitive)...")
    lr = LogisticRegression(
        random_state=42,
        class_weight=class_weight,
        C=0.01,
        max_iter=2000,
        solver='liblinear'
    )
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    y_proba = lr.predict_proba(X_test)[:, 1]
    models['logistic_regression'] = lr
    results['logistic_regression'] = evaluate_model(y_test, y_pred, y_proba)
    
    # 2. Decision Tree
    print("2. Training Decision Tree (Attack-Sensitive)...")
    dt = DecisionTreeClassifier(
        random_state=42,
        class_weight=class_weight,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5
    )
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)
    y_proba = dt.predict_proba(X_test)[:, 1]
    models['decision_tree'] = dt
    results['decision_tree'] = evaluate_model(y_test, y_pred, y_proba)
    
    # 3. Random Forest
    print("3. Training Random Forest (Attack-Sensitive)...")
    rf = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight=class_weight,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features=0.5,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    y_proba = rf.predict_proba(X_test)[:, 1]
    models['random_forest'] = rf
    results['random_forest'] = evaluate_model(y_test, y_pred, y_proba)
    
    # 4. XGBoost
    print("4. Training XGBoost (Attack-Sensitive)...")
    xgb = XGBClassifier(
        n_estimators=200,
        random_state=42,
        max_depth=10,
        learning_rate=0.05,
        scale_pos_weight=3,
        subsample=0.7,
        colsample_bytree=0.7,
        eval_metric='logloss',
        n_jobs=-1
    )
    xgb.fit(X_train, y_train)
    y_pred = xgb.predict(X_test)
    y_proba = xgb.predict_proba(X_test)[:, 1]
    models['xgboost'] = xgb
    results['xgboost'] = evaluate_model(y_test, y_pred, y_proba)
    
    # 5. Gaussian Naive Bayes
    print("5. Training Gaussian Naive Bayes...")
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    y_pred = gnb.predict(X_test)
    y_proba = gnb.predict_proba(X_test)[:, 1]
    models['gaussian_nb'] = gnb
    results['gaussian_nb'] = evaluate_model(y_test, y_pred, y_proba)
    
    # 6. SVM
    print("6. Training SVM (Attack-Sensitive)...")
    svm = SVC(
        kernel='rbf',
        random_state=42,
        class_weight=class_weight,
        probability=True,
        C=0.1,
        gamma='scale',
        max_iter=2000
    )
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)
    y_proba = svm.predict_proba(X_test)[:, 1]
    models['svm'] = svm
    results['svm'] = evaluate_model(y_test, y_pred, y_proba)
    
    # 7. K-Nearest Neighbors
    print("7. Training K-Nearest Neighbors (Attack-Sensitive)...")
    knn = KNeighborsClassifier(
        n_neighbors=5,
        weights='distance',
        metric='manhattan',
        n_jobs=-1
    )
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    y_proba = knn.predict_proba(X_test)[:, 1]
    models['knn'] = knn
    results['knn'] = evaluate_model(y_test, y_pred, y_proba)
    
    # 8. Isolation Forest
    print("8. Training Isolation Forest (Enhanced for Stealthy Attacks)...")
    contamination = 0.35
    
    iso_forest = IsolationForest(
        contamination=contamination,
        random_state=42,
        n_estimators=300,
        max_samples=0.7,
        max_features=0.9,
        n_jobs=-1,
        verbose=0
    )
    iso_forest.fit(X_train)
    y_pred_raw = iso_forest.predict(X_test)
    y_pred = (y_pred_raw == -1).astype(int)
    
    # Calculate confidence scores
    decision_scores = iso_forest.decision_function(X_test)
    confidence_scores = 1 / (1 + np.exp(-decision_scores))
    
    models['isolation_forest'] = iso_forest
    results['isolation_forest'] = evaluate_model(y_test, y_pred, confidence_scores)
    
    # Display results
    print("\n" + "="*70)
    print("📊 BALANCED MODEL PERFORMANCE")
    print("="*70)
    
    for model_name, result in results.items():
        print(f"\n{model_name.upper():<20}")
        print(f"  Accuracy:  {result['accuracy']:.4f}")
        print(f"  F1-Score:  {result['f1_score']:.4f}")
        if 'roc_auc' in result and result['roc_auc'] is not None:
            print(f"  ROC-AUC:   {result['roc_auc']:.4f}")
        print(f"  Precision: {result['precision']:.4f}")
        print(f"  Recall:    {result['recall']:.4f}")
        print(f"  FP Rate:   {result['fp_rate']:.4f}")
    
    # Save models
    print("\n💾 Saving balanced models...")
    import joblib
    
    for name, model in models.items():
        file_path = os.path.join(MODELS_DIR, f'{name}.joblib')
        joblib.dump(model, file_path)
        print(f"   ✅ Saved {name}")
    
    # Save binary preprocessor
    binary_preprocessor.save_preprocessor(os.path.join(MODELS_DIR, 'preprocessor.joblib'))
    print("   ✅ Saved binary preprocessor")

    # Save evaluation artifacts (confusion matrix, ROC, classification report)
    print("\n🖼️ Saving evaluation plots/reports for binary classifiers...")
    for name, model in models.items():
        try:
            if name == 'isolation_forest':
                y_pred = np.where(model.predict(X_test) == -1, 1, 0)
                y_scores = -model.decision_function(X_test)
            else:
                y_pred = model.predict(X_test)
                if hasattr(model, 'predict_proba'):
                    y_scores = model.predict_proba(X_test)[:, 1]
                elif hasattr(model, 'decision_function'):
                    y_scores = model.decision_function(X_test)
                else:
                    y_scores = None
            _save_confusion_and_roc(name, y_test, y_pred, y_scores)
        except Exception as e:
            print(f"⚠️ Could not save metrics for {name}: {e}")
    
    # Test with both normal and attack traffic
    test_balanced_detection(models, binary_preprocessor, attack_clf, attack_label_encoder)
    
    return models, binary_preprocessor, results, attack_clf, attack_label_encoder

def evaluate_model(y_true, y_pred, y_proba):
    """Comprehensive model evaluation with false positive tracking"""
    report = classification_report(y_true, y_pred, output_dict=True)
    
    # Calculate false positive rate
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        fp_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
    else:
        fp_rate = 0
    
    result = {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred),
        'precision': report['1']['precision'],
        'recall': report['1']['recall'],
        'fp_rate': fp_rate
    }
    
    if y_proba is not None:
        try:
            result['roc_auc'] = roc_auc_score(y_true, y_proba)
        except:
            result['roc_auc'] = 0.5
    else:
        result['roc_auc'] = None
    
    return result


def _save_confusion_and_roc(model_name, y_true, y_pred, y_scores, results_dir=RESULTS_DIR):
    """Save confusion matrix, ROC curve, and classification report safely"""
    os.makedirs(results_dir, exist_ok=True)
    safe_name = model_name.replace(" ", "_").lower()

    # ---- CONFUSION MATRIX ----
    labels = [0, 1]  # Force both classes
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    plt.figure(figsize=(4.5, 3.6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        cbar=False,
        xticklabels=['Normal', 'Attack'],
        yticklabels=['Normal', 'Attack']
    )
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.tight_layout()

    cm_path = os.path.join(results_dir, f'{safe_name}_confusion_matrix.png')
    plt.savefig(cm_path, dpi=150)
    plt.close()

    # ---- ROC CURVE ----
    auc_val = None
    if y_scores is not None and len(np.unique(y_true)) == 2:
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        auc_val = roc_auc_score(y_true, y_scores)

        plt.figure(figsize=(4, 3))
        plt.plot(fpr, tpr, label=f'AUC = {auc_val:.3f}')
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.6)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend(loc='lower right')
        plt.tight_layout()

        roc_path = os.path.join(results_dir, f'{safe_name}_roc_curve.png')
        plt.savefig(roc_path, dpi=150)
        plt.close()

    # ---- CLASSIFICATION REPORT ----
    report = classification_report(
        y_true,
        y_pred,
        target_names=['Normal', 'Attack'],
        digits=3
    )

    report_path = os.path.join(results_dir, f'{safe_name}_classification_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"Model: {model_name}\n")
        f.write("=" * 50 + "\n\n")
        f.write(report)
        if auc_val is not None:
            f.write(f"\nROC-AUC: {auc_val:.3f}\n")

    print(f"🖼️ Saved metrics for {model_name}")


def test_balanced_detection(models, preprocessor, attack_clf=None, attack_label_encoder=None):
    """Test with both NORMAL and ATTACK traffic to check false positives"""
    print("\n" + "="*70)
    print("⚖️ TESTING BALANCED DETECTION (Normal vs Attack)")
    print("="*70)
    
    # REALISTIC NORMAL TRAFFIC PATTERNS
    normal_patterns = [
        {
            'name': 'NORMAL_WebBrowsing',
            'data': {
                'Fwd Packet Length Mean': 450.0, 'Fwd Packet Length Std': 180.0,
                'Bwd Packet Length Mean': 350.0, 'Bwd Packet Length Std': 150.0,
                'Flow Bytes/s': 120000.0, 'Flow Packets/s': 150.0,
                'Flow IAT Mean': 1.2, 'Flow IAT Std': 1.8,
                'Fwd IAT Mean': 1.0, 'Fwd IAT Std': 1.5,
                'Bwd IAT Mean': 1.4, 'Bwd IAT Std': 2.0,
                'Fwd Packets/s': 80.0, 'Bwd Packets/s': 70.0,
                'Packet Length Mean': 400.0, 'Packet Length Std': 170.0,
                'Packet Length Variance': 30000.0,
                'Average Packet Size': 410.0,
                'Avg Fwd Segment Size': 450.0, 'Avg Bwd Segment Size': 350.0,
                'Active Mean': 0.8, 'Active Std': 1.2,
                'Idle Mean': 1.8, 'Idle Std': 2.5
            }
        }
    ]
    
    # MORE DISTINCT ATTACK PATTERNS
    attack_patterns = [
        {
            'name': 'ATTACK_DDoS',
            'label': 'DDoS',
            'data': {
                'Fwd Packet Length Mean': 1500.0, 'Fwd Packet Length Std': 50.0,
                'Bwd Packet Length Mean': 10.0, 'Bwd Packet Length Std': 5.0,
                'Flow Bytes/s': 2000000.0, 'Flow Packets/s': 5000.0,
                'Flow IAT Mean': 0.001, 'Flow IAT Std': 0.0005,
                'Fwd IAT Mean': 0.001, 'Fwd IAT Std': 0.0005,
                'Bwd IAT Mean': 0.0, 'Bwd IAT Std': 0.0,
                'Fwd Packets/s': 5000.0, 'Bwd Packets/s': 5.0,
                'Packet Length Mean': 1400.0, 'Packet Length Std': 50.0,
                'Packet Length Variance': 2500.0,
                'Average Packet Size': 1400.0,
                'Avg Fwd Segment Size': 1500.0, 'Avg Bwd Segment Size': 10.0,
                'Active Mean': 0.001, 'Active Std': 0.0005,
                'Idle Mean': 0.0, 'Idle Std': 0.0
            }
        },
        {
            'name': 'ATTACK_PortScan', 
            'label': 'PortScan',
            'data': {
                'Fwd Packet Length Mean': 60.0, 'Fwd Packet Length Std': 5.0,
                'Bwd Packet Length Mean': 0.0, 'Bwd Packet Length Std': 0.0,
                'Flow Bytes/s': 50000.0, 'Flow Packets/s': 1500.0,
                'Flow IAT Mean': 0.005, 'Flow IAT Std': 0.002,
                'Fwd IAT Mean': 0.005, 'Fwd IAT Std': 0.002,
                'Bwd IAT Mean': 0.0, 'Bwd IAT Std': 0.0,
                'Fwd Packets/s': 1500.0, 'Bwd Packets/s': 0.0,
                'Packet Length Mean': 60.0, 'Packet Length Std': 5.0,
                'Packet Length Variance': 25.0,
                'Average Packet Size': 60.0,
                'Avg Fwd Segment Size': 60.0, 'Avg Bwd Segment Size': 0.0,
                'Active Mean': 0.005, 'Active Std': 0.002,
                'Idle Mean': 0.0, 'Idle Std': 0.0
            }
        },
        {
            'name': 'ATTACK_BruteForce',
            'label': 'BruteForce',
            'data': {
                'Fwd Packet Length Mean': 70.0, 'Fwd Packet Length Std': 3.0,
                'Bwd Packet Length Mean': 60.0, 'Bwd Packet Length Std': 2.0,
                'Flow Bytes/s': 500000.0, 'Flow Packets/s': 3000.0,
                'Flow IAT Mean': 0.0001, 'Flow IAT Std': 0.00005,
                'Fwd IAT Mean': 0.0001, 'Fwd IAT Std': 0.00005,
                'Bwd IAT Mean': 0.0001, 'Bwd IAT Std': 0.00005,
                'Fwd Packets/s': 1500.0, 'Bwd Packets/s': 1500.0,
                'Packet Length Mean': 65.0, 'Packet Length Std': 4.0,
                'Packet Length Variance': 16.0,
                'Average Packet Size': 65.0,
                'Avg Fwd Segment Size': 70.0, 'Avg Bwd Segment Size': 60.0,
                'Active Mean': 0.0001, 'Active Std': 0.00005,
                'Idle Mean': 0.0, 'Idle Std': 0.0
            }
        }
    ]
    
    # Test NORMAL patterns (should be classified as NORMAL)
    print("\n🔍 TESTING NORMAL TRAFFIC (Should be NORMAL):")
    print("-" * 50)
    
    normal_false_positives = 0
    total_normal_tests = 0
    
    for pattern in normal_patterns:
        print(f"\nTesting: {pattern['name']}")
        df = pd.DataFrame([pattern['data']])
        X_processed = preprocessor.prepare_real_time_features(df)
        
        if X_processed is None:
            continue
            
        for name, model in models.items():
            try:
                if hasattr(model, 'predict_proba'):
                    prediction = model.predict(X_processed)[0]
                    confidence = np.max(model.predict_proba(X_processed)[0])
                    label = 'ATTACK' if prediction == 1 else 'NORMAL'
                else:
                    # Isolation Forest
                    raw_pred = model.predict(X_processed)[0]
                    is_anomaly = (raw_pred == -1)
                    label = 'ATTACK' if is_anomaly else 'NORMAL'
                    decision_score = model.decision_function(X_processed)[0]
                    confidence = 1 / (1 + np.exp(-decision_score))
                
                total_normal_tests += 1
                if label == 'ATTACK':
                    normal_false_positives += 1
                    print(f"  ❌ {name}: FALSE POSITIVE (confidence: {confidence:.2f})")
                else:
                    print(f"  ✅ {name}: NORMAL (confidence: {confidence:.2f})")
                    
            except Exception as e:
                print(f"  ⚠️ {name}: ERROR - {e}")
    
    # Test ATTACK patterns (should be classified as ATTACK)
    print("\n🔥 TESTING ATTACK TRAFFIC (Should be ATTACK):")
    print("-" * 50)
    
    attack_detections = 0
    total_attack_tests = 0
    correct_attack_type = 0
    
    for pattern in attack_patterns:
        print(f"\nTesting: {pattern['name']} (Expected: {pattern['label']})")
        df = pd.DataFrame([pattern['data']])
        X_processed = preprocessor.prepare_real_time_features(df)
        
        if X_processed is None:
            continue
            
        for name, model in models.items():
            try:
                if hasattr(model, 'predict_proba'):
                    prediction = model.predict(X_processed)[0]
                    confidence = np.max(model.predict_proba(X_processed)[0])
                    label = 'ATTACK' if prediction == 1 else 'NORMAL'
                else:
                    # Isolation Forest
                    raw_pred = model.predict(X_processed)[0]
                    is_anomaly = (raw_pred == -1)
                    label = 'ATTACK' if is_anomaly else 'NORMAL'
                    decision_score = model.decision_function(X_processed)[0]
                    confidence = 1 / (1 + np.exp(-decision_score))
                
                total_attack_tests += 1
                
                if label == 'ATTACK':
                    attack_detections += 1
                    
                    # Try to classify attack type if attack classifier exists
                    attack_type = "Unknown"
                    if attack_clf is not None and attack_label_encoder is not None:
                        try:
                            # For attack type prediction, we need to use the original features
                            attack_pred = predict_attack_type_improved(df, attack_clf, attack_label_encoder, preprocessor)
                            attack_type = attack_pred[0]
                            attack_confidence = attack_pred[1]
                            
                            # Check if attack type is correct
                            if attack_type == pattern['label']:
                                correct_attack_type += 1
                                type_indicator = "✅"
                            else:
                                type_indicator = "❌"
                            
                            print(f"  {type_indicator} {name}: ATTACK - Type: {attack_type} (confidence: {attack_confidence:.2f})")
                        except Exception as e:
                            print(f"  ⚠️ {name}: ATTACK - Type classification failed: {e}")
                    else:
                        print(f"  ✅ {name}: ATTACK (confidence: {confidence:.2f})")
                else:
                    print(f"  ❌ {name}: MISSED ATTACK (confidence: {confidence:.2f})")
                    
            except Exception as e:
                print(f"  ⚠️ {name}: ERROR - {e}")
    
    # Final results
    print("\n" + "="*70)
    print("🏁 BALANCED DETECTION RESULTS")
    print("="*70)
    
    false_positive_rate = (normal_false_positives / total_normal_tests) * 100 if total_normal_tests > 0 else 0
    attack_detection_rate = (attack_detections / total_attack_tests) * 100 if total_attack_tests > 0 else 0
    
    print(f"📊 False Positives: {normal_false_positives}/{total_normal_tests} ({false_positive_rate:.1f}%)")
    print(f"📊 Attack Detection: {attack_detections}/{total_attack_tests} ({attack_detection_rate:.1f}%)")
    
    if attack_clf is not None and correct_attack_type > 0:
        attack_type_accuracy = (correct_attack_type / attack_detections) * 100
        print(f"📊 Attack Type Accuracy: {correct_attack_type}/{attack_detections} ({attack_type_accuracy:.1f}%)")
    
    if false_positive_rate <= 10 and attack_detection_rate >= 80:
        print("🎉 EXCELLENT BALANCE: Good attack detection with low false positives!")
    elif false_positive_rate <= 20 and attack_detection_rate >= 70:
        print("👍 GOOD BALANCE: Reasonable performance")
    else:
        print("⚠️ NEEDS ADJUSTMENT: Poor balance between detection and false positives")


def predict_attack_type_improved(features_df, attack_clf, attack_label_encoder, preprocessor=None):
    """
    Improved attack type prediction using the original features
    """
    try:
        # Load the feature names used during training
        import joblib
        
        # Try to load the preprocessor for attack type classification
        attack_preprocessor_path = os.path.join(MODELS_DIR, "attack_preprocessor.joblib")
        feature_names_path = os.path.join(MODELS_DIR, "attack_type_features.joblib")
        
        if not os.path.exists(feature_names_path):
            print("⚠️ Attack type feature names not found")
            return "Unknown", 0.0
        
        feature_names = joblib.load(feature_names_path)
        print(f"🔍 Attack type classifier expects {len(feature_names)} features")
        
        # Extract only the features that were used during training
        available_features = [f for f in feature_names if f in features_df.columns]
        
        print(f"🔍 Found {len(available_features)} matching features in input")
        print(f"🔍 Sample available features: {available_features[:5]}...")
        
        if len(available_features) == 0:
            print(f"⚠️ No matching features found for attack type prediction")
            print(f"   Input columns: {list(features_df.columns)[:10]}...")
            return "Unknown", 0.0
        
        # Prepare the features - IMPORTANT: Use same preprocessing as during training
        X_input = features_df[available_features].copy()
        
        print(f"🔍 Input shape before preprocessing: {X_input.shape}")
        
        # Apply the same preprocessing used during training
        X_input = X_input.fillna(X_input.median())
        X_input = X_input.replace([np.inf, -np.inf], np.nan)
        X_input = X_input.fillna(X_input.median())
        
        # Clip extreme values using the same method as during training
        for col in X_input.columns:
            if X_input[col].nunique() <= 1:
                continue
                
            try:
                q1 = X_input[col].quantile(0.01)
                q99 = X_input[col].quantile(0.99)
                X_input[col] = X_input[col].clip(lower=q1, upper=q99)
                X_input[col] = X_input[col].fillna(X_input[col].median())
            except:
                X_input[col] = X_input[col].fillna(X_input[col].median())
        
        # Ensure we have all required features
        missing_features = set(feature_names) - set(available_features)
        if missing_features:
            print(f"⚠️ Missing {len(missing_features)} features for attack type prediction")
            # Add missing features with default values
            for feature in missing_features:
                X_input[feature] = 0.0  # Use 0 as default
        
        # Reorder columns to match training order
        X_input = X_input[feature_names]
        
        # If we have a separate preprocessor for attack type, use it
        if os.path.exists(attack_preprocessor_path):
            attack_preprocessor = joblib.load(attack_preprocessor_path)
            
            # Apply the same transformations as during training
            if 'scaler' in attack_preprocessor:
                X_input_scaled = attack_preprocessor['scaler'].transform(X_input)
                print(f"✅ Scaled features: {X_input_scaled.shape}")
                
                if 'feature_selector' in attack_preprocessor and attack_preprocessor['feature_selector'] is not None:
                    X_input_selected = attack_preprocessor['feature_selector'].transform(X_input_scaled)
                    print(f"✅ Selected features: {X_input_selected.shape}")
                else:
                    X_input_selected = X_input_scaled
                
                if 'pca' in attack_preprocessor and attack_preprocessor['pca'] is not None:
                    X_input_processed = attack_preprocessor['pca'].transform(X_input_selected)
                    print(f"✅ PCA transformed: {X_input_processed.shape}")
                else:
                    X_input_processed = X_input_selected
            else:
                X_input_processed = X_input.values
        else:
            # Use the main preprocessor if available
            if preprocessor and hasattr(preprocessor, 'scaler') and hasattr(preprocessor.scaler, 'mean_'):
                # Try to use the main preprocessor's transformation
                try:
                    # First scale
                    X_input_scaled = preprocessor.scaler.transform(X_input)
                    
                    # Then feature selection if available
                    if hasattr(preprocessor, 'feature_selector') and preprocessor.feature_selector:
                        X_input_selected = preprocessor.feature_selector.transform(X_input_scaled)
                    else:
                        X_input_selected = X_input_scaled
                    
                    # Then PCA if available
                    if hasattr(preprocessor, 'pca') and preprocessor.pca:
                        X_input_processed = preprocessor.pca.transform(X_input_selected)
                    else:
                        X_input_processed = X_input_selected
                        
                    print(f"✅ Using main preprocessor for transformation")
                except Exception as e:
                    print(f"⚠️ Could not use main preprocessor: {e}")
                    X_input_processed = X_input.values
            else:
                X_input_processed = X_input.values
        
        print(f"🔍 Final input shape for prediction: {X_input_processed.shape}")
        
        # Predict attack type
        attack_pred = attack_clf.predict(X_input_processed)[0]
        attack_type = attack_label_encoder.inverse_transform([attack_pred])[0]
        
        if hasattr(attack_clf, 'predict_proba'):
            confidence = np.max(attack_clf.predict_proba(X_input_processed)[0])
        else:
            confidence = 1.0
        
        return attack_type, confidence
    except Exception as e:
        print(f"⚠️ Attack type prediction error: {e}")
        import traceback
        traceback.print_exc()
        return "Unknown", 0.0

def main():
    """Main training function"""
    print("⚖️ Starting REAL CIC-IDS-2017 Attack Model Training...")
    try:
        # Train models including attack type classifier
        models, preprocessor, results, attack_clf, attack_label_encoder = train_balanced_models()
        
        if models is not None:
            print("\n" + "="*70)
            print("✅ BALANCED TRAINING COMPLETED!")
            print("="*70)
            print("🎯 Models trained:")
            print("   1. Binary Attack Detection Models (8 algorithms)")
            if attack_clf is not None:
                print("   2. Attack Type Classifier")
                print(f"      - Attack classes: {list(attack_label_encoder.classes_)}")
            print("\n📊 All metrics and graphs saved in results/ directory")
            print("💾 All models saved in models/ directory")
            print("🚀 You can now run the dashboard: python main.py")
            
        else:
            print("\n❌ Training failed!")
            
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()