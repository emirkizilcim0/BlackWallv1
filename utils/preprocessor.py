import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
import joblib
import os
from config import CIC_DATA_DIR, MODELS_DIR, DATASET_CONFIG
import warnings
from utils.feature_mapper import FeatureMapper

warnings.filterwarnings('ignore')

# Attack type mapping inspired by the shared CIC-IDS2017 notebook
NOTEBOOK_ATTACK_MAP = {
    'BENIGN': 'BENIGN',
    'DDoS': 'DDoS',
    'DoS Hulk': 'DoS',
    'DoS GoldenEye': 'DoS',
    'DoS slowloris': 'DoS',
    'DoS Slowhttptest': 'DoS',
    'PortScan': 'Port Scan',
    'FTP-Patator': 'Brute Force',
    'SSH-Patator': 'Brute Force',
    'Bot': 'Bot',
    'Web Attack � Brute Force': 'Web Attack',
    'Web Attack � XSS': 'Web Attack',
    'Web Attack � Sql Injection': 'Web Attack',
    'Infiltration': 'Infiltration',
    'Heartbleed': 'Heartbleed'
}

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_selector = None
        self.pca = None
        self.column_mappings = {}
        self._fitted = False  # ADD THIS - critical for real-time detection
        self.feature_columns = []  # ADD THIS - ensure it's initialized
        self.feature_defaults = {}  # Store per-feature default (median) for realtime
        self.attack_map = NOTEBOOK_ATTACK_MAP.copy()


    def load_single_file(self, file_path, sample_fraction=1.0):
        """Public method to load single file - called by train_model.py"""
        return self._load_single_file(file_path, sample_fraction)

    def clean_dataframe(df):
        """Clean dataframe by removing infinity and extreme values"""
        if df is None or len(df) == 0:
            return df
        
        # Replace infinity with NaN
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Remove rows with any NaN values
        original_len = len(df)
        df = df.dropna()
        
        if original_len - len(df) > 0:
            print(f"🧹 Removed {original_len - len(df)} rows with NaN/infinity values")
        
        return df

    def prepare_attack_type_features(self, df, use_pca=False, n_components=30, min_samples_per_class=2, simplify_classes=True):
        """
        Prepare features specifically for attack type classification
        (attack samples only)
        """
        print("🔧 Preparing features for attack type classification...")

        # Filter only attack samples
        attack_df = df[df['is_attack'] == 1].copy()

        if len(attack_df) == 0:
            print("⚠️ No attack samples found for attack type training")
            return None, None, None

        # Apply simplified attack mapping if requested
        if simplify_classes:
            attack_df['Label'] = attack_df['Label'].map(self.attack_map)
            # Fill any unmapped labels with 'Other'
            attack_df['Label'] = attack_df['Label'].fillna('Other')

        # Check and filter classes with too few samples
        label_counts = attack_df['Label'].value_counts()
        print(f"📊 Attack class distribution:")
        for label, count in label_counts.items():
            print(f"   {label}: {count} samples")

        # Filter classes with at least min_samples_per_class
        valid_classes = label_counts[label_counts >= min_samples_per_class].index
        attack_df = attack_df[attack_df['Label'].isin(valid_classes)].copy()

        if len(attack_df) == 0:
            print("⚠️ No attack classes with sufficient samples for training")
            return None, None, None

        print(f"✅ Using {len(valid_classes)} classes with ≥{min_samples_per_class} samples")
        print(f"📊 Total attack samples: {len(attack_df)}")

        # Separate features and target
        feature_columns = [
            col for col in attack_df.columns
            if col not in ['Label', 'label', 'is_attack', 'attack_type']
            and np.issubdtype(attack_df[col].dtype, np.number)
        ]

        print(f"📐 Using {len(feature_columns)} numerical features")

        X = attack_df[feature_columns]
        y = attack_df['Label']  # Use mapped labels

        # Remove columns with constant values
        constant_columns = [col for col in X.columns if X[col].nunique() <= 1]
        if constant_columns:
            print(f"🗑️ Removing constant columns: {constant_columns}")
            X = X.drop(columns=constant_columns)

        # Store the feature columns for attack type classification
        self.attack_feature_columns = X.columns.tolist()
        print(f"💾 Stored {len(self.attack_feature_columns)} attack feature columns")

        # Handle infinite values
        X = X.replace([np.inf, -np.inf], np.nan)

        # Fill any remaining NaN values
        X = X.fillna(X.median())

        # Scale features
        print("⚖️ Scaling features for attack type...")
        X_scaled = self.scaler.fit_transform(X)
        print("✅ Scaler fitted for attack type")

        # Feature selection
        from sklearn.feature_selection import SelectKBest, f_classif
        k = min(50, X_scaled.shape[1])
        attack_feature_selector = SelectKBest(score_func=f_classif, k=k)

        # Encode y for feature selection
        from sklearn.preprocessing import LabelEncoder
        attack_label_encoder = LabelEncoder()
        y_encoded = attack_label_encoder.fit_transform(y)

        X_selected = attack_feature_selector.fit_transform(X_scaled, y_encoded)
        print("✅ Feature selector fitted for attack type")

        # Apply PCA if requested
        if use_pca:
            n_components = min(n_components, X_selected.shape[1])
            pca = PCA(n_components=n_components)
            X_processed = pca.fit_transform(X_selected)
            explained_variance = pca.explained_variance_ratio_.sum()
            print(f"🌀 PCA fitted with {n_components} components "
                  f"(explained variance: {explained_variance:.3f})")
        else:
            X_processed = X_selected
            pca = None

        print(f"✅ Final feature shape for attack type: {X_processed.shape}")
        print(f"📋 Attack classes: {list(attack_label_encoder.classes_)}")

        return X_processed, y_encoded, attack_label_encoder

    def save_preprocessor(self, file_path):
        """Save preprocessor objects to file"""
        # Check if preprocessor is fitted
        if not self.is_fitted():
            raise ValueError("Cannot save preprocessor: Preprocessor not fitted")

        preprocessor_data = {
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_selector': self.feature_selector,
            'pca': self.pca,
            'feature_columns': self.feature_columns,  # Save feature columns
            'feature_defaults': self.feature_defaults,  # Save per-feature defaults
            '_fitted': self._fitted  # Save fitted status
        }

        joblib.dump(preprocessor_data, file_path)
        print(f"💾 Preprocessor saved to {file_path}")
    
    def _get_label_column(self, df):
        """Find the label column in the dataset"""
        possible_labels = ['Label', 'label', 'Attack', 'attack', 'Class', 'class']
        for label in possible_labels:
            if label in df.columns:
                return label
        raise KeyError("No label column found in dataset. Available columns: " + str(list(df.columns)))

    def add_attack_type(self, df):
        """Add human-readable attack type grouping from notebook mapping"""
        if 'Label' not in df.columns:
            return df

        df = df.copy()
        df['Attack Type'] = df['Label'].map(self.attack_map).fillna(df['Label'])
        return df

    def drop_constant_columns(self, df):
        """Remove columns with a single unique value"""
        protected = {'Label', 'label', 'Attack', 'attack', 'Class', 'class', 'Attack Type'}
        constant_cols = [
            col for col in df.columns
            if col not in protected and df[col].nunique(dropna=False) <= 1
        ]
        if constant_cols:
            df = df.drop(columns=constant_cols)
            # Limit console spam
            print(f"🗑️ Removed {len(constant_cols)} constant columns")
        return df

    def optimize_memory(self, df):
        """Downcast numeric columns to reduce memory pressure (matches notebook)"""
        if df is None or df.empty:
            return df

        old_mem = df.memory_usage().sum() / 1024 ** 2
        for col in df.columns:
            col_type = df[col].dtype
            if col_type == object:
                continue

            c_min = df[col].min()
            c_max = df[col].max()

            if str(col_type).find('float') >= 0 and np.isfinite(c_min) and np.isfinite(c_max):
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
            elif str(col_type).find('int') >= 0:
                if c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)

        new_mem = df.memory_usage().sum() / 1024 ** 2
        print(f"💾 Memory usage: {old_mem:.2f}MB -> {new_mem:.2f}MB")
        return df

    def load_preprocessor(self, file_path):
        """Load preprocessor objects from file"""
        if not os.path.exists(file_path):
            print(f"❌ Preprocessor file not found: {file_path}")
            return False

        try:
            preprocessor_data = joblib.load(file_path)

            self.scaler = preprocessor_data['scaler']
            self.label_encoder = preprocessor_data['label_encoder']
            self.feature_selector = preprocessor_data['feature_selector']
            self.pca = preprocessor_data['pca']
            self.feature_columns = preprocessor_data.get('feature_columns', [])
            self.feature_defaults = preprocessor_data.get('feature_defaults', {})
            self._fitted = preprocessor_data.get('_fitted', False)  # Load fitted status

            # Verify that scaler is fitted
            if not hasattr(self.scaler, 'mean_') or self.scaler.mean_ is None:
                print("❌ Loaded scaler is not fitted")
                return False

            print(f"✅ Preprocessor loaded from {file_path}")
            print(f"   - Scaler fitted: {hasattr(self.scaler, 'mean_')}")
            print(f"   - Feature selector fitted: {self.feature_selector is not None}")
            print(f"   - PCA fitted: {self.pca is not None}")
            print(f"   - Feature columns: {len(self.feature_columns)}")
            print(f"   - Preprocessor fitted: {self._fitted}")

            return True

        except Exception as e:
            print(f"❌ Error loading preprocessor: {e}")
            return False

    def is_fitted(self):
        """Check if preprocessor is properly fitted - robust version"""
        # Check if we have the _fitted flag
        if hasattr(self, '_fitted') and self._fitted:
            return True
        
        # If _fitted is False or not set, check if components are actually fitted
        scaler_fitted = hasattr(self.scaler, 'mean_') and self.scaler.mean_ is not None
        selector_fitted = self.feature_selector is not None and hasattr(self.feature_selector, 'scores_')
        has_features = hasattr(self, 'feature_columns') and len(self.feature_columns) > 0
        
        # If all components are fitted, consider the preprocessor as fitted
        if scaler_fitted and selector_fitted and has_features:
            print("⚠️  Preprocessor components are fitted but _fitted flag was False. Auto-correcting...")
            self._fitted = True
            return True
        
        return False
    
    def load_cic_2017_data(self, file_path=None, sample_fraction=0.1):
        """Load CIC-IDS-2017 data with proper handling"""
        if file_path:
            # Load single file
            return self._load_single_file(file_path, sample_fraction)
        else:
            # Load all files in the directory
            return self._load_all_files(sample_fraction)
    
    def _load_all_files(self, sample_fraction=0.1):
        """Load and combine all CIC-IDS-2017 files"""
        all_data = []
        
        for filename in DATASET_CONFIG['expected_files']:
            file_path = os.path.join(CIC_DATA_DIR, filename)
            if os.path.exists(file_path):
                print(f"📁 Loading {filename}...")
                try:
                    df = self._load_single_file(file_path, sample_fraction)
                    if df is not None:
                        all_data.append(df)
                        print(f"✅ Loaded {len(df)} rows from {filename}")
                except Exception as e:
                    print(f"❌ Error loading {filename}: {e}")
            else:
                print(f"⚠️ File not found: {filename}")
        
        if not all_data:
            raise ValueError("No CIC-IDS-2017 files found or loaded successfully")
        
        # Combine all data
        combined_df = pd.concat(all_data, ignore_index=True)
        print(f"🎯 Combined dataset: {len(combined_df)} total rows")
        
        return combined_df
    
    def _load_single_file(self, file_path, sample_fraction=0.1):
        """Load a single CIC-IDS-2017 CSV file"""
        try:
            # CIC-IDS-2017 files are large, so we read in chunks or sample
            print(f"Reading {file_path}...")
            
            # First, let's check the file size and structure
            with open(file_path, 'r') as f:
                first_line = f.readline().strip()
            
            # Detect separator and encoding
            separator = ',' if ',' in first_line else ';'
            
            # Read the first few rows to understand structure
            df_sample = pd.read_csv(file_path, nrows=1000, sep=separator, low_memory=False)
            
            # Handle column names (CIC-IDS-2017 sometimes has spaces)
            df_sample.columns = df_sample.columns.str.strip()
            
            # Check if Label column exists
            if 'Label' not in df_sample.columns:
                # Try to find alternative label columns
                label_candidates = ['label', 'Attack', 'attack', 'Class']
                for candidate in label_candidates:
                    if candidate in df_sample.columns:
                        df_sample = df_sample.rename(columns={candidate: 'Label'})
                        break
            
            if 'Label' not in df_sample.columns:
                print(f"⚠️ No label column found in {file_path}")
                return None
            
            # Now read the full file with sampling
            if sample_fraction < 1.0:
                # Calculate number of rows for sampling
                total_rows = sum(1 for line in open(file_path)) - 1  # minus header
                sample_rows = int(total_rows * sample_fraction)
                
                # Read sampled rows
                skip_rows = np.random.choice(
                    range(1, total_rows + 1), 
                    total_rows - sample_rows, 
                    replace=False
                )
                
                df = pd.read_csv(file_path, sep=separator, skiprows=skip_rows, low_memory=False)
            else:
                # Read full file (be careful with memory!)
                df = pd.read_csv(file_path, sep=separator, low_memory=False)
            
            # Clean column names
            df.columns = df.columns.str.strip()
            
            # Apply the same renaming if needed
            if 'Label' not in df.columns:
                for candidate in label_candidates:
                    if candidate in df.columns:
                        df = df.rename(columns={candidate: 'Label'})
                        break
            
            return self._basic_cleaning(df)
            
        except Exception as e:
            print(f"❌ Error reading {file_path}: {e}")
            return None
    
    def _basic_cleaning(self, df):
        """Perform basic data cleaning for CIC-IDS-2017"""
        print("🧹 Cleaning data...")

        # Remove duplicate rows
        initial_count = len(df)
        df = df.drop_duplicates()
        print(f"Removed {initial_count - len(df)} duplicate rows")
        
        # Remove columns with all missing values
        df = df.dropna(axis=1, how='all')
        
        # Remove unnecessary columns
        columns_to_drop = []
        for col in DATASET_CONFIG['timestamp_columns'] + DATASET_CONFIG['ip_columns']:
            if col in df.columns:
                columns_to_drop.append(col)
        
        if columns_to_drop:
            df = df.drop(columns=columns_to_drop)
            print(f"Dropped columns: {columns_to_drop}")
        
        # Handle missing values
        df = self._handle_missing_values(df)

        # Add notebook-style attack grouping for downstream tasks
        df = self.add_attack_type(df)

        # Drop constant columns to align with notebook preprocessing
        df = self.drop_constant_columns(df)

        # Downcast numerics to ease memory pressure
        df = self.optimize_memory(df)

        # Encode labels
        df = self._encode_labels(df)

        return df
    
    def _handle_missing_values(self, df):
        """Handle missing values in CIC-IDS-2017 dataset"""
        print("🔧 Handling missing values...")
        
        # For numerical columns, fill with median
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].median())
                print(f"Filled missing values in {col} with median")
        
        # For categorical columns, fill with mode
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col != 'Label' and df[col].isna().any():
                if not df[col].mode().empty:
                    df[col] = df[col].fillna(df[col].mode()[0])
                    print(f"Filled missing values in {col} with mode")
                else:
                    df[col] = df[col].fillna('Unknown')
        
        return df
    
    def _encode_labels(self, df):
        """Encode target labels for CIC-IDS-2017"""
        print("🏷️ Encoding labels...")

        label_column = self._get_label_column(df)

        if label_column in df.columns:
            # Ensure Attack Type is present
            if 'Attack Type' not in df.columns:
                df['Attack Type'] = df[label_column].map(self.attack_map).fillna(df[label_column])

            # Convert to binary classification (Normal vs Attack)
            df['is_attack'] = df[label_column].apply(
                lambda x: 0 if str(x).strip().upper() == 'BENIGN' else 1
            )

            # Print label distribution
            attack_count = df['is_attack'].sum()
            normal_count = len(df) - attack_count
            print(f"📊 Label distribution: {normal_count} normal, {attack_count} attacks")

            # Also keep multi-class labels for detailed analysis
            unique_labels = df[label_column].unique()
            print(f"📋 Unique labels: {list(unique_labels)[:10]}...")  # Show first 10

            self.label_encoder.fit(df[label_column])
            df['attack_type'] = self.label_encoder.transform(df[label_column])

        return df

    def prepare_real_time_features(self, df):
        """Prepare features for real-time prediction with feature mapping"""
        print("🔧 Preparing real-time features...")
        
        # Check if preprocessor is fitted
        if not self.is_fitted():
            raise ValueError("Preprocessor not fitted. Please train models first.")

        if not self.feature_defaults:
            raise ValueError("Feature defaults missing. Please retrain and save the preprocessor.")
        
        # Initialize feature mapper
        feature_mapper = FeatureMapper()
        
        # Map features to training feature names
        df_mapped = feature_mapper.map_features(df)
        print(f"📊 After mapping: {len(df_mapped.columns)} features")

        # Build a frame initialized with training medians, then override with incoming values
        base_rows = []
        for _ in range(len(df_mapped)):
            base_rows.append(self.feature_defaults.copy())
        X_aligned = pd.DataFrame(base_rows)

        # Override with any incoming (mapped) numeric columns that match training features
        for col in self.feature_columns:
            if col in df_mapped.columns:
                X_aligned[col] = pd.to_numeric(df_mapped[col], errors='coerce').fillna(self.feature_defaults.get(col, 0.0)).values
        # Ensure order
        X_aligned = X_aligned[self.feature_columns]
        print(f"📐 Using {X_aligned.shape[1]} numerical features for real-time prediction")
        
        # Apply preprocessing pipeline
        try:
            X_scaled = self.scaler.transform(X_aligned)
            print(f"✅ Scaling complete: {X_scaled.shape}")
            
            if self.feature_selector:
                X_selected = self.feature_selector.transform(X_scaled)
                print(f"✅ Feature selection complete: {X_selected.shape}")
            else:
                X_selected = X_scaled
                
            if self.pca:
                X_processed = self.pca.transform(X_selected)
                print(f"✅ PCA complete: {X_processed.shape}")
            else:
                X_processed = X_selected
                
            print(f"🎯 Final processed features: {X_processed.shape}")
            return X_processed
            
        except Exception as e:
            raise ValueError(f"Preprocessing error: {str(e)}")

    def _align_features_with_training(self, X):
        """Align input features with the features used during training"""
        print("🔄 Aligning features with training data...")

        if not hasattr(self, 'feature_columns') or not self.feature_columns:
            raise ValueError("No training feature columns found. Preprocessor may not be fitted properly.")

        trained_features = self.feature_columns
        current_features = X.columns.tolist()

        print(f"📋 Training features: {len(trained_features)}")
        print(f"📋 Current features: {len(current_features)}")
        print(f"📋 Sample training features: {trained_features[:5]}...")

        # Find common features
        common_features = [col for col in trained_features if col in current_features]
        missing_features = [col for col in trained_features if col not in current_features]
        extra_features = [col for col in current_features if col not in trained_features]

        print(f"✅ Common features: {len(common_features)}")
        print(f"❌ Missing features: {len(missing_features)}")
        if missing_features:
            print(f"   Missing: {missing_features[:3]}{'...' if len(missing_features) > 3 else ''}")
        print(f"📈 Extra features: {len(extra_features)}")

        if not common_features:
            raise ValueError(f"No common features found. Training used: {trained_features[:5]}...")

        # Start with common features
        X_aligned = X[common_features].copy()

        # Add missing features with default value (0)
        for feature in missing_features:
            default_val = self.feature_defaults.get(feature, 0.0)
            X_aligned[feature] = default_val
            print(f"   ➕ Added missing feature: {feature} (default: {default_val})")

        # Ensure the order matches the training data exactly
        X_aligned = X_aligned[trained_features]

        print(f"✅ Final aligned features: {X_aligned.shape[1]} features")
        return X_aligned
    
    def prepare_features(self, df, use_pca=False, n_components=30, multiclass=False):
        """Prepare features for training with CIC-IDS-2017"""
        print("🔧 Preparing features...")
    
        # Check if is_attack column exists, if not create it
        if 'is_attack' not in df.columns:
            print("⚠️ 'is_attack' column not found. Creating it...")
            label_column = self._get_label_column(df)
            df['is_attack'] = df[label_column].apply(
                lambda x: 0 if str(x).strip().upper() == 'BENIGN' else 1
            )
    
        # Separate features and target
        feature_columns = [
            col for col in df.columns
            if col not in ['Label', 'label', 'is_attack', 'attack_type']
            and np.issubdtype(df[col].dtype, np.number)
        ]
    
        print(f"📐 Using {len(feature_columns)} numerical features")
    
        X = df[feature_columns]
        
        # Choose target based on multiclass flag
        if multiclass:
            # For multiclass, use attack_type column or create it
            if 'attack_type' in df.columns:
                y = df['attack_type']
            elif 'Label' in df.columns:
                # Create attack type encoding
                # Get attack-only samples
                attack_df = df[df['is_attack'] == 1].copy()
                if not attack_df.empty:
                    # Fit label encoder on attack labels only
                    from sklearn.preprocessing import LabelEncoder
                    attack_label_encoder = LabelEncoder()
                    y_attack = attack_label_encoder.fit_transform(attack_df['Label'])
                    # Create full y array with -1 for normal traffic
                    y = np.full(len(df), -1)
                    y[df['is_attack'] == 1] = y_attack
                else:
                    print("⚠️ No attack samples found for multiclass training")
                    y = df['is_attack']
            else:
                y = df['is_attack']
            y_binary = df['is_attack']
            print(f"📊 Multiclass mode: {len(X)} samples, {len(np.unique(y[y != -1]))} attack classes")
        else:
            # For binary classification
            y = df['is_attack']
            y_binary = y
            print(f"📊 Binary mode: {len(X)} total samples, {y.sum()} attacks")
    
        # Remove columns with constant values
        constant_columns = [col for col in X.columns if X[col].nunique() <= 1]
        if constant_columns:
            print(f"🗑️ Removing constant columns: {constant_columns}")
            X = X.drop(columns=constant_columns)
    
        # Store the feature columns for later use - CRITICAL!
        self.feature_columns = X.columns.tolist()
        print(f"💾 Stored {len(self.feature_columns)} feature columns")
    
        # Store median defaults for realtime filling
        self.feature_defaults = X.median().to_dict()
        print(f"💾 Stored feature medians for {len(self.feature_defaults)} columns")
    
        # Handle infinite values
        X = X.replace([np.inf, -np.inf], np.nan)
    
        # Fill any remaining NaN values
        X = X.fillna(X.median())
    
        # Scale features
        print("⚖️ Scaling features...")
        X_scaled = self.scaler.fit_transform(X)
        print("✅ Scaler fitted")
    
        # Feature selection - use y_binary for feature selection
        print("🎯 Selecting best features...")
        k = min(50, X_scaled.shape[1])
        self.feature_selector = SelectKBest(score_func=f_classif, k=k)
        X_selected = self.feature_selector.fit_transform(X_scaled, y_binary)
        print("✅ Feature selector fitted")
    
        # Apply PCA if requested
        if use_pca:
            n_components = min(n_components, X_selected.shape[1])
            self.pca = PCA(n_components=n_components)
            X_processed = self.pca.fit_transform(X_selected)
            explained_variance = self.pca.explained_variance_ratio_.sum()
            print(f"🌀 PCA fitted with {n_components} components "
                  f"(explained variance: {explained_variance:.3f})")
        else:
            X_processed = X_selected
    
        print(f"✅ Final feature shape: {X_processed.shape}")
    
        # Mark as fitted - CRITICAL FOR REAL-TIME DETECTION
        self._fitted = True
    
        if multiclass:
            return X_processed, y_binary, y
        else:
            return X_processed, y_binary, None
    