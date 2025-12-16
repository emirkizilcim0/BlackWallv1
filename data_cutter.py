import pandas as pd
import numpy as np
import os
from config import DATA_CUTTING_CONFIG, CIC_DATA_DIR
import random

class DataCutter:
    def __init__(self):
        self.config = DATA_CUTTING_CONFIG
        
    def load_and_cut_data(self, strategy='random', sample_fraction=0.1, specific_attacks=None):
        """
        Load data and apply cutting strategy
        
        Args:
            strategy: 'random', 'time_based', 'attack_focused', 'balanced'
            sample_fraction: Fraction of data to use
            specific_attacks: List of specific attacks to focus on
        """
        print(f"🎯 Applying data cutting strategy: {strategy}")
        
        # Load all data
        all_data = self._load_all_data()
        
        # Apply cutting strategy
        if strategy == 'random':
            cut_data = self._random_cut(all_data, sample_fraction)
        elif strategy == 'time_based':
            cut_data = self._time_based_cut(all_data, sample_fraction)
        elif strategy == 'attack_focused':
            cut_data = self._attack_focused_cut(all_data, sample_fraction, specific_attacks)
        elif strategy == 'balanced':
            cut_data = self._balanced_cut(all_data, sample_fraction)
        else:
            print(f"⚠️ Unknown strategy: {strategy}. Using random cut.")
            cut_data = self._random_cut(all_data, sample_fraction)
        
        # ENCODE LABELS after cutting
        cut_data = self._encode_labels(cut_data)
        
        return cut_data
    
    def _encode_labels(self, df):
        """Encode labels to create is_attack column"""
        print("🏷️ Encoding labels in cut data...")
        
        label_column = self._get_label_column(df)
        
        if label_column in df.columns:
            # Convert to binary classification (Normal vs Attack)
            df['is_attack'] = df[label_column].apply(
                lambda x: 0 if str(x).strip().upper() == 'BENIGN' else 1
            )
            
            # Print label distribution
            attack_count = df['is_attack'].sum()
            normal_count = len(df) - attack_count
            print(f"📊 Binary distribution: {normal_count} normal, {attack_count} attacks")
        
        return df
    
    def _load_all_data(self):
        """Load all CIC-IDS-2017 files"""
        all_data = []
        
        for filename in os.listdir(CIC_DATA_DIR):
            if filename.endswith('.csv'):
                file_path = os.path.join(CIC_DATA_DIR, filename)
                print(f"📁 Loading {filename}...")
                try:
                    # Read with error handling for different encodings and separators
                    df = self._read_csv_with_fallback(file_path)
                    if df is not None:
                        all_data.append(df)
                        print(f"✅ Loaded {len(df)} rows from {filename}")
                    else:
                        print(f"❌ Failed to load {filename}")
                except Exception as e:
                    print(f"❌ Error loading {filename}: {e}")
        
        if not all_data:
            raise ValueError("No data loaded successfully")
        
        combined_data = pd.concat(all_data, ignore_index=True)
        print(f"📊 Combined dataset: {len(combined_data)} total rows")
        
        return combined_data
    
    def _read_csv_with_fallback(self, file_path):
        """Read CSV file with fallback for different formats"""
        try:
            # Try reading with different parameters
            for encoding in ['utf-8', 'latin-1', 'iso-8859-1']:
                try:
                    # Read first few rows to detect structure
                    df_sample = pd.read_csv(file_path, nrows=5, encoding=encoding, low_memory=False)
                    
                    # Clean column names
                    df_sample.columns = df_sample.columns.str.strip()
                    
                    # Find label column
                    label_column = self._find_label_column(df_sample)
                    if label_column:
                        # Now read the full file (with sampling for large files)
                        df = pd.read_csv(file_path, encoding=encoding, low_memory=False, nrows=50000)  # Limit rows for memory
                        df.columns = df.columns.str.strip()
                        return df
                        
                except (UnicodeDecodeError, pd.errors.ParserError):
                    continue
                    
            print(f"⚠️ Could not read {file_path} with standard encodings")
            return None
            
        except Exception as e:
            print(f"❌ Error reading {file_path}: {e}")
            return None
    
    def _find_label_column(self, df_sample):
        """Find the label column in the dataset"""
        # Common label column names in CIC-IDS-2017
        possible_labels = ['Label', 'label', 'Attack', 'attack', 'Class', 'class', 
                          'Category', 'category', 'Result', 'result']
        
        for label in possible_labels:
            if label in df_sample.columns:
                print(f"   Found label column: '{label}'")
                return label
        
        print(f"⚠️ No standard label column found. Available columns: {list(df_sample.columns)}")
        return None
    
    def _get_label_column(self, data):
        """Get the actual label column name from data"""
        possible_labels = ['Label', 'label', 'Attack', 'attack', 'Class', 'class']
        for label in possible_labels:
            if label in data.columns:
                return label
        raise KeyError("No label column found in dataset")
    
    def _random_cut(self, data, sample_fraction):
        """Random sampling from entire dataset"""
        print(f"🎲 Random cutting: using {sample_fraction*100}% of data")
        sample_size = int(len(data) * sample_fraction)
        return data.sample(n=sample_size, random_state=42)
    
    def _time_based_cut(self, data, sample_fraction):
        """Cut data based on time windows"""
        print("⏰ Time-based cutting...")
        
        # If timestamp column exists, use it. Otherwise use random cut
        timestamp_columns = ['Timestamp', 'timestamp', 'Time', 'time']
        timestamp_col = None
        
        for col in timestamp_columns:
            if col in data.columns:
                timestamp_col = col
                break
        
        if timestamp_col:
            data = data.sort_values(timestamp_col)
            time_window = int(len(data) * sample_fraction)
            return data.head(time_window)  # Use first time window
        else:
            print("⚠️ No timestamp column found. Using random cut instead.")
            return self._random_cut(data, sample_fraction)
    
    def _attack_focused_cut(self, data, sample_fraction, specific_attacks=None):
        """Focus on specific attack types"""
        print("🎯 Attack-focused cutting...")
        
        label_column = self._get_label_column(data)
        
        if specific_attacks:
            print(f"Focusing on attacks: {specific_attacks}")
            attack_data = data[data[label_column].isin(specific_attacks)]
            # Get normal data (BENIGN)
            normal_data = data[data[label_column] == 'BENIGN']
            
            if len(attack_data) == 0:
                print("⚠️ No attack data found with specified attacks. Using balanced cut.")
                return self._balanced_cut(data, sample_fraction)
            
            # Sample from both attack and normal data
            attack_sample_size = min(len(attack_data), int(len(data) * sample_fraction * 0.7))
            normal_sample_size = min(len(normal_data), int(len(data) * sample_fraction * 0.3))
            
            attack_sample = attack_data.sample(n=attack_sample_size, random_state=42)
            normal_sample = normal_data.sample(n=normal_sample_size, random_state=42)
            
            return pd.concat([attack_sample, normal_sample], ignore_index=True)
        else:
            print("⚠️ No specific attacks provided. Using balanced cut.")
            return self._balanced_cut(data, sample_fraction)
    
    def _balanced_cut(self, data, sample_fraction):
        """Balance classes in the dataset"""
        print("⚖️ Balanced cutting...")
        
        label_column = self._get_label_column(data)
        
        # Get value counts for each class
        label_counts = data[label_column].value_counts()
        print(f"Original class distribution:\n{label_counts}")
        
        balanced_data = []
        max_samples = self.config['max_samples_per_class']
        
        for label in data[label_column].unique():
            class_data = data[data[label_column] == label]
            sample_size = min(len(class_data), max_samples, 
                            int(len(data) * sample_fraction / len(data[label_column].unique())))
            
            sampled_class = class_data.sample(n=sample_size, random_state=42)
            balanced_data.append(sampled_class)
            print(f"✅ Class '{label}': {len(sampled_class)} samples")
        
        result = pd.concat(balanced_data, ignore_index=True)
        print(f"🎯 Final balanced dataset: {len(result)} samples")
        print(f"Final class distribution:\n{result[label_column].value_counts()}")
        
        return result