# data_diagnostic.py
import pandas as pd
from utils.preprocessor import DataPreprocessor
from config import CIC_DATA_DIR
import os

def diagnose_data_issues():
    """Check what's wrong with your dataset"""
    print("🔍 Diagnosing Data Issues...")
    print("=" * 50)
    
    preprocessor = DataPreprocessor()
    
    # Load data
    df = preprocessor.load_cic_2017_data(sample_fraction=0.1)
    
    if df is None or len(df) == 0:
        print("❌ No data loaded at all!")
        return
    
    print(f"📊 Dataset loaded: {len(df)} rows, {len(df.columns)} columns")
    
    # Check for Label column
    if 'Label' not in df.columns:
        print("❌ No 'Label' column found!")
        print("Available columns:", df.columns.tolist()[:10])
        return
    
    # Check class distribution
    print("\n📈 Class Distribution:")
    label_counts = df['Label'].value_counts()
    print(label_counts)
    
    # Check what values are in Label column
    print(f"\n🔍 Label column analysis:")
    print(f"   - Unique values: {df['Label'].unique()}")
    print(f"   - Data types: {df['Label'].dtype}")
    print(f"   - Sample values: {df['Label'].head(10).tolist()}")
    
    # Check if we need to map labels
    if 'Label' in df.columns:
        # Try different possible label encodings
        possible_attack_indicators = ['ATTACK', 'attack', 'Malicious', 'malicious', 1, '1', True]
        
        for indicator in possible_attack_indicators:
            attack_count = (df['Label'] == indicator).sum()
            if attack_count > 0:
                print(f"✅ Found {attack_count} attacks using indicator: {indicator}")
        
        # Check for string labels that might need conversion
        if df['Label'].dtype == 'object':
            print(f"\n🔄 Label column contains strings: {df['Label'].unique()}")
            # Try to find attack patterns in strings
            attack_keywords = ['ddos', 'portscan', 'brute', 'bot', 'infiltration', 'web', 'attack']
            for keyword in attack_keywords:
                matching = df[df['Label'].str.contains(keyword, case=False, na=False)]
                if len(matching) > 0:
                    print(f"   - Found {len(matching)} rows with '{keyword}' in label")
    
    # Check specific files in data directory
    print(f"\n📁 Checking CIC-IDS-2017 files:")
    csv_files = [f for f in os.listdir(CIC_DATA_DIR) if f.endswith('.csv')]
    for file in csv_files[:5]:  # Check first 5 files
        file_path = os.path.join(CIC_DATA_DIR, file)
        try:
            sample_df = pd.read_csv(file_path, nrows=5)
            if 'Label' in sample_df.columns:
                labels = sample_df['Label'].unique()
                print(f"   - {file}: Labels = {labels}")
            else:
                print(f"   - {file}: No Label column")
        except Exception as e:
            print(f"   - {file}: Error reading - {e}")

def fix_label_mapping(df):
    """Fix label mapping issues in CIC-IDS-2017 dataset"""
    print("\n🔄 Attempting to fix label mapping...")
    
    # Common CIC-IDS-2017 label patterns
    attack_patterns = [
        'DDoS', 'PortScan', 'Bot', 'Brute Force', 'Web Attack', 
        'Infiltration', 'Heartbleed', 'FTP-Patator', 'SSH-Patator'
    ]
    
    # Check if labels are strings that need conversion
    if df['Label'].dtype == 'object':
        print("   - Label column is string type")
        
        # Create binary labels
        df['Label_Binary'] = 0  # Default to normal
        
        for pattern in attack_patterns:
            mask = df['Label'].str.contains(pattern, case=False, na=False)
            attack_count = mask.sum()
            if attack_count > 0:
                print(f"   - Found {attack_count} '{pattern}' attacks")
                df.loc[mask, 'Label_Binary'] = 1
        
        total_attacks = df['Label_Binary'].sum()
        print(f"   - Total attacks after mapping: {total_attacks}")
        
        if total_attacks > 0:
            return df, 'Label_Binary'
    
    return df, 'Label'

if __name__ == "__main__":
    diagnose_data_issues()