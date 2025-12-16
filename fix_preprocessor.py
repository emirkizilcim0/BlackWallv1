# fix_preprocessor_infinity.py
import pandas as pd
import numpy as np
import joblib
import os
from config import MODELS_DIR, CIC_DATA_DIR
from utils.preprocessor import DataPreprocessor
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA

def clean_infinity_values(df):
    """Clean infinity and extreme values from the dataset"""
    print("🧹 Cleaning infinity and extreme values...")
    
    # Select numerical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'Label' in numerical_cols:
        numerical_cols.remove('Label')
    
    original_shape = df.shape
    rows_cleaned = 0
    
    for col in numerical_cols:
        # Replace infinity with NaN
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        
        # Count infinity values
        inf_count = df[col].isna().sum()
        if inf_count > 0:
            print(f"   - {col}: {inf_count} infinity values")
            rows_cleaned += inf_count
    
    # Remove rows with any NaN values (including those from infinity)
    df_cleaned = df.dropna()
    
    print(f"📊 Removed {original_shape[0] - df_cleaned.shape[0]} rows with infinity/NaN values")
    
    # Also cap extreme values to prevent scaling issues
    for col in numerical_cols:
        if col in df_cleaned.columns:
            # Calculate robust statistics (ignore outliers)
            Q1 = df_cleaned[col].quantile(0.01)
            Q3 = df_cleaned[col].quantile(0.99)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            
            # Cap extreme values
            extreme_mask = (df_cleaned[col] < lower_bound) | (df_cleaned[col] > upper_bound)
            extreme_count = extreme_mask.sum()
            
            if extreme_count > 0:
                df_cleaned.loc[extreme_mask, col] = np.clip(
                    df_cleaned.loc[extreme_mask, col], lower_bound, upper_bound
                )
                print(f"   - {col}: Capped {extreme_count} extreme values")
    
    return df_cleaned

def quick_retrain_preprocessor_fixed():
    """Quick retrain with infinity handling"""
    print("⚡ Quick Preprocessor Retraining (Fixed)...")
    print("=" * 50)
    
    preprocessor = DataPreprocessor()
    
    # Load very small dataset for quick fitting
    df = preprocessor.load_cic_2017_data(sample_fraction=0.02)
    
    if df is None:
        print("❌ Could not load data")
        return False
    
    print(f"📊 Original dataset: {len(df)} samples")
    
    # Clean infinity values
    df_cleaned = clean_infinity_values(df)
    
    if len(df_cleaned) == 0:
        print("❌ No valid data after cleaning")
        return False
    
    print(f"📊 Using {len(df_cleaned)} clean samples for preprocessor fitting")
    
    # Get numerical features
    numerical_features = df_cleaned.select_dtypes(include=[np.number]).columns.tolist()
    if 'Label' in numerical_features:
        numerical_features.remove('Label')
    
    X = df_cleaned[numerical_features]
    
    # Verify no infinity or extreme values remain
    print("🔍 Verifying data quality...")
    for col in numerical_features:
        if X[col].isna().any():
            print(f"   ❌ {col} still has NaN values")
        if np.isinf(X[col]).any():
            print(f"   ❌ {col} still has infinity values")
    
    print("📊 Fitting scaler...")
    try:
        preprocessor.scaler.fit(X)
        print("   ✅ Scaler fitted successfully")
    except Exception as e:
        print(f"   ❌ Scaler fitting failed: {e}")
        return False
    
    print("🎯 Fitting feature selector...")
    try:
        preprocessor.feature_selector = SelectKBest(f_classif, k=min(30, X.shape[1]))
        y_binary = (df_cleaned['Label'] > 0).astype(int) if 'Label' in df_cleaned.columns else np.zeros(len(df_cleaned))
        preprocessor.feature_selector.fit(X, y_binary)
        print("   ✅ Feature selector fitted successfully")
    except Exception as e:
        print(f"   ❌ Feature selector fitting failed: {e}")
        # Continue without feature selector
        preprocessor.feature_selector = None
    
    print("🌀 Fitting PCA...")
    try:
        if preprocessor.feature_selector is not None:
            X_selected = preprocessor.feature_selector.transform(X)
        else:
            X_selected = X
        
        preprocessor.pca = PCA(n_components=min(15, X_selected.shape[1]))
        preprocessor.pca.fit(X_selected)
        print("   ✅ PCA fitted successfully")
    except Exception as e:
        print(f"   ❌ PCA fitting failed: {e}")
        # Continue without PCA
        preprocessor.pca = None
    
    preprocessor.feature_columns = numerical_features
    preprocessor.is_fitted_flag = True
    
    # Save the preprocessor
    try:
        preprocessor.save_preprocessor(os.path.join(MODELS_DIR, 'preprocessor.joblib'))
        print("💾 Preprocessor saved successfully!")
        
        # Test loading and fitting
        test_preprocessor_fit()
        
        return True
        
    except Exception as e:
        print(f"❌ Error saving preprocessor: {e}")
        return False

def test_preprocessor_fit():
    """Test if the preprocessor works with sample data"""
    print("\n🧪 Testing preprocessor with sample data...")
    
    # Create sample attack data
    sample_attack_data = {
        'Fwd Packet Length Mean': 95.0, 'Fwd Packet Length Std': 120.0,
        'Bwd Packet Length Mean': 0.0, 'Bwd Packet Length Std': 0.0,
        'Flow Bytes/s': 450000.0, 'Flow Packets/s': 3200.0,
        'Flow IAT Mean': 0.2, 'Flow IAT Std': 0.4,
        'Fwd IAT Mean': 0.2, 'Fwd IAT Std': 0.4,
        'Fwd Packets/s': 3200.0, 'Bwd Packets/s': 0.0,
        'Packet Length Mean': 95.0, 'Packet Length Std': 120.0,
    }
    
    # Add more features to match expected feature set
    for i in range(20):
        sample_attack_data[f'feature_{i}'] = np.random.normal(0, 1)
    
    from utils.preprocessor import DataPreprocessor
    preprocessor = DataPreprocessor()
    
    if preprocessor.load_preprocessor(os.path.join(MODELS_DIR, 'preprocessor.joblib')):
        df = pd.DataFrame([sample_attack_data])
        
        try:
            X_processed = preprocessor.prepare_real_time_features(df)
            if X_processed is not None and len(X_processed) > 0:
                print("✅ SUCCESS: Preprocessor can process real-time features!")
                print(f"   Output shape: {X_processed.shape}")
                return True
            else:
                print("❌ Preprocessor returned no features")
                return False
        except Exception as e:
            print(f"❌ Preprocessor test failed: {e}")
            return False
    else:
        print("❌ Could not load preprocessor for testing")
        return False

def create_simple_preprocessor():
    """Create a simple preprocessor if all else fails"""
    print("🛠️ Creating simple preprocessor...")
    
    # Create minimal preprocessor data
    preprocessor_data = {
        'scaler': StandardScaler(),
        'feature_selector': None,  # Skip feature selection for simplicity
        'pca': None,  # Skip PCA for simplicity
        'feature_columns': [
            'Fwd Packet Length Mean', 'Fwd Packet Length Std',
            'Bwd Packet Length Mean', 'Bwd Packet Length Std', 
            'Flow Bytes/s', 'Flow Packets/s',
            'Flow IAT Mean', 'Flow IAT Std',
            'Fwd IAT Mean', 'Fwd IAT Std',
            'Fwd Packets/s', 'Bwd Packets/s',
            'Packet Length Mean', 'Packet Length Std'
        ],
        'fitted_features': True
    }
    
    # Fit scaler with reasonable values
    dummy_data = np.random.randn(100, len(preprocessor_data['feature_columns']))
    preprocessor_data['scaler'].fit(dummy_data)
    
    # Save it
    file_path = os.path.join(MODELS_DIR, 'preprocessor.joblib')
    joblib.dump(preprocessor_data, file_path)
    
    print("✅ Simple preprocessor created!")
    print("⚠️ This uses dummy data - models may need retraining")
    
    # Test it
    test_preprocessor_fit()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Fix Preprocessor Infinity Issues')
    parser.add_argument('--action', type=str, default='quick',
                       choices=['quick', 'simple', 'test'],
                       help='Action to perform')
    
    args = parser.parse_args()
    
    if args.action == 'quick':
        success = quick_retrain_preprocessor_fixed()
        if not success:
            print("\n🔄 Quick fix failed, trying simple preprocessor...")
            create_simple_preprocessor()
    elif args.action == 'simple':
        create_simple_preprocessor()
    elif args.action == 'test':
        test_preprocessor_fit()