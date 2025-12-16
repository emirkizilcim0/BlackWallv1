import os

# Dataset configuration for CIC-IDS-2017
DATASET_CONFIG = {
    'dataset_name': 'CIC-IDS-2017',
    'data_directory': os.path.join('data', 'CIC_IDS_2017'),
    'label_column': 'Label',
    'normal_label': 'BENIGN',
    'attack_labels': [
        'DDoS', 'PortScan', 'Bot', 'Infiltration', 
        'Web Attack', 'Brute Force', 'DoS', 'FTP-Patator',
        'SSH-Patator', 'Heartbleed'
    ],
    'timestamp_columns': ['Timestamp'],
    'ip_columns': ['Source IP', 'Destination IP', 'Src IP', 'Dst IP'],
    'expected_files': [
        'Monday-WorkingHours.pcap_ISCX.csv',
        'Tuesday-WorkingHours.pcap_ISCX.csv',
        'Wednesday-workingHours.pcap_ISCX.csv',
        'Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv',
        'Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv',
        'Friday-WorkingHours-Morning.pcap_ISCX.csv',
        'Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv',
        'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv'
    ]
}

# Model configuration
MODEL_CONFIG = {
    'test_size': 0.2,
    'random_state': 42,
    'sample_fraction': 0.1,
    'models_to_train': [
        'logistic_regression',
        'gaussian_nb', 
        'random_forest',
        'isolation_forest'
    ]
}

# Data Cutting Configuration
DATA_CUTTING_CONFIG = {
    'enabled': True,
    'strategies': ['random', 'time_based', 'attack_focused', 'balanced'],
    'max_samples_per_class': 10000,  # Limit samples per class for balancing
    'time_window_hours': 24,  # For time-based cutting
}

# Hyperparameter Tuning Configuration
TUNING_CONFIG = {
    'enabled': True,
    'n_iter': 10,  # Number of random search iterations
    'cv_folds': 3,  # Cross-validation folds
    'scoring': 'f1_weighted',
}

# Individual Model Hyperparameters
HYPERPARAMETERS = {
    'logistic_regression': {
        'C': [0.1, 1.0, 10.0, 100.0],
        'solver': ['liblinear', 'lbfgs'],
        'max_iter': [100, 200, 500]
    },
    'gaussian_nb': {
        'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6]
    },
    'random_forest': {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    'isolation_forest': {
        'n_estimators': [50, 100, 200],
        'contamination': [0.05, 0.1, 0.15, 0.2],
        'max_samples': [0.5, 0.8, 1.0]
    }
}

# Path configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
CIC_DATA_DIR = os.path.join(DATA_DIR, 'CIC_IDS_2017')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
EXPERIMENTS_DIR = os.path.join(BASE_DIR, 'experiments')

# Create directories
for directory in [DATA_DIR, CIC_DATA_DIR, MODELS_DIR, RESULTS_DIR, EXPERIMENTS_DIR]:
    os.makedirs(directory, exist_ok=True)