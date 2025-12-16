#!/usr/bin/env python3
"""
Feature name mapping utility to fix mismatches between training and real-time features
"""

class FeatureMapper:
    def __init__(self):
        # Map from real-time feature names to training feature names
        self.feature_mapping = {
            # Map underscore names to spaced names
            'dst_port': 'Destination Port',
            'flow_duration': 'Flow Duration',
            'total_fwd_packets': 'Total Fwd Packets',
            'total_backward_packets': 'Total Backward Packets',
            'total_length_of_fwd_packets': 'Total Length of Fwd Packets',
            'total_length_of_bwd_packets': 'Total Length of Bwd Packets',
            'fwd_packet_length_max': 'Fwd Packet Length Max',
            'fwd_packet_length_min': 'Fwd Packet Length Min',
            'fwd_packet_length_mean': 'Fwd Packet Length Mean',
            'fwd_packet_length_std': 'Fwd Packet Length Std', 
            'bwd_packet_length_max': 'Bwd Packet Length Max',
            'bwd_packet_length_min': 'Bwd Packet Length Min',
            'bwd_packet_length_mean': 'Bwd Packet Length Mean',
            'bwd_packet_length_std': 'Bwd Packet Length Std',
            'flow_bytes_s': 'Flow Bytes/s',
            'flow_packets_s': 'Flow Packets/s',
            'flow_iat_mean': 'Flow IAT Mean',
            'flow_iat_std': 'Flow IAT Std',
            'flow_iat_max': 'Flow IAT Max',
            'flow_iat_min': 'Flow IAT Min',
            'fwd_iat_mean': 'Fwd IAT Mean',
            'fwd_iat_std': 'Fwd IAT Std',
            'fwd_iat_max': 'Fwd IAT Max',
            'fwd_iat_min': 'Fwd IAT Min',
            'bwd_iat_mean': 'Bwd IAT Mean', 
            'bwd_iat_std': 'Bwd IAT Std',
            'bwd_iat_max': 'Bwd IAT Max',
            'bwd_iat_min': 'Bwd IAT Min',
            'fwd_iat_total': 'Fwd IAT Total',
            'bwd_iat_total': 'Bwd IAT Total',
            'fwd_packets_s': 'Fwd Packets/s',
            'bwd_packets_s': 'Bwd Packets/s',
            'packet_length_mean': 'Packet Length Mean',
            'packet_length_std': 'Packet Length Std',
            'packet_length_variance': 'Packet Length Variance',
            'average_packet_size': 'Average Packet Size',
            'avg_fwd_segment_size': 'Avg Fwd Segment Size',
            'avg_bwd_segment_size': 'Avg Bwd Segment Size',
            'active_mean': 'Active Mean',
            'active_std': 'Active Std',
            'active_max': 'Active Max',
            'active_min': 'Active Min',
            'idle_mean': 'Idle Mean',
            'idle_std': 'Idle Std',
            'idle_max': 'Idle Max',
            'idle_min': 'Idle Min',
            'fin_flag_count': 'FIN Flag Count',
            'syn_flag_count': 'SYN Flag Count',
            'rst_flag_count': 'RST Flag Count',
            'psh_flag_count': 'PSH Flag Count',
            'ack_flag_count': 'ACK Flag Count',
            'urg_flag_count': 'URG Flag Count',
            'ece_flag_count': 'ECE Flag Count',
            'down_up_ratio': 'Down/Up Ratio',
            'fwd_header_length': 'Fwd Header Length',
            'bwd_header_length': 'Bwd Header Length',
            'fwd_header_length_1': 'Fwd Header Length.1',
            'subflow_fwd_packets': 'Subflow Fwd Packets',
            'subflow_fwd_bytes': 'Subflow Fwd Bytes',
            'subflow_bwd_packets': 'Subflow Bwd Packets',
            'subflow_bwd_bytes': 'Subflow Bwd Bytes',
            'init_win_bytes_forward': 'Init_Win_bytes_forward',
            'init_win_bytes_backward': 'Init_Win_bytes_backward',
            'act_data_pkt_fwd': 'act_data_pkt_fwd',
            'min_seg_size_forward': 'min_seg_size_forward',
            'fwd_urg_flags': 'Fwd URG Flags',
            'cwe_flag_count': 'CWE Flag Count',
            
            # Also map common variations
            'flow_bytes_per_sec': 'Flow Bytes/s',
            'flow_packets_per_sec': 'Flow Packets/s',
            'fwd_packets_per_sec': 'Fwd Packets/s',
            'bwd_packets_per_sec': 'Bwd Packets/s',
        }
    
    def map_features(self, df):
        """Map DataFrame columns to training feature names"""
        mapped_df = df.copy()

        def normalize(col):
            return col.strip().replace(' ', '_').lower()

        # Build a lookup from normalized incoming name to original
        normalized = {normalize(c): c for c in mapped_df.columns}

        for current_name, training_name in self.feature_mapping.items():
            # Try exact, then normalized match
            if current_name in mapped_df.columns:
                src_col = current_name
            elif current_name in normalized:
                src_col = normalized[current_name]
            else:
                continue

            mapped_df[training_name] = mapped_df[src_col]

        return mapped_df
    
    def get_training_features(self):
        """Get the list of expected training features"""
        return list(set(self.feature_mapping.values()))
