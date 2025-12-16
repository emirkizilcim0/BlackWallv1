import time
import pandas as pd
from scapy.all import sniff, IP, TCP, UDP
import threading
from datetime import datetime
import numpy as np
from collections import defaultdict, deque

class LiveTrafficCapture:
    def __init__(self, interface=None):
        self.interface = interface
        self.is_capturing = False
        self.captured_packets = []
        self.max_packets = 1000
        # Flow-level state so we can build CIC-IDS-2017 style features
        self.flows = {}
        self.sniff_thread = None
        
    def start_capture(self):
        """Start packet capture on all interfaces"""
        if self.is_capturing:
            return
            
        self.is_capturing = True
        self.captured_packets = []
        self.flows = {}
        
        print("🎯 Starting packet capture on all interfaces...")
        
        # Start sniffing in a separate thread
        self.sniff_thread = threading.Thread(target=self._sniff_packets, daemon=True)
        self.sniff_thread.start()
        
    def _sniff_packets(self):
        """Background packet sniffing"""
        try:
            while self.is_capturing:
                sniff(
                    prn=self._packet_handler,
                    store=False,
                    stop_filter=lambda x: not self.is_capturing,
                    timeout=5  # restart sniff periodically so thread stays alive
                )
        except Exception as e:
            print(f"❌ Packet capture error: {e}")
            
    def _packet_handler(self, packet):
        """Process each captured packet with ALL required features"""
        if not self.is_capturing:
            return
            
        try:
            now = time.time()
            if not packet.haslayer(IP):
                return

            ip_layer = packet[IP]
            src_ip = ip_layer.src
            dst_ip = ip_layer.dst
            proto = ip_layer.proto

            if packet.haslayer(TCP):
                l4 = packet[TCP]
                src_port = l4.sport
                dst_port = l4.dport
                flags = l4.flags
                win = getattr(l4, "window", 0)
            elif packet.haslayer(UDP):
                l4 = packet[UDP]
                src_port = l4.sport
                dst_port = l4.dport
                flags = 0
                win = 0
            else:
                src_port = 0
                dst_port = 0
                flags = 0
                win = 0

            packet_len = len(packet)

            # Flow key: 4-tuple; orientation determined on first packet
            flow_key = (src_ip, dst_ip, dst_port, proto)
            reverse_key = (dst_ip, src_ip, src_port, proto)

            if flow_key in self.flows:
                flow = self.flows[flow_key]
                direction = "fwd"
            elif reverse_key in self.flows:
                flow = self.flows[reverse_key]
                direction = "bwd"
            else:
                flow = {
                    "fwd_src": src_ip,
                    "dst_port": dst_port,
                    "proto": proto,
                    "start_time": now,
                    "end_time": now,
                    "fwd_lengths": [],
                    "bwd_lengths": [],
                    "fwd_times": [],
                    "bwd_times": [],
                    "flags": defaultdict(int),
                    "init_win_fwd": win if flags else 0,
                    "init_win_bwd": 0,
                }
                self.flows[flow_key] = flow
                direction = "fwd"

            flow["end_time"] = now

            if direction == "fwd":
                flow["fwd_lengths"].append(packet_len)
                flow["fwd_times"].append(now)
                if flow["init_win_fwd"] == 0:
                    flow["init_win_fwd"] = win
            else:
                flow["bwd_lengths"].append(packet_len)
                flow["bwd_times"].append(now)
                if flow["init_win_bwd"] == 0:
                    flow["init_win_bwd"] = win

            # Track TCP flag counts when available
            if flags:
                flag_map = {
                    "F": "FIN Flag Count",
                    "S": "SYN Flag Count",
                    "R": "RST Flag Count",
                    "P": "PSH Flag Count",
                    "A": "ACK Flag Count",
                    "U": "URG Flag Count",
                    "E": "ECE Flag Count",
                }
                for bit, name in flag_map.items():
                    if bit in flags:
                        flow["flags"][name] += 1

        except Exception as e:
            print(f"❌ Packet processing error: {e}")
    
    def _compute_flow_features(self, flow):
        """Build a single-row dict with all 70 training features"""
        fwd_lens = flow["fwd_lengths"]
        bwd_lens = flow["bwd_lengths"]
        fwd_times = flow["fwd_times"]
        bwd_times = flow["bwd_times"]
        all_lens = fwd_lens + bwd_lens
        all_times = sorted(fwd_times + bwd_times)

        def safe_stats(values):
            if len(values) == 0:
                return 0, 0, 0, 0
            arr = np.array(values, dtype=float)
            return float(arr.mean()), float(arr.std(ddof=0)), float(arr.max()), float(arr.min())

        def iat_stats(times):
            if len(times) < 2:
                return 0, 0, 0, 0, 0
            diffs = np.diff(sorted(times))
            return (
                float(diffs.sum()),
                float(diffs.mean()),
                float(diffs.std(ddof=0)),
                float(diffs.max()),
                float(diffs.min()),
            )

        duration_sec = max(1e-6, flow["end_time"] - flow["start_time"])
        duration_micro = duration_sec * 1e6  # CIC-IDS uses microseconds

        total_fwd_pkts = len(fwd_lens)
        total_bwd_pkts = len(bwd_lens)
        total_fwd_bytes = sum(fwd_lens)
        total_bwd_bytes = sum(bwd_lens)
        total_pkts = total_fwd_pkts + total_bwd_pkts
        total_bytes = total_fwd_bytes + total_bwd_bytes

        fwd_mean, fwd_std, fwd_max, fwd_min = safe_stats(fwd_lens)
        bwd_mean, bwd_std, bwd_max, bwd_min = safe_stats(bwd_lens)
        pkt_mean, pkt_std, pkt_max, pkt_min = safe_stats(all_lens)
        pkt_var = float(np.var(all_lens)) if len(all_lens) > 0 else 0.0

        flow_iat = iat_stats(all_times)
        fwd_iat = iat_stats(fwd_times)
        bwd_iat = iat_stats(bwd_times)

        flow_bytes_s = total_bytes / duration_sec if duration_sec > 0 else total_bytes
        flow_pkts_s = total_pkts / duration_sec if duration_sec > 0 else total_pkts
        fwd_pkts_s = total_fwd_pkts / duration_sec if duration_sec > 0 else total_fwd_pkts
        bwd_pkts_s = total_bwd_pkts / duration_sec if duration_sec > 0 else total_bwd_pkts

        down_up_ratio = total_bwd_pkts / max(total_fwd_pkts, 1)

        features = {
            'Destination Port': flow.get("dst_port", 0),
            'Flow Duration': duration_micro,
            'Total Fwd Packets': total_fwd_pkts,
            'Total Backward Packets': total_bwd_pkts,
            'Total Length of Fwd Packets': total_fwd_bytes,
            'Total Length of Bwd Packets': total_bwd_bytes,
            'Fwd Packet Length Max': fwd_max,
            'Fwd Packet Length Min': fwd_min,
            'Fwd Packet Length Mean': fwd_mean,
            'Fwd Packet Length Std': fwd_std,
            'Bwd Packet Length Max': bwd_max,
            'Bwd Packet Length Min': bwd_min,
            'Bwd Packet Length Mean': bwd_mean,
            'Bwd Packet Length Std': bwd_std,
            'Flow Bytes/s': flow_bytes_s,
            'Flow Packets/s': flow_pkts_s,
            'Flow IAT Mean': flow_iat[1],
            'Flow IAT Std': flow_iat[2],
            'Flow IAT Max': flow_iat[3],
            'Flow IAT Min': flow_iat[4],
            'Fwd IAT Total': fwd_iat[0],
            'Fwd IAT Mean': fwd_iat[1],
            'Fwd IAT Std': fwd_iat[2],
            'Fwd IAT Max': fwd_iat[3],
            'Fwd IAT Min': fwd_iat[4],
            'Bwd IAT Total': bwd_iat[0],
            'Bwd IAT Mean': bwd_iat[1],
            'Bwd IAT Std': bwd_iat[2],
            'Bwd IAT Max': bwd_iat[3],
            'Bwd IAT Min': bwd_iat[4],
            'Fwd PSH Flags': flow["flags"].get("PSH Flag Count", 0),
            'Fwd Header Length': 0,
            'Bwd Header Length': 0,
            'Fwd Packets/s': fwd_pkts_s,
            'Bwd Packets/s': bwd_pkts_s,
            'Min Packet Length': pkt_min,
            'Max Packet Length': pkt_max,
            'Packet Length Mean': pkt_mean,
            'Packet Length Std': pkt_std,
            'Packet Length Variance': pkt_var,
            'FIN Flag Count': flow["flags"].get("FIN Flag Count", 0),
            'SYN Flag Count': flow["flags"].get("SYN Flag Count", 0),
            'RST Flag Count': flow["flags"].get("RST Flag Count", 0),
            'PSH Flag Count': flow["flags"].get("PSH Flag Count", 0),
            'ACK Flag Count': flow["flags"].get("ACK Flag Count", 0),
            'URG Flag Count': flow["flags"].get("URG Flag Count", 0),
            'ECE Flag Count': flow["flags"].get("ECE Flag Count", 0),
            'Down/Up Ratio': down_up_ratio,
            'Average Packet Size': pkt_mean,
            'Avg Fwd Segment Size': fwd_mean,
            'Avg Bwd Segment Size': bwd_mean,
            'Fwd Header Length.1': 0,
            'Subflow Fwd Packets': total_fwd_pkts,
            'Subflow Fwd Bytes': total_fwd_bytes,
            'Subflow Bwd Packets': total_bwd_pkts,
            'Subflow Bwd Bytes': total_bwd_bytes,
            'Init_Win_bytes_forward': flow.get("init_win_fwd", 0),
            'Init_Win_bytes_backward': flow.get("init_win_bwd", 0),
            'act_data_pkt_fwd': total_fwd_pkts,
            'min_seg_size_forward': fwd_min,
            'Active Mean': duration_sec,  # coarse approximation
            'Active Std': 0,
            'Active Max': duration_sec,
            'Active Min': 0,
            'Idle Mean': 0,
            'Idle Std': 0,
            'Idle Max': 0,
            'Idle Min': 0,
            'Fwd URG Flags': flow["flags"].get("URG Flag Count", 0),
            'CWE Flag Count': 0,
        }

        # Meta fields for debugging/UI
        features['src_ip'] = flow.get("fwd_src", "")
        return features
    
    def get_captured_packets(self, max_packets=100):
        """Return flow-level feature rows (aligned to training schema)"""
        flow_items = list(self.flows.items())
        packets = []
        for key, flow in flow_items[:max_packets]:
            packets.append(self._compute_flow_features(flow))
            # Remove returned flows to avoid duplicate predictions
            self.flows.pop(key, None)

        return packets
    
    def stop_capture(self):
        """Stop packet capture"""
        self.is_capturing = False
        print("🛑 Stopped live capture")
