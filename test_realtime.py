#!/usr/bin/env python3
"""
Test script for real-time functionality
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.real_time_predictor import RealTimePredictor

def test_realtime():
    print("🧪 Testing Real-time Protection...")
    
    try:
        predictor = RealTimePredictor()
        
        if not predictor.models:
            print("❌ No models loaded. Please train models first.")
            return False
        
        print("✅ Models loaded successfully")
        print("🛡️ Starting real-time protection (press Ctrl+C to stop)...")
        
        # Test for a short period
        import time
        predictor.capture.start_capture()
        
        print("📡 Capturing network traffic for 10 seconds...")
        time.sleep(10)
        
        packets = predictor.capture.get_captured_packets()
        print(f"📦 Captured {len(packets)} packets")
        
        if packets:
            print("🔍 Processing captured packets...")
            predictor._process_captured_packets()
        
        predictor.capture.stop_capture()
        print("✅ Real-time test completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Real-time test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_realtime()