#!/usr/bin/env python3
"""
Main script to run BlackWall experiments with data cutting and hyperparameter tuning
"""

import os
import sys
from experiment_runner import ExperimentRunner

def main():
    print("🚀 BlackWall Comprehensive Experiment Runner")
    print("=" * 50)
    print("This will run experiments with:")
    print("✅ Different data cutting strategies")
    print("✅ Various sample sizes") 
    print("✅ Hyperparameter tuning")
    print("✅ Model performance comparison")
    print("=" * 50)
    
    runner = ExperimentRunner()
    
    try:
        results = runner.run_comprehensive_experiment()
        print(f"\n🎉 All experiments completed! Check the 'experiments' folder for results.")
        
    except Exception as e:
        print(f"❌ Experiment runner failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()