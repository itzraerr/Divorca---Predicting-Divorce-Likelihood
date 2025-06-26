#!/usr/bin/env python3
"""
Model training entry point for the Divorce Prediction System
Run this script to train/retrain the divorce prediction model
"""

import subprocess
import sys
import os

def main():
    """Train the divorce prediction model"""
    try:
        # Change to the project directory
        project_dir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(project_dir)
        
        print("Starting model training...")
        
        # Run the training script
        result = subprocess.run([
            sys.executable, "src/train_divorce_model.py"
        ], capture_output=True, text=True)
        
        # Print the output
        print(result.stdout)
        if result.stderr:
            print("Errors:", result.stderr)
            
        if result.returncode == 0:
            print("Model training completed successfully!")
        else:
            print("Model training failed!")
            sys.exit(1)
        
    except Exception as e:
        print(f"Error training model: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
