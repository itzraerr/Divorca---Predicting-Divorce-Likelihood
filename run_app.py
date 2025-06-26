#!/usr/bin/env python3
"""
Main entry point for the Divorce Prediction System
Run this script to start the Streamlit application
"""

import subprocess
import sys
import os

def main():
    """Run the Streamlit divorce prediction app"""
    try:
        # Change to the project directory
        project_dir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(project_dir)
        
        # Run the Streamlit app
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "app/divorce_app.py",
            "--server.port=8501",
            "--server.address=localhost"
        ])
        
    except KeyboardInterrupt:
        print("\nApplication stopped by user")
    except Exception as e:
        print(f"Error running application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
