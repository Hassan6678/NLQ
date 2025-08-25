#!/usr/bin/env python3
"""
Simple startup script for the NLQ Web Application
"""

import os
import sys
from pathlib import Path

def check_requirements():
    """Check if required files and directories exist."""
    required_files = [
        'app.py',
        'templates/view.html',
        'config.py',
        'helper/__init__.py'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ Missing required files:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        print("\nPlease ensure all files are present before running the application.")
        return False
    
    return True

def main():
    """Main startup function."""
    print("🚀 Starting NLQ Web Application...")
    
    # Check requirements
    if not check_requirements():
        sys.exit(1)
    
    # Add current directory to Python path
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    try:
        from app import app
        
        print("✅ Application loaded successfully!")
        print("🌐 Starting web server...")
        print("📱 Open your browser and go to: http://localhost:5000")
        print("⏹️  Press Ctrl+C to stop the server")
        print("-" * 50)
        
        # Run the Flask app
        app.run(
            debug=True,
            host='0.0.0.0',
            port=5000,
            use_reloader=False  # Disable reloader to avoid duplicate processes
        )
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please ensure all dependencies are installed:")
        print("   pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Failed to start application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
