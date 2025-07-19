#!/usr/bin/env python3
"""
Setup script for CSV ML Analyzer
This script helps set up the environment and install dependencies
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n{description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"Error: {e.stderr}")
        return False

def main():
    print("🚀 CSV ML Analyzer Setup Script")
    print("=" * 40)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    
    print(f"✅ Python version: {sys.version.split()[0]}")
    
    # Check if virtual environment should be created
    create_venv = input("\n📦 Create virtual environment? (y/N): ").lower().strip() == 'y'
    
    if create_venv:
        if not run_command("python -m venv venv", "Creating virtual environment"):
            return False
        
        # Activate virtual environment (Windows)
        if os.name == 'nt':
            activate_cmd = r"venv\Scripts\activate"
            pip_cmd = r"venv\Scripts\pip"
        else:
            activate_cmd = "source venv/bin/activate"
            pip_cmd = "venv/bin/pip"
        
        print(f"\n📝 To activate the virtual environment later, run:")
        print(f"   {activate_cmd}")
    else:
        pip_cmd = "pip"
    
    # Install requirements
    if not run_command(f"{pip_cmd} install -r requirements.txt", "Installing Python packages"):
        return False
    
    # Create necessary directories
    directories = ['uploads', 'models', 'static', 'templates']
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"✅ Created directory: {directory}")
    
    print("\n🎉 Setup completed successfully!")
    print("\n📋 Next steps:")
    if create_venv:
        print(f"1. Activate virtual environment: {activate_cmd}")
    print("2. Run the application: python app.py")
    print("3. Open your browser to: http://localhost:5000")
    print("\n📖 For more information, see README.md")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
