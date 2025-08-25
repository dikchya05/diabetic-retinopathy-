#!/usr/bin/env python3
"""
Script to download APTOS 2019 dataset using Kaggle API
Requirements:
1. Install kaggle: pip install kaggle
2. Setup API credentials: https://github.com/Kaggle/kaggle-api#api-credentials
3. Place kaggle.json in ~/.kaggle/ or C:\Users\{username}\.kaggle\
"""
import os
import subprocess
import sys
from pathlib import Path

def check_kaggle_setup():
    """Check if Kaggle API is properly configured"""
    try:
        result = subprocess.run(['kaggle', '--version'], 
                              capture_output=True, text=True, check=True)
        print(f"✅ Kaggle API found: {result.stdout.strip()}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Kaggle API not found. Please install: pip install kaggle")
        return False

def check_api_credentials():
    """Check if API credentials are configured"""
    try:
        result = subprocess.run(['kaggle', 'competitions', 'list'], 
                              capture_output=True, text=True, check=True)
        print("✅ Kaggle API credentials are working")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ API credentials error: {e.stderr}")
        print("Please setup credentials: https://github.com/Kaggle/kaggle-api#api-credentials")
        return False

def download_aptos_dataset():
    """Download APTOS 2019 dataset"""
    print("📥 Downloading APTOS 2019 Blindness Detection dataset...")
    
    try:
        # Download dataset
        result = subprocess.run([
            'kaggle', 'competitions', 'download', 
            '-c', 'aptos2019-blindness-detection',
            '-p', '.'  # Download to current directory
        ], capture_output=True, text=True, check=True)
        
        print("✅ Dataset downloaded successfully!")
        print(result.stdout)
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Download failed: {e.stderr}")
        return False

def setup_data_structure():
    """Create proper directory structure"""
    print("📁 Setting up data directory structure...")
    
    # Create directories
    data_dir = Path('data')
    ml_data_dir = Path('ml/data')
    
    data_dir.mkdir(exist_ok=True)
    ml_data_dir.mkdir(parents=True, exist_ok=True)
    
    print("✅ Directory structure created")

def main():
    print("🏥 APTOS 2019 Dataset Downloader")
    print("=" * 40)
    
    # Check prerequisites
    if not check_kaggle_setup():
        print("\n📋 Setup Instructions:")
        print("1. Install Kaggle API: pip install kaggle")
        print("2. Create Kaggle account and get API token")
        print("3. Place kaggle.json in ~/.kaggle/ (Linux/Mac) or C:\\Users\\{username}\\.kaggle\\ (Windows)")
        return False
    
    if not check_api_credentials():
        return False
    
    # Setup directories
    setup_data_structure()
    
    # Download dataset
    if download_aptos_dataset():
        print("\n🎉 Download completed!")
        print("\n📋 Next steps:")
        print("1. Run: python unzip.py")
        print("2. Check data structure in ./data/ directory")
        print("3. Start training: python -m ml.train_advanced")
        return True
    else:
        print("\n❌ Download failed. Try manual download from:")
        print("https://www.kaggle.com/competitions/aptos2019-blindness-detection")
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)