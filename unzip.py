#!/usr/bin/env python3
"""
Enhanced script to extract APTOS 2019 dataset and organize files properly
"""
import zipfile
import os
import shutil
from pathlib import Path

def extract_aptos_dataset():
    """Extract and organize APTOS 2019 dataset"""
    # Path to your ZIP file and extraction directory
    zip_file_path = 'aptos2019-blindness-detection.zip'
    extract_to_path = './data'
    
    # Check if ZIP file exists
    if not os.path.exists(zip_file_path):
        print(f"❌ ERROR: {zip_file_path} not found!")
        print("Please download the dataset first:")
        print("1. Go to: https://www.kaggle.com/competitions/aptos2019-blindness-detection")
        print("2. Download the dataset ZIP file")
        print("3. Place it in this directory")
        return False
    
    print(f"📦 Found dataset: {zip_file_path}")
    print(f"📁 Extracting to: {extract_to_path}")
    
    # Create the target directory if it doesn't exist
    os.makedirs(extract_to_path, exist_ok=True)
    os.makedirs('./ml/data', exist_ok=True)
    
    try:
        # Extract the ZIP file
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to_path)
        
        print(f"✅ Extracted files to {extract_to_path}")
        
        # Organize files properly
        organize_dataset_structure()
        
        # Display dataset info
        display_dataset_info()
        
        return True
        
    except Exception as e:
        print(f"❌ ERROR: Failed to extract dataset: {e}")
        return False

def organize_dataset_structure():
    """Organize extracted files into proper structure"""
    print("🗂️  Organizing dataset structure...")
    
    data_dir = Path('./data')
    ml_data_dir = Path('./ml/data')
    
    # Common file mappings
    file_mappings = [
        ('train.csv', 'train.csv'),
        ('test.csv', 'test.csv'),
        ('sample_submission.csv', 'sample_submission.csv')
    ]
    
    # Copy CSV files to both locations
    for src_name, dst_name in file_mappings:
        src_path = data_dir / src_name
        if src_path.exists():
            # Copy to ml/data
            dst_path = ml_data_dir / dst_name
            shutil.copy2(src_path, dst_path)
            print(f"📄 Copied {src_name} -> ml/data/{dst_name}")
    
    # Handle image directories
    image_dirs = ['train_images', 'test_images']
    for img_dir in image_dirs:
        src_img_dir = data_dir / img_dir
        dst_img_dir = ml_data_dir / img_dir
        
        if src_img_dir.exists():
            if not dst_img_dir.exists():
                print(f"📁 Creating symlink: ml/data/{img_dir} -> data/{img_dir}")
                try:
                    # Try to create symlink (works on Windows 10+ with dev mode)
                    dst_img_dir.symlink_to(src_img_dir.resolve(), target_is_directory=True)
                except OSError:
                    # Fallback: copy reference file
                    dst_img_dir.mkdir(exist_ok=True)
                    print(f"📝 Note: Images are in data/{img_dir}/, update paths in config if needed")

def display_dataset_info():
    """Display information about the extracted dataset"""
    print("\n" + "="*50)
    print("📊 DATASET INFORMATION")
    print("="*50)
    
    data_dir = Path('./data')
    
    # Check for CSV files
    csv_files = ['train.csv', 'test.csv', 'sample_submission.csv']
    for csv_file in csv_files:
        csv_path = data_dir / csv_file
        if csv_path.exists():
            print(f"✅ {csv_file}: Found")
        else:
            print(f"❌ {csv_file}: Not found")
    
    # Check for image directories
    image_dirs = ['train_images', 'test_images']
    for img_dir in image_dirs:
        img_path = data_dir / img_dir
        if img_path.exists():
            image_count = len(list(img_path.glob('*.jpg'))) + len(list(img_path.glob('*.png')))
            print(f"✅ {img_dir}: {image_count} images")
        else:
            print(f"❌ {img_dir}: Not found")
    
    # Check training CSV content
    train_csv = data_dir / 'train.csv'
    if train_csv.exists():
        try:
            import pandas as pd
            df = pd.read_csv(train_csv)
            print(f"\n📈 Training data: {len(df)} samples")
            print("Label distribution:")
            label_counts = df['diagnosis'].value_counts().sort_index()
            labels = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']
            for i, count in label_counts.items():
                if i < len(labels):
                    print(f"  {i} ({labels[i]}): {count} samples")
        except Exception as e:
            print(f"⚠️  Could not analyze training data: {e}")
    
    print("\n🎯 Next Steps:")
    print("1. Train model: python -m ml.train_advanced")
    print("2. Start API: python backend/start_server.py")
    print("3. Run tests: pytest tests/ -v")

def main():
    """Main function"""
    print("🏥 APTOS 2019 Dataset Extractor")
    print("="*40)
    
    if extract_aptos_dataset():
        print("\n🎉 Dataset extraction completed successfully!")
        return True
    else:
        print("\n❌ Dataset extraction failed!")
        return False

if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
