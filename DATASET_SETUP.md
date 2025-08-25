# 📊 Dataset Setup Guide

## APTOS 2019 Blindness Detection Dataset

This project uses the **APTOS 2019 Blindness Detection** dataset from Kaggle, which contains retinal fundus images for diabetic retinopathy classification.

## 🎯 Dataset Overview

- **Competition**: APTOS 2019 Blindness Detection
- **Source**: Kaggle
- **Images**: ~5,590 retinal fundus photographs  
- **Format**: High-resolution JPEG images
- **Labels**: 5-class severity classification (0-4)

### Severity Classes:
- **0**: No DR (No Diabetic Retinopathy)
- **1**: Mild DR  
- **2**: Moderate DR
- **3**: Severe DR
- **4**: Proliferative DR

## 🔗 Dataset URL
https://www.kaggle.com/competitions/aptos2019-blindness-detection

---

## 📥 Download Methods

### Method 1: Manual Download (Recommended)

1. **Visit Kaggle**:
   - Go to: https://www.kaggle.com/competitions/aptos2019-blindness-detection
   - Create/login to your Kaggle account

2. **Download Dataset**:
   - Click the "Data" tab
   - Click "Download All" button
   - Save `aptos2019-blindness-detection.zip` to project root

3. **Extract Dataset**:
   ```bash
   python unzip.py
   ```

### Method 2: Kaggle API (Advanced Users)

1. **Setup Kaggle API**:
   ```bash
   pip install kaggle
   ```

2. **Configure Credentials**:
   - Go to Kaggle → Account → Create New API Token
   - Download `kaggle.json`
   - Place in:
     - **Windows**: `C:\Users\{username}\.kaggle\kaggle.json`
     - **Linux/Mac**: `~/.kaggle/kaggle.json`

3. **Download Using Script**:
   ```bash
   python download_dataset.py
   ```
   
   **Or manually**:
   ```bash
   kaggle competitions download -c aptos2019-blindness-detection
   python unzip.py
   ```

### Method 3: Windows Batch Script

For Windows users:
```bash
download_dataset.bat
```

---

## 📁 Expected Directory Structure

After downloading and extracting, you should have:

```
diabetic-retinopathy-/
├── data/                          # Extracted dataset
│   ├── train_images/              # Training images (~3,662 images)
│   │   ├── 000c1434d8d7.jpg
│   │   ├── 001639a390f0.jpg
│   │   └── ...
│   ├── test_images/               # Test images (~1,928 images)
│   │   ├── 0005cfc8afb6.jpg
│   │   └── ...
│   ├── train.csv                  # Training labels
│   ├── test.csv                   # Test image IDs
│   └── sample_submission.csv      # Submission format
├── ml/
│   └── data/                      # Symlinked/copied for ML scripts
│       ├── train_images/          # -> ../../../data/train_images/
│       ├── test_images/           # -> ../../../data/test_images/
│       ├── train.csv
│       ├── test.csv
│       └── sample_submission.csv
└── aptos2019-blindness-detection.zip  # Original download
```

---

## 🔍 Dataset Verification

### Verify Download
```bash
python unzip.py
```

The script will show:
- ✅ Files found/extracted
- 📊 Dataset statistics
- 📈 Label distribution
- 🎯 Next steps

### Manual Verification

Check if you have these files:
```bash
# Windows
dir data\
dir data\train_images\
dir data\test_images\

# Linux/Mac  
ls data/
ls data/train_images/
ls data/test_images/
```

### Expected Counts:
- **train.csv**: ~3,662 rows
- **train_images/**: ~3,662 JPG files
- **test_images/**: ~1,928 JPG files

---

## 📊 Data Analysis

### Training Data Distribution
The APTOS dataset typically has this distribution:

| Class | Severity Level | Count | Percentage |
|-------|---------------|-------|-----------|
| 0     | No DR         | ~1,805| ~49.3%    |
| 1     | Mild          | ~370  | ~10.1%    |  
| 2     | Moderate      | ~999  | ~27.3%    |
| 3     | Severe        | ~193  | ~5.3%     |
| 4     | Proliferative | ~295  | ~8.1%     |

**Note**: This is an imbalanced dataset - most images are "No DR" class.

---

## 🚨 Troubleshooting

### Common Issues:

#### 1. "ZIP file not found"
- Ensure `aptos2019-blindness-detection.zip` is in project root
- Check filename spelling exactly

#### 2. "Kaggle API credentials not found"
- Download `kaggle.json` from Kaggle account settings
- Place in correct directory with proper permissions
- Check: `kaggle competitions list`

#### 3. "Access denied" or "Competition rules"
- You must accept competition rules on Kaggle website
- Visit the competition page and click "I Understand and Accept"

#### 4. "Slow download"
- Large dataset (~900MB zipped, ~2.3GB extracted)  
- Use stable internet connection
- Consider manual download for reliability

#### 5. "Extraction failed"
- Check available disk space (~3GB needed)
- Ensure write permissions in project directory
- Try manual extraction with system tools

---

## 🔧 Configuration

### Update Paths
If you place data elsewhere, update these files:

**ml/train_advanced.py**:
```python
DATA_DIR = "path/to/your/data"
```

**backend/app/config.py**:
```python
DATA_PATH = "path/to/your/data"
```

### Environment Variables
You can also set:
```bash
export DATA_DIR="/path/to/data"
```

---

## ✅ Verification Checklist

Before training, ensure:

- [ ] `aptos2019-blindness-detection.zip` downloaded
- [ ] `python unzip.py` completed successfully  
- [ ] `data/train_images/` contains ~3,662 JPG files
- [ ] `data/train.csv` exists with diagnosis column
- [ ] `data/test_images/` contains ~1,928 JPG files
- [ ] No extraction errors in console
- [ ] Dataset stats look reasonable

---

## 🎯 Next Steps

Once dataset is ready:

1. **Train Model**:
   ```bash
   python -m ml.train_advanced
   ```

2. **Start API Server**:
   ```bash
   python backend/start_server.py
   ```

3. **Run Tests**:
   ```bash
   pytest tests/ -v
   ```

4. **Explore Data** (optional):
   ```bash
   jupyter lab
   ```

---

## 📚 Dataset Papers & References

- **Original Paper**: "Diabetic Retinopathy Detection via Deep Convolutional Networks"
- **Competition**: APTOS 2019 Blindness Detection Challenge  
- **Medical Context**: Aravind Eye Care System dataset
- **Image Source**: Fundus photography from multiple sources

**Citation**: APTOS 2019 Blindness Detection. (2019). Kaggle. https://kaggle.com/competitions/aptos2019-blindness-detection

---

## 🆘 Need Help?

If you encounter issues:

1. **Check this guide** for common solutions
2. **Run**: `python unzip.py` for detailed error messages  
3. **Verify**: Internet connection and disk space
4. **Try**: Manual download from Kaggle website
5. **Check**: Kaggle API credentials and competition rules

**Manual Download Link**: https://www.kaggle.com/competitions/aptos2019-blindness-detection/data