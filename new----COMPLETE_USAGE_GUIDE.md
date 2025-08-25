# 🏥 Complete Usage Guide - Diabetic Retinopathy Detection System

## 📋 Table of Contents

1. [System Overview](#system-overview)
2. [Prerequisites](#prerequisites)
3. [Initial Setup](#initial-setup)
4. [Dataset Setup](#dataset-setup)
5. [Training Models](#training-models)
6. [Running the System](#running-the-system)
7. [API Usage](#api-usage)
8. [Frontend Usage](#frontend-usage)
9. [Testing](#testing)
10. [Advanced Features](#advanced-features)
11. [Troubleshooting](#troubleshooting)
12. [File Structure](#file-structure)

---

## 🎯 System Overview

This is a comprehensive AI-powered diabetic retinopathy detection system featuring:

- **Advanced ML Models**: Vision Transformers, CNNs with attention, ensemble models
- **Production API**: FastAPI backend with health checks and error handling
- **Modern Frontend**: Next.js with medical-grade UI components
- **Complete Pipeline**: Data preprocessing → Training → Inference → Evaluation
- **Clinical Metrics**: Sensitivity, specificity, AUC-ROC, confusion matrices
- **Explainable AI**: Grad-CAM visualizations for model interpretability

---

## 🔧 Prerequisites

### System Requirements
- **OS**: Windows 10/11, macOS, or Linux
- **Python**: 3.8 or higher (3.13.7 recommended)
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 10GB free space for dataset and models
- **GPU**: Optional but recommended (CUDA-compatible)

### Required Software
1. **Python** with pip
2. **Node.js** (18+ for frontend)
3. **Git** (for version control)
4. **Kaggle account** (for dataset download)

---

## 🚀 Initial Setup

### Step 1: Environment Setup

#### Option A: Automatic Setup (Recommended)
```bash
# Run the automated setup script
python setup_project.py
```

#### Option B: Manual Setup
```bash
# 1. Create virtual environment
python -m venv venv

# 2. Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. Install additional ML packages
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install timm albumentations grad-cam plotly fastapi uvicorn pytest
```

### Step 2: Verify Installation
```bash
# Test core components
python -c "
import torch
import timm
import albumentations
import fastapi
print('✅ All packages installed successfully!')
print(f'PyTorch: {torch.__version__}')
print(f'Timm: {timm.__version__}')
"
```

### Step 3: Frontend Setup
```bash
# Navigate to frontend directory
cd frontend

# Install Node.js dependencies
npm install

# Build the frontend
npm run build
```

---

## 📊 Dataset Setup

### Step 1: Get Kaggle API Credentials
1. Go to [Kaggle Account Settings](https://www.kaggle.com/account)
2. Click "Create New API Token"
3. Download `kaggle.json`
4. Place it in:
   - **Windows**: `C:\Users\{username}\.kaggle\kaggle.json`
   - **Linux/macOS**: `~/.kaggle/kaggle.json`

### Step 2: Download Dataset

#### Option A: Automated Download (Windows)
```batch
# Run the batch script
download_dataset.bat
```

#### Option B: Python Script
```bash
# Download APTOS 2019 dataset
python download_dataset.py
```

#### Option C: Manual Download
1. Visit: https://www.kaggle.com/competitions/aptos2019-blindness-detection
2. Accept competition rules
3. Download dataset ZIP file
4. Save as `aptos2019-blindness-detection.zip` in project root

### Step 3: Extract Dataset
```bash
# Extract and organize dataset
python unzip.py
```

### Step 4: Analyze Dataset (Optional)
```bash
# Generate comprehensive dataset analysis
python -m ml.dataset_analyzer
```

This creates:
- **Dataset statistics** and class distribution
- **Image quality analysis**
- **Interactive visualizations**
- **HTML report** in `ml/dataset_analysis/`

---

## 🤖 Training Models

### Step 1: Basic Training
```bash
# Train with default settings (EfficientNet-B0)
python -m ml.train_advanced
```

### Step 2: Advanced Training Options

#### Train Specific Model Architecture
```bash
# Vision Transformer
python -m ml.train_advanced --model-type vision_transformer --epochs 50

# Advanced CNN with attention
python -m ml.train_advanced --model-type advanced_cnn --backbone efficientnet_b2

# Ensemble model
python -m ml.train_advanced --model-type ensemble --epochs 30
```

#### Train with Model Recipes
```bash
# Lightweight model (for resource constraints)
python -m ml.train_advanced --recipe lightweight --epochs 20

# High performance ensemble
python -m ml.train_advanced --recipe high_performance --epochs 50

# Hybrid CNN-Transformer
python -m ml.train_advanced --recipe hybrid_best --epochs 40
```

#### Cross-Validation Training
```bash
# 5-fold cross-validation
python -m ml.train_advanced --use-cv --n-folds 5 --epochs 30
```

### Step 3: Monitor Training

Training creates:
- **Model checkpoints** in `ml/checkpoints/`
- **Training logs** and metrics
- **Tensorboard logs** (view with `tensorboard --logdir ml/logs`)
- **Evaluation reports** after each epoch

### Step 4: Model Evaluation
```bash
# Evaluate trained model
python -m ml.evaluation --model-path ml/checkpoints/best_model.pth
```

Generates:
- **Confusion matrices** and ROC curves
- **Clinical metrics** (sensitivity, specificity)
- **Interactive dashboard** (`ml/evaluation_results/evaluation_dashboard.html`)
- **Detailed JSON reports**

---

## 🖥️ Running the System

### Step 1: Start Backend API
```bash
# Start FastAPI server
python backend/start_server.py

# Or using uvicorn directly
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```

API will be available at:
- **Main API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

### Step 2: Start Frontend (Optional)
```bash
# Navigate to frontend
cd frontend

# Start development server
npm run dev
```

Frontend will be available at: http://localhost:3000

### Step 3: Docker Deployment (Production)
```bash
# Build and run with Docker Compose
docker-compose up --build

# Or build individual services
docker build -t dr-detection .
docker run -p 8000:8000 dr-detection
```

---

## 🔌 API Usage

### Health Check
```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": 1703123456.789,
  "version": "1.0.0",
  "model": {
    "available": true,
    "device": "cpu"
  }
}
```

### Prediction Endpoint
```bash
# Upload image for prediction
curl -X POST "http://localhost:8000/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@path/to/retinal_image.jpg"
```

**Response:**
```json
{
  "prediction": {
    "class": 2,
    "confidence": 0.892,
    "severity": "Moderate DR",
    "probabilities": [0.023, 0.045, 0.892, 0.035, 0.005]
  },
  "gradcam": "data:image/png;base64,iVBOR...",
  "recommendations": [
    "Refer to ophthalmologist within 1 month",
    "Monitor blood glucose levels"
  ]
}
```

### Python API Client
```python
import requests

# Load image
with open('retinal_image.jpg', 'rb') as f:
    files = {'file': f}
    response = requests.post('http://localhost:8000/predict', files=files)

result = response.json()
print(f"Prediction: {result['prediction']['severity']}")
print(f"Confidence: {result['prediction']['confidence']:.2%}")
```

---

## 🎨 Frontend Usage

### Features
1. **Image Upload**: Drag & drop or click to upload retinal images
2. **Real-time Prediction**: Instant AI analysis with confidence scores
3. **Grad-CAM Visualization**: See what the AI is looking at
4. **Probability Charts**: Interactive bar charts showing class probabilities
5. **Medical Recommendations**: Clinical guidance based on severity
6. **Professional UI**: Medical-grade interface with proper disclaimers

### Usage Steps
1. **Open Frontend**: Navigate to http://localhost:3000
2. **Upload Image**: Click upload area or drag retinal image
3. **View Results**: 
   - Severity classification
   - Confidence percentage
   - Probability distribution
   - Grad-CAM heatmap
   - Clinical recommendations
4. **Export Results**: Download predictions and visualizations

---

## 🧪 Testing

### Run All Tests
```bash
# Complete test suite
pytest tests/ -v

# With coverage report
pytest tests/ --cov=ml --cov=backend --cov-report=html
```

### Specific Test Categories

#### Model Tests
```bash
# Test model creation and training
pytest tests/test_models.py -v
```

#### API Tests
```bash
# Test backend endpoints
pytest tests/test_backend.py -v
```

#### Data Tests
```bash
# Test data preprocessing
pytest tests/test_data_utils.py -v
```

#### Evaluation Tests
```bash
# Test evaluation metrics
pytest tests/test_evaluation.py -v
```

### Manual Testing
```bash
# Test complete workflow
python -c "
import sys
sys.path.append('.')

# Test data preprocessing
from ml.data_preprocessing import DataPreprocessor
preprocessor = DataPreprocessor()
print('✅ Data preprocessing working')

# Test model creation  
from ml.models.model import create_model
model = create_model()
print('✅ Model creation working')

# Test API
from fastapi.testclient import TestClient
from backend.app.main import app
client = TestClient(app)
response = client.get('/health')
print(f'✅ API health check: {response.status_code}')
"
```

---

## 🚀 Advanced Features

### Custom Model Training

#### Create Custom Configuration
```python
# training_config.py
config = {
    'model_type': 'advanced_cnn',
    'backbone': 'efficientnet_b3',
    'epochs': 100,
    'batch_size': 16,
    'learning_rate': 1e-4,
    'use_mixed_precision': True,
    'advanced_preprocessing': True,
    'use_class_weights': True,
    'scheduler': 'cosine'
}
```

#### Train with Custom Config
```python
from ml.train_advanced import AdvancedTrainer

trainer = AdvancedTrainer(config)
metrics = trainer.train()
```

### Model Ensembles

#### Create Ensemble Configuration
```python
ensemble_config = {
    'model_configs': [
        {'type': 'advanced_cnn', 'backbone': 'efficientnet_b2'},
        {'type': 'vision_transformer', 'model_name': 'vit_base_patch16_224'},
        {'type': 'timm', 'name': 'resnet101'}
    ]
}
```

#### Use Pre-configured Recipes
```python
from ml.models.advanced_architectures import get_model_recipe

# Get high-performance ensemble
model = get_model_recipe('high_performance', num_classes=5)

# Get lightweight model
lightweight_model = get_model_recipe('lightweight', num_classes=5)
```

### Custom Preprocessing

#### Available Preprocessing Modes
```python
from ml.data_preprocessing import DataPreprocessor

# Standard preprocessing
preprocessor = DataPreprocessor(preprocess_mode='standard')

# CLAHE (Contrast Limited Adaptive Histogram Equalization)
preprocessor = DataPreprocessor(preprocess_mode='clahe')

# Ben Graham's preprocessing
preprocessor = DataPreprocessor(preprocess_mode='ben_graham')

# Crop and resize
preprocessor = DataPreprocessor(preprocess_mode='crop_resize')
```

### Evaluation and Analysis

#### Generate Comprehensive Reports
```python
from ml.evaluation import ModelEvaluator

evaluator = ModelEvaluator()
metrics = evaluator.evaluate_model(model, test_loader, device='cuda')

# Generates:
# - confusion_matrix.png
# - roc_curves.png  
# - evaluation_dashboard.html
# - detailed_results.json
```

#### Dataset Analysis
```python
from ml.dataset_analyzer import DatasetAnalyzer

analyzer = DatasetAnalyzer()
results_path = analyzer.generate_comprehensive_report()

# Creates interactive dashboard and HTML report
```

---

## 🛠️ Troubleshooting

### Common Issues

#### 1. Kaggle API Issues
**Problem**: `OSError: Could not find kaggle.json`
```bash
# Solution: Set up Kaggle credentials
mkdir -p ~/.kaggle  # Linux/macOS
# or
mkdir %USERPROFILE%\.kaggle  # Windows

# Copy kaggle.json to the directory
# Set permissions (Linux/macOS only)
chmod 600 ~/.kaggle/kaggle.json
```

#### 2. CUDA/GPU Issues
**Problem**: CUDA out of memory
```python
# Solution: Reduce batch size or use CPU
config = {
    'batch_size': 8,  # Reduce from 32
    'device': 'cpu',  # Force CPU usage
    'use_mixed_precision': False  # Disable if using CPU
}
```

#### 3. Import Errors
**Problem**: `ModuleNotFoundError`
```bash
# Solution: Install missing packages
pip install -r requirements.txt
pip install torch torchvision timm albumentations fastapi uvicorn pytest
```

#### 4. Frontend Build Issues
**Problem**: Next.js build fails
```bash
# Solution: Clear cache and reinstall
cd frontend
rm -rf .next node_modules package-lock.json
npm install
npm run build
```

#### 5. Model Loading Issues
**Problem**: Model checkpoint not found
```python
# Solution: Check model path
import os
model_path = 'ml/checkpoints/best_model.pth'
if not os.path.exists(model_path):
    print(f"Model not found at {model_path}")
    print("Train a model first: python -m ml.train_advanced")
```

### Performance Optimization

#### Speed Up Training
```python
config = {
    'batch_size': 32,  # Increase if GPU memory allows
    'num_workers': 4,  # Parallel data loading
    'use_mixed_precision': True,  # Use AMP for faster training
    'pin_memory': True,  # Faster GPU transfer
}
```

#### Reduce Memory Usage
```python
config = {
    'batch_size': 8,  # Smaller batches
    'image_size': 224,  # Smaller input size
    'gradient_accumulation': 4,  # Simulate larger batches
}
```

### Debugging Tips

#### Enable Detailed Logging
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Or set environment variable
# export PYTHONPATH=.
# export CUDA_VISIBLE_DEVICES=0
```

#### Check System Resources
```python
# Check GPU availability
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device count: {torch.cuda.device_count()}")

# Check memory usage
import psutil
memory = psutil.virtual_memory()
print(f"RAM usage: {memory.percent}%")
```

---

## 📁 File Structure

```
diabetic-retinopathy/
├── 📁 backend/                 # FastAPI backend
│   ├── app/
│   │   ├── main.py            # Main FastAPI application
│   │   ├── predict.py         # Prediction logic & Grad-CAM
│   │   └── config.py          # Configuration settings
│   └── start_server.py        # Server startup script
│
├── 📁 frontend/               # Next.js frontend
│   ├── app/
│   │   └── page.tsx           # Main page component
│   ├── components/
│   │   ├── ui/                # Base UI components
│   │   └── medical/           # Medical-specific components
│   ├── lib/
│   │   └── utils.ts           # Utility functions
│   └── package.json           # Node.js dependencies
│
├── 📁 ml/                     # Machine learning components
│   ├── models/
│   │   ├── model.py           # Base model implementations
│   │   └── advanced_architectures.py  # Advanced models
│   ├── data_preprocessing.py   # Data preprocessing pipeline
│   ├── train_advanced.py      # Advanced training script
│   ├── evaluation.py          # Model evaluation tools
│   ├── dataset_analyzer.py    # Dataset analysis tools
│   └── utils.py               # ML utilities
│
├── 📁 tests/                  # Test suite
│   ├── test_models.py         # Model tests
│   ├── test_backend.py        # API tests
│   ├── test_data_utils.py     # Data processing tests
│   └── test_evaluation.py     # Evaluation tests
│
├── 📁 scripts/                # Utility scripts
│   └── setup_environment.py   # Environment setup
│
├── 📁 data/                   # Dataset storage (created after download)
│   ├── train_images/          # Training images
│   ├── test_images/           # Test images
│   ├── train.csv             # Training labels
│   └── test.csv              # Test metadata
│
├── 📄 setup_project.py        # Automated project setup
├── 📄 download_dataset.py     # Dataset download script
├── 📄 download_dataset.bat    # Windows batch downloader
├── 📄 unzip.py               # Dataset extraction
├── 📄 requirements.txt       # Python dependencies
├── 📄 docker-compose.yml     # Docker configuration
├── 📄 Dockerfile            # Docker image definition
├── 📄 Makefile              # Build automation
├── 📄 pytest.ini           # Test configuration
├── 📄 advancement.md        # Technical documentation
└── 📄 COMPLETE_USAGE_GUIDE.md # This comprehensive guide
```

---

## 🎯 Quick Start Commands

### First Time Setup
```bash
# 1. Setup environment and dependencies
python setup_project.py

# 2. Download and extract dataset
python download_dataset.py
python unzip.py

# 3. Train a model
python -m ml.train_advanced --recipe lightweight --epochs 20

# 4. Start the API
python backend/start_server.py

# 5. Test the system
curl http://localhost:8000/health
```

### Daily Usage
```bash
# Start backend
python backend/start_server.py

# Start frontend (optional)
cd frontend && npm run dev

# Train new model
python -m ml.train_advanced --model-type vision_transformer

# Run tests
pytest tests/ -v

# Generate analysis
python -m ml.dataset_analyzer
```

---

## 🆘 Getting Help

### Documentation Resources
- **Technical Details**: See `advancement.md`
- **API Reference**: Visit http://localhost:8000/docs (when running)
- **Test Results**: Check `pytest` output for validation status
- **Training Logs**: Monitor `ml/logs/` directory

### Common Commands Reference
```bash
# Environment
python setup_project.py          # Setup everything
source venv/bin/activate         # Activate environment

# Data
python download_dataset.py       # Download APTOS dataset
python unzip.py                 # Extract dataset
python -m ml.dataset_analyzer   # Analyze dataset

# Training
python -m ml.train_advanced                    # Basic training
python -m ml.train_advanced --recipe lightweight  # Quick training
python -m ml.train_advanced --use-cv --n-folds 5  # Cross-validation

# Testing
pytest tests/ -v                # All tests
pytest tests/test_models.py     # Model tests only
pytest tests/test_backend.py    # API tests only

# Running
python backend/start_server.py  # Start API server
cd frontend && npm run dev      # Start frontend
docker-compose up              # Full stack with Docker
```

### Support
- **Issues**: Check error messages and logs
- **Performance**: Monitor GPU/CPU usage and memory
- **Debugging**: Enable detailed logging and check file paths
- **Updates**: Pull latest code and reinstall dependencies

---

🎉 **Congratulations! Your comprehensive diabetic retinopathy detection system is now ready for production use!**

For questions or issues, refer to the troubleshooting section or check the generated logs and error messages for specific guidance.