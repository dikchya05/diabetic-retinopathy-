# 🛠️ Installation Guide

This guide will help you set up the complete Diabetic Retinopathy Detection project with all dependencies and requirements.

## 📋 Prerequisites

### System Requirements
- **Python**: 3.8 or higher (recommended: 3.11)
- **Operating System**: Windows 10/11, macOS, or Linux
- **Memory**: At least 8GB RAM (16GB+ recommended for training)
- **Storage**: At least 10GB free space
- **GPU**: NVIDIA GPU with CUDA support (optional but recommended for training)

### Check Your Python Version
```bash
python --version
# Should show Python 3.8+ 
```

If you don't have Python installed:
- **Windows**: Download from [python.org](https://www.python.org/downloads/)
- **macOS**: Use Homebrew `brew install python` or download from python.org
- **Linux**: Use your package manager `sudo apt-get install python3.11`

## 🚀 Quick Installation

### Option 1: Automatic Setup (Recommended)

I'll create an installation script for you:

```bash
# Clone or navigate to the project directory
cd diabetic-retinopathy-

# Run the setup script
python setup_project.py
```

### Option 2: Manual Installation

#### Step 1: Create Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

#### Step 2: Upgrade pip
```bash
python -m pip install --upgrade pip
```

#### Step 3: Install PyTorch (Choose based on your system)

**For CPU-only (smaller download):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

**For GPU (CUDA 11.8):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**For GPU (CUDA 12.1):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

#### Step 4: Install All Dependencies
```bash
# Install main dependencies
pip install -r requirements.txt

# Install backend dependencies
pip install -r backend/requirements.txt
```

#### Step 5: Install Frontend Dependencies (Optional)
```bash
cd frontend
npm install
cd ..
```

## 🔧 Detailed Installation Steps

### 1. Core ML Dependencies

```bash
# Core scientific computing
pip install numpy>=1.24.0 pandas>=2.0.0

# Computer vision and image processing
pip install opencv-python>=4.8.0 Pillow>=10.0.0

# Machine learning framework
pip install torch>=2.0.0 torchvision>=0.15.0

# Pre-trained models and augmentation
pip install timm>=0.9.0 albumentations>=1.3.0

# Traditional ML algorithms
pip install scikit-learn>=1.3.0
```

### 2. Visualization and Analysis

```bash
# Plotting and visualization
pip install matplotlib>=3.7.0 seaborn>=0.12.0

# Interactive plots
pip install plotly>=5.15.0 kaleido>=0.2.1

# Model explainability
pip install pytorch-grad-cam>=1.4.8
```

### 3. Web API Framework

```bash
# FastAPI and server
pip install fastapi>=0.104.0 uvicorn[standard]>=0.24.0

# File upload support
pip install python-multipart>=0.0.6

# Data validation
pip install pydantic>=2.0.0
```

### 4. Development and Testing Tools

```bash
# Testing framework
pip install pytest>=7.4.0 pytest-cov>=4.1.0

# API testing
pip install httpx>=0.25.0

# Code formatting and linting
pip install black>=23.0.0 flake8>=6.0.0 isort>=5.12.0
```

### 5. Optional Tools

```bash
# Experiment tracking
pip install mlflow>=2.5.0

# Jupyter notebooks
pip install jupyter>=1.0.0 ipykernel>=6.25.0

# Progress bars
pip install tqdm>=4.64.0

# Logging
pip install python-json-logger>=2.0.0
```

## 🐳 Docker Installation (Alternative)

If you prefer using Docker:

```bash
# Build the Docker image
docker build -t diabetic-retinopathy:latest --target production .

# Or use Docker Compose
docker-compose up -d api
```

## ✅ Verify Installation

### Test Core Dependencies
```bash
python -c "
import torch
import torchvision
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import timm
import albumentations
from sklearn import metrics
import fastapi
print('✅ All core dependencies installed successfully!')
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA device: {torch.cuda.get_device_name(0)}')
"
```

### Test Advanced Features
```bash
python -c "
import plotly.express as px
from pytorch_grad_cam import GradCAM
import pytest
import mlflow
print('✅ All advanced features available!')
"
```

### Test Project Modules
```bash
# Test if our custom modules can be imported
python -c "
import sys
import os
sys.path.append('.')
try:
    from ml.utils import RetinopathyDataset, get_transforms
    from ml.evaluation import ModelEvaluator
    from ml.models.advanced_architectures import create_advanced_model
    from backend.app.config import settings
    print('✅ All project modules can be imported!')
except ImportError as e:
    print(f'❌ Import error: {e}')
"
```

### Run Tests
```bash
# Run the test suite
pytest tests/ -v
```

### Test API
```bash
# Start the API server
python backend/start_server.py

# In another terminal, test the health endpoint
curl http://localhost:8000/health
```

## 🚨 Troubleshooting

### Common Issues and Solutions

#### 1. PyTorch Installation Issues
```bash
# If torch installation fails, try:
pip install torch torchvision --no-cache-dir

# For specific CUDA version:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

#### 2. OpenCV Issues
```bash
# If opencv-python fails:
pip install opencv-python-headless

# On Linux, you might need:
sudo apt-get update
sudo apt-get install libgl1-mesa-glx libglib2.0-0
```

#### 3. Memory Issues During Installation
```bash
# Install with no cache to save memory
pip install --no-cache-dir -r requirements.txt

# Or install packages one by one
pip install numpy pandas torch torchvision
```

#### 4. Permission Errors (Windows)
- Run command prompt as Administrator
- Or use: `pip install --user -r requirements.txt`

#### 5. Version Conflicts
```bash
# Create a fresh virtual environment
rm -rf venv
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install --upgrade pip
pip install -r requirements.txt
```

### Platform-Specific Notes

#### Windows
- Use `venv\Scripts\activate` to activate virtual environment
- You might need Visual Studio Build Tools for some packages
- Install Git for Windows if not already installed

#### macOS
- Install Xcode Command Line Tools: `xcode-select --install`
- Use Homebrew for system dependencies: `brew install python`
- For M1/M2 Macs, PyTorch has native support

#### Linux (Ubuntu/Debian)
```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install python3-pip python3-venv python3-dev
sudo apt-get install libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev

# For CUDA support
# Install NVIDIA drivers and CUDA toolkit from NVIDIA website
```

## 🎯 Next Steps After Installation

1. **Download Dataset**: Get the APTOS 2019 Blindness Detection dataset from Kaggle
2. **Configure Environment**: Copy `.env.example` to `.env` and adjust settings
3. **Train Model**: Run `python -m ml.train_advanced --help` to see training options
4. **Start API**: Run `python backend/start_server.py` to start the web API
5. **Run Tests**: Execute `pytest` to ensure everything works
6. **Try Frontend**: Navigate to `frontend/` and run `npm run dev`

## 📚 Additional Resources

- [PyTorch Installation Guide](https://pytorch.org/get-started/locally/)
- [CUDA Installation Guide](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Docker Installation](https://docs.docker.com/get-docker/)

---

If you encounter any issues not covered here, please check the project's GitHub issues or create a new issue with:
- Your operating system
- Python version
- Full error message
- Steps you've tried