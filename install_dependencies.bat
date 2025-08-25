@echo off
echo ======================================
echo Diabetic Retinopathy Detection Setup
echo ======================================

echo Checking Python version...
python --version
if %errorlevel% neq 0 (
    echo ERROR: Python not found. Please install Python 3.8+ first.
    pause
    exit /b 1
)

echo.
echo Creating virtual environment...
if not exist "venv" (
    python -m venv venv
    if %errorlevel% neq 0 (
        echo ERROR: Failed to create virtual environment
        pause
        exit /b 1
    )
    echo Virtual environment created successfully
) else (
    echo Virtual environment already exists
)

echo.
echo Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo Upgrading pip...
python -m pip install --upgrade pip

echo.
echo Installing PyTorch (this may take a while)...
echo Checking for NVIDIA GPU...
nvidia-smi >nul 2>&1
if %errorlevel% equ 0 (
    echo NVIDIA GPU detected - installing PyTorch with CUDA support
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
) else (
    echo No NVIDIA GPU detected - installing CPU-only PyTorch
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
)

echo.
echo Installing project dependencies...
pip install -r requirements.txt

echo.
echo Installing backend dependencies...
pip install -r backend\requirements.txt

echo.
echo Setting up environment file...
if not exist ".env" (
    copy .env.example .env
    echo Environment file created from template
) else (
    echo Environment file already exists
)

echo.
echo Testing installation...
python -c "import torch; import cv2; import numpy; import pandas; import fastapi; print('SUCCESS: All core dependencies installed!')"
if %errorlevel% neq 0 (
    echo WARNING: Some dependencies may not be installed correctly
) else (
    echo All dependencies verified successfully!
)

echo.
echo ======================================
echo Installation completed!
echo ======================================
echo.
echo To activate the environment in future sessions:
echo   venv\Scripts\activate.bat
echo.
echo To start the API server:
echo   python backend\start_server.py
echo.
echo To run tests:
echo   pytest tests\ -v
echo.
echo To train a model:
echo   python -m ml.train_advanced --help
echo.
echo Press any key to continue...
pause >nul