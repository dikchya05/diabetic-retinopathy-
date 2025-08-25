@echo off
echo =============================================
echo APTOS 2019 Dataset Downloader
echo =============================================

echo.
echo Checking Kaggle API installation...
python -c "import kaggle; print('Kaggle API is installed')" 2>nul
if %errorlevel% neq 0 (
    echo ERROR: Kaggle API not found. Installing...
    venv\Scripts\pip install kaggle
    if %errorlevel% neq 0 (
        echo ERROR: Failed to install Kaggle API
        pause
        exit /b 1
    )
)

echo.
echo Downloading APTOS 2019 dataset...
echo This may take several minutes depending on your internet speed...

echo.
echo Creating data directories...
if not exist "data" mkdir data
if not exist "ml\data" mkdir ml\data

echo.
echo Downloading dataset from Kaggle...
venv\Scripts\python -m kaggle competitions download -c aptos2019-blindness-detection -p .

if %errorlevel% neq 0 (
    echo.
    echo ========================================
    echo Download failed. Please try manually:
    echo ========================================
    echo 1. Go to: https://www.kaggle.com/competitions/aptos2019-blindness-detection
    echo 2. Click "Data" tab
    echo 3. Click "Download All"
    echo 4. Save aptos2019-blindness-detection.zip to this directory
    echo 5. Run: python unzip.py
    echo.
    pause
    exit /b 1
)

echo.
echo ======================================
echo Download completed successfully!
echo ======================================
echo.
echo Next steps:
echo 1. Run: python unzip.py
echo 2. Check data in .\data\ directory
echo 3. Start training: python -m ml.train_advanced
echo.
pause