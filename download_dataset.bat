@echo off
REM Windows batch script to download APTOS 2019 dataset
REM This script automates the dataset download process on Windows

echo =========================================
echo   APTOS 2019 Dataset Downloader for Windows
echo =========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python from https://python.org
    pause
    exit /b 1
)
echo ✓ Python is installed

REM Check if pip is available
pip --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: pip is not available
    echo Please ensure pip is installed with Python
    pause
    exit /b 1
)
echo ✓ pip is available

REM Install kaggle if not already installed
echo.
echo Checking Kaggle API installation...
pip show kaggle >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Installing Kaggle API...
    pip install kaggle
    if %ERRORLEVEL% NEQ 0 (
        echo ERROR: Failed to install Kaggle API
        pause
        exit /b 1
    )
    echo ✓ Kaggle API installed successfully
) else (
    echo ✓ Kaggle API is already installed
)

REM Check for Kaggle credentials
echo.
echo Checking Kaggle API credentials...
if not exist "%USERPROFILE%\.kaggle\kaggle.json" (
    echo.
    echo ERROR: Kaggle API credentials not found!
    echo.
    echo Setup Instructions:
    echo 1. Go to https://www.kaggle.com/account
    echo 2. Click "Create New API Token" to download kaggle.json
    echo 3. Create folder: %USERPROFILE%\.kaggle\
    echo 4. Move kaggle.json to: %USERPROFILE%\.kaggle\kaggle.json
    echo 5. Run this script again
    echo.
    pause
    exit /b 1
)
echo ✓ Kaggle credentials found

REM Test Kaggle API
echo.
echo Testing Kaggle API connection...
kaggle competitions list --page-size=1 >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Kaggle API test failed
    echo Please check your credentials and internet connection
    pause
    exit /b 1
)
echo ✓ Kaggle API connection successful

REM Create data directories
echo.
echo Creating data directories...
if not exist "data" mkdir data
if not exist "ml\data" mkdir ml\data
echo ✓ Data directories created

REM Download dataset
echo.
echo Downloading APTOS 2019 Blindness Detection dataset...
echo This may take several minutes depending on your internet connection...
echo.

kaggle competitions download -c aptos2019-blindness-detection
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: Dataset download failed
    echo.
    echo Possible solutions:
    echo 1. Check your internet connection
    echo 2. Verify you have accepted the competition rules at:
    echo    https://www.kaggle.com/competitions/aptos2019-blindness-detection/rules
    echo 3. Try manual download from the competition page
    echo.
    pause
    exit /b 1
)

echo.
echo ✓ Dataset downloaded successfully!

REM Check if unzip.py exists and offer to run it
if exist "unzip.py" (
    echo.
    set /p choice="Do you want to extract the dataset now? (Y/N): "
    if /i "%choice%"=="Y" (
        echo.
        echo Extracting dataset...
        python unzip.py
        if %ERRORLEVEL% EQU 0 (
            echo ✓ Dataset extracted successfully!
        ) else (
            echo ! Dataset extraction had some issues
        )
    )
) else (
    echo.
    echo NOTE: unzip.py not found. You'll need to manually extract the dataset.
)

echo.
echo =========================================
echo   Download Process Completed!
echo =========================================
echo.
echo Next steps:
echo 1. Extract the dataset: python unzip.py
echo 2. Analyze the data: python -m ml.dataset_analyzer
echo 3. Start training: python -m ml.train_advanced
echo 4. Run the backend: python backend/start_server.py
echo.
pause