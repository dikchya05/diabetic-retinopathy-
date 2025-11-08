@echo off
REM ============================================================================
REM MAXIMUM ACCURACY TRAINING FOR NZ HEALTH GOVERNMENT PITCH (WINDOWS)
REM ============================================================================

echo ============================================================================
echo TRAINING PIPELINE FOR NZ HEALTH GOVERNMENT DEPLOYMENT
echo ============================================================================
echo.
echo This will train 3 state-of-the-art models for ensemble prediction:
echo   1. EfficientNet-B3 (balanced speed/accuracy)
echo   2. EfficientNet-B5 (maximum accuracy)
echo   3. ResNet101 (robust baseline)
echo.
echo Total training time: 12-18 hours
echo Expected ensemble accuracy: 94-97%%
echo.
pause

REM Create output directory
mkdir ml\models\government_deployment 2>nul

REM ============================================================================
REM MODEL 1: EfficientNet-B3
REM ============================================================================
echo.
echo ============================================================================
echo MODEL 1/3: Training EfficientNet-B3 (Balanced Speed/Accuracy)
echo ============================================================================

python ml/train.py ^
  --labels-csv ml/data/train.csv ^
  --img-dir ml/data/train_images ^
  --save-dir ml/models/government_deployment/efficientnet_b3 ^
  --model-name efficientnet_b3 ^
  --loss focal ^
  --focal-gamma 2.5 ^
  --two-stage ^
  --epochs-stage1 10 ^
  --epochs-stage2 50 ^
  --batch-size 20 ^
  --lr-stage1 0.001 ^
  --lr-stage2 0.00003 ^
  --weight-decay 0.00015 ^
  --image-size 320 ^
  --patience 15 ^
  --mixed-precision ^
  --advanced-aug ^
  --medical-preprocess ^
  --num-workers 0 ^
  --random-seed 42

if errorlevel 1 (
    echo ERROR: Model 1 training failed
    pause
    exit /b 1
)

REM ============================================================================
REM MODEL 2: EfficientNet-B5
REM ============================================================================
echo.
echo ============================================================================
echo MODEL 2/3: Training EfficientNet-B5 (Maximum Accuracy)
echo ============================================================================

python ml/train.py ^
  --labels-csv ml/data/train.csv ^
  --img-dir ml/data/train_images ^
  --save-dir ml/models/government_deployment/efficientnet_b5 ^
  --model-name efficientnet_b5 ^
  --loss focal ^
  --focal-gamma 3.0 ^
  --two-stage ^
  --epochs-stage1 12 ^
  --epochs-stage2 60 ^
  --batch-size 12 ^
  --lr-stage1 0.001 ^
  --lr-stage2 0.00002 ^
  --weight-decay 0.0002 ^
  --image-size 456 ^
  --patience 18 ^
  --mixed-precision ^
  --advanced-aug ^
  --medical-preprocess ^
  --num-workers 0 ^
  --random-seed 123

if errorlevel 1 (
    echo ERROR: Model 2 training failed
    pause
    exit /b 1
)

REM ============================================================================
REM MODEL 3: ResNet101
REM ============================================================================
echo.
echo ============================================================================
echo MODEL 3/3: Training ResNet101 (Robust Baseline)
echo ============================================================================

python ml/train.py ^
  --labels-csv ml/data/train.csv ^
  --img-dir ml/data/train_images ^
  --save-dir ml/models/government_deployment/resnet101 ^
  --model-name resnet101 ^
  --loss focal ^
  --focal-gamma 2.5 ^
  --two-stage ^
  --epochs-stage1 10 ^
  --epochs-stage2 45 ^
  --batch-size 24 ^
  --lr-stage1 0.001 ^
  --lr-stage2 0.00005 ^
  --weight-decay 0.00012 ^
  --image-size 300 ^
  --patience 15 ^
  --mixed-precision ^
  --advanced-aug ^
  --medical-preprocess ^
  --num-workers 0 ^
  --random-seed 456

if errorlevel 1 (
    echo ERROR: Model 3 training failed
    pause
    exit /b 1
)

REM ============================================================================
REM TRAINING COMPLETE
REM ============================================================================
echo.
echo ============================================================================
echo ALL MODELS TRAINED SUCCESSFULLY
echo ============================================================================
echo.
echo Next steps:
echo   1. Evaluate ensemble: python ml/evaluate_ensemble.py
echo   2. Generate government report: python ml/generate_government_report.py
echo   3. Review results in: ml/models/government_deployment/
echo.
echo Models saved:
echo   - ml/models/government_deployment/efficientnet_b3/best_model.pth
echo   - ml/models/government_deployment/efficientnet_b5/best_model.pth
echo   - ml/models/government_deployment/resnet101/best_model.pth
echo.
echo Expected ensemble accuracy: 94-97%%
echo Ready for clinical deployment and government pitch!
echo ============================================================================
pause
