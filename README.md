# Diabetic Retinopathy Detection Project

This project implements a machine learning pipeline for detecting diabetic retinopathy from retinal images. It includes:

- Model training scripts using PyTorch and timm
- A FastAPI backend serving an inference API with Grad-CAM visualization
- Utilities for data preprocessing and prediction

---

## Project Structure

```
diabetic-retinopathy-project/
├── backend/
│   └── app/
│       ├── main.py           # FastAPI server entrypoint
│       ├── predict.py        # Model loading, preprocessing, inference, Grad-CAM
│       └── config.py         # Configuration variables and paths
├── ml/
│   ├── data/                 # Training dataset & CSV labels
│   ├── models/               # Model architectures, training scripts, saved weights
│   └── train.py              # Training script
├── data/                     # Raw and processed dataset files
├── scripts/                  # Utility scripts like unzip.py
├── requirements.txt          # Common dependencies
└── README.md
```

---

## Setup Instructions

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd diabetic-retinopathy-project
```

### 2. Create and activate a Python virtual environment

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r backend/requirements.txt
pip install -r requirements.txt
```

### 4. Prepare dataset

- Download the Kaggle **APTOS 2019 Blindness Detection** dataset.
- Place images in `ml/data/train_images/` and labels CSV in `ml/data/train.csv`.
- Adjust paths in config files if needed.

---

## Training the Model

### Quick Start - Train ResNet50

Use the comprehensive training script:

```bash
python train_model.py --labels-csv ml/data/train.csv \
                      --img-dir ml/data/train_images \
                      --epochs 20
```

**Features:**
- ✅ ResNet50 architecture (matches report)
- ✅ Class-weighted loss (handles imbalanced data)
- ✅ Mixed precision training (faster on GPU)
- ✅ Early stopping (prevents overfitting)
- ✅ Automatic best model saving
- ✅ Comprehensive logging

### Alternative Methods

**Method 1: Using new training script (Recommended)**
```bash
python train_model.py --labels-csv ml/data/train.csv \
                      --img-dir ml/data/train_images \
                      --epochs 20 \
                      --batch-size 16
```

**Method 2: Using existing training script**
```bash
python ml/train.py --labels-csv ml/data/train.csv \
                   --img-dir ml/data/train_images \
                   --epochs 20 \
                   --batch-size 32
```

**Method 3: Using training loop directly**
```bash
python -m ml.train --labels-csv ml/data/train.csv \
                   --img-dir ml/data/train_images \
                   --epochs 5
```

### Training Options

Common options for `train_model.py`:

```bash
--epochs 20              # Number of training epochs
--batch-size 16          # Batch size (reduce if out of memory)
--lr 2e-4               # Learning rate
--model-name resnet50   # Model architecture
--num-workers 0         # Data loading workers (0 for Windows)
```

### Complete Training Guide

See **[TRAINING_GUIDE.md](TRAINING_GUIDE.md)** for:
- Dataset preparation
- Detailed options explanation
- Troubleshooting
- Best practices
- GPU vs CPU training
- Advanced configurations

**Quick reference:** [TRAINING_QUICK_REFERENCE.md](TRAINING_QUICK_REFERENCE.md)

### Model Output

- The best model checkpoint will be saved at `ml/models/best_model.pth`
- Includes metadata (architecture, epochs, validation loss, etc.)
- Can be used for evaluation and inference

---

## Running the Inference API

### 1. Set the model path environment variable (optional)

```bash
export MODEL_PATH=/full/path/to/ml/models/best_model.pth  # macOS/Linux
set MODEL_PATH=D:\path\to\ml\models\best_model.pth       # Windows CMD
```

### 2. Start FastAPI server

```bash
uvicorn backend.app.main:app --reload
```

### 3. Test the API

Send a POST request with an image file to `/predict` endpoint:

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@path_to_image.png;type=image/png"
```

Example response:

```json
{
  "prediction": 2,
  "label": "2",
  "confidence": 0.51,
  "probabilities": [0.01, 0.37, 0.51, 0.003, 0.09],
  "gradcam_base64": "<base64 encoded heatmap image or null>"
}
```

---

## Testing & Evaluation

### NEW: Comprehensive Model Evaluation

**Added for Final Year Project Submission**

The project now includes a complete testing phase with proper train/val/test split and comprehensive evaluation metrics.

#### 1. Verify Model Architecture

First, check which model architecture is loaded:

```bash
python ml/check_model.py
```

This will display:
- Model architecture (should be ResNet50)
- Number of classes
- Image size
- Training metadata

#### 2. Run Comprehensive Evaluation

Evaluate the model on a held-out test set:

```bash
python ml/evaluate.py --labels-csv ml/data/train.csv \
                      --img-dir ml/data/train_images \
                      --model-path ml/models/best_model-1.pth \
                      --output-dir results
```

**What this does:**
- Creates proper train/val/test split (70/15/15) with stratification
- Evaluates model on **unseen** test data
- Generates comprehensive metrics and visualizations

#### 3. View Results

All evaluation results are saved in the `results/` directory:

```
results/
├── metrics.json              # Comprehensive metrics in JSON
├── classification_report.txt # Detailed per-class performance
├── test_predictions.csv      # All predictions with confidence
├── confusion_matrix.png      # Visual confusion matrix
├── roc_curves.png           # ROC curves for all classes
└── summary.txt              # Quick summary
```

**Quick summary:**
```bash
cat results/summary.txt
```

#### 4. Evaluation Metrics Included

- **Overall:** Accuracy, Cohen's Kappa
- **Per-class:** Precision, Recall, F1-Score, AUC-ROC
- **Medical:** Sensitivity, Specificity
- **Visualizations:** Confusion Matrix, ROC Curves

#### 5. Update Final Year Report

Use the generated `EVALUATION_RESULTS.md` template:
1. Run the evaluation script
2. Fill in the template with results from `results/`
3. Include visualizations in your report
4. Discuss findings and clinical implications

See `EVALUATION_RESULTS.md` for detailed guidance.

---

## Model Architecture

**Model:** ResNet50 (pre-trained on ImageNet)

This matches the architecture mentioned in the final year report for consistency with documentation.

**Key specifications:**
- Architecture: ResNet50 via TIMM
- Input size: 224x224 RGB images
- Output: 5 classes (DR severity 0-4)
- Transfer learning: ImageNet pre-trained weights
- Training: Class-weighted loss for imbalance handling

---

## Data Split Strategy

For proper academic evaluation:

| Split | Percentage | Purpose |
|-------|-----------|---------|
| Training | 70% | Model training |
| Validation | 15% | Hyperparameter tuning & early stopping |
| Test | 15% | Final evaluation on unseen data |

**Stratification:** Yes - maintains class distribution across splits

---

## Notes

- The API provides Grad-CAM visualizations as a base64 encoded PNG to help interpret model predictions.
- The model and API expect 5 classes labeled `0` through `4`.
- Adjust `IMAGE_SIZE` and other configs in `backend/app/config.py` as needed.
- Use `.gitignore` to exclude large files like model weights and datasets.
- **NEW:** Testing phase implemented with train/val/test split (addresses supervisor feedback)
- **NEW:** Model architecture standardized to ResNet50 (matches final year report)

---

## Supervisor Feedback Addressed

✅ **Testing Phase Added:**
- Proper train/validation/test split (70/15/15)
- Comprehensive evaluation on unseen test data
- Multiple metrics: accuracy, precision, recall, F1, AUC, Cohen's Kappa
- Visualizations: confusion matrix, ROC curves

✅ **Model Consistency Fixed:**
- Code updated to use ResNet50 (matches report)
- Clear documentation of architecture choice
- Model verification script added

---

## Dependencies

- Python 3.8+
- PyTorch
- timm
- torchvision
- fastapi
- uvicorn
- Pillow
- opencv-python
- pytorch-grad-cam
- scikit-learn (for evaluation metrics)
- matplotlib, seaborn (for visualizations)

---

## License

Specify your license here.

