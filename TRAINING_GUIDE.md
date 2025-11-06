# Training Guide - Diabetic Retinopathy Detection Model

## Complete Guide to Training Your ResNet50 Model

This guide provides step-by-step instructions for training the Diabetic Retinopathy detection model from scratch.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Dataset Preparation](#dataset-preparation)
3. [Quick Start](#quick-start)
4. [Training Options](#training-options)
5. [Monitoring Training](#monitoring-training)
6. [Troubleshooting](#troubleshooting)
7. [Advanced Options](#advanced-options)

---

## Prerequisites

### 1. Install Dependencies

Make sure all required packages are installed:

```bash
pip install -r requirements.txt
```

**Required packages:**
- torch
- torchvision
- timm
- albumentations
- pandas
- numpy
- tqdm
- scikit-learn
- Pillow

### 2. Check GPU (Optional but Recommended)

```python
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

**Expected output:**
- `CUDA available: True` → You have GPU (faster training)
- `CUDA available: False` → Will use CPU (slower but works)

---

## Dataset Preparation

### 1. Download APTOS 2019 Dataset

From Kaggle: https://www.kaggle.com/c/aptos2019-blindness-detection/data

**Files needed:**
- `train.csv` - Labels file
- `train_images/` - Folder with retinal images

### 2. Organize Your Data

Place files in this structure:

```
diabetic-retinopathy/
├── ml/
│   └── data/
│       ├── train.csv           ← Labels CSV
│       └── train_images/       ← Image folder
│           ├── 000c1434d8d7.png
│           ├── 001639a390f0.png
│           └── ...
```

### 3. Verify CSV Format

Your `train.csv` should have these columns:

| id_code | diagnosis |
|---------|-----------|
| 000c1434d8d7 | 0 |
| 001639a390f0 | 2 |
| ... | ... |

**Note:** The script will automatically rename `diagnosis` to `label` if needed.

**DR Severity Labels:**
- 0 = No DR
- 1 = Mild DR
- 2 = Moderate DR
- 3 = Severe DR
- 4 = Proliferative DR

---

## Quick Start

### Option 1: Using the New Training Script (Recommended)

```bash
python train_model.py --labels-csv ml/data/train.csv \
                      --img-dir ml/data/train_images \
                      --epochs 20 \
                      --batch-size 16
```

### Option 2: Using Existing Training Script

```bash
python ml/train.py --labels-csv ml/data/train.csv \
                   --img-dir ml/data/train_images \
                   --epochs 20 \
                   --batch-size 32
```

### Option 3: Using Training Loop Directly

```bash
python -c "
import pandas as pd
from ml.models.model import train_loop

df = pd.read_csv('ml/data/train.csv')
if 'diagnosis' in df.columns:
    df = df.rename(columns={'diagnosis': 'label'})

train_loop(
    labels_df=df,
    img_dir='ml/data/train_images',
    model_name='resnet50',
    epochs=20,
    batch_size=16,
    lr=2e-4
)
"
```

---

## Training Options

### Basic Training (Default Settings)

```bash
python train_model.py --labels-csv ml/data/train.csv \
                      --img-dir ml/data/train_images
```

**Default settings:**
- Model: ResNet50
- Epochs: 20
- Batch size: 16
- Learning rate: 0.0002
- Image size: 224x224
- Early stopping: 5 epochs patience

### Fast Training (Quick Test)

For testing if everything works:

```bash
python train_model.py --labels-csv ml/data/train.csv \
                      --img-dir ml/data/train_images \
                      --epochs 5 \
                      --batch-size 8
```

### Production Training (Best Results)

For final model training:

```bash
python train_model.py --labels-csv ml/data/train.csv \
                      --img-dir ml/data/train_images \
                      --epochs 50 \
                      --batch-size 32 \
                      --lr 2e-4 \
                      --early-stopping-patience 10
```

### Low Memory Training

If you get "Out of Memory" errors:

```bash
python train_model.py --labels-csv ml/data/train.csv \
                      --img-dir ml/data/train_images \
                      --batch-size 4 \
                      --num-workers 0
```

### Resume Training

If training was interrupted:

```bash
python train_model.py --labels-csv ml/data/train.csv \
                      --img-dir ml/data/train_images \
                      --resume ml/models/best_model.pth
```

---

## Monitoring Training

### What You'll See During Training

```
================================================================================
DIABETIC RETINOPATHY MODEL TRAINING
================================================================================

📋 Configuration:
   Model: resnet50
   Dataset: ml/data/train.csv
   Images: ml/data/train_images
   Epochs: 20
   Batch Size: 16
   Learning Rate: 0.0002
   Device: cuda
   GPU: NVIDIA GeForce RTX 3080
================================================================================

📊 Loading dataset...
   ✅ Loaded 3662 images
   Columns: ['id_code', 'label']

📈 Class Distribution:
   No DR: 1805 images (49.3%)
   Mild DR: 370 images (10.1%)
   Moderate DR: 999 images (27.3%)
   Severe DR: 193 images (5.3%)
   Proliferative DR: 295 images (8.1%)

🚀 Starting training...

Epoch 1: train_loss=1.2345, val_loss=1.1234, val_acc=0.4567
Saved best model to ml/models/best_model.pth

Epoch 2: train_loss=0.9876, val_loss=0.8765, val_acc=0.5678
Saved best model to ml/models/best_model.pth

...
```

### Key Metrics to Watch

1. **Train Loss**: Should decrease over time
   - Good: Steadily decreasing
   - Bad: Not changing or increasing

2. **Val Loss**: Should decrease over time
   - Good: Decreasing with train loss
   - Bad: Increasing while train loss decreases (overfitting)

3. **Val Accuracy**: Should increase over time
   - Good: Steadily increasing
   - Target: 60-80% (depends on dataset)

4. **Learning Rate**: Will decrease automatically
   - Starts at 0.0002
   - Reduces when validation loss plateaus

### Training Time Estimates

**With GPU (NVIDIA RTX 3080 or similar):**
- Small dataset (1000 images): ~5-10 minutes per epoch
- Medium dataset (3000 images): ~10-20 minutes per epoch
- Large dataset (10000 images): ~30-45 minutes per epoch

**With CPU:**
- Small dataset: ~30-60 minutes per epoch
- Medium dataset: ~1-3 hours per epoch
- Large dataset: ~4-8 hours per epoch

**Total training time:**
- 20 epochs on GPU: 3-6 hours
- 20 epochs on CPU: 20-40 hours

---

## Troubleshooting

### Error: "CUDA out of memory"

**Solution:** Reduce batch size

```bash
python train_model.py --labels-csv ml/data/train.csv \
                      --img-dir ml/data/train_images \
                      --batch-size 8  # or try 4
```

### Error: "RuntimeError: DataLoader worker"

**Solution:** Set num_workers to 0

```bash
python train_model.py --labels-csv ml/data/train.csv \
                      --img-dir ml/data/train_images \
                      --num-workers 0
```

### Error: "FileNotFoundError: [image_name].png"

**Solutions:**
1. Check image file extension (PNG vs JPG)
2. Verify image directory path
3. Check CSV has correct image IDs

```bash
# List first few images
ls ml/data/train_images/ | head -5

# Check CSV
python -c "import pandas as pd; df=pd.read_csv('ml/data/train.csv'); print(df.head())"
```

### Training is Too Slow

**Solutions:**

1. **Use GPU if available:**
   - Check: `nvidia-smi`
   - Install CUDA version of PyTorch

2. **Increase batch size (if you have memory):**
   ```bash
   --batch-size 32  # or 64
   ```

3. **Reduce image size (less accurate but faster):**
   ```bash
   --image-size 128  # instead of 224
   ```

4. **Use fewer workers:**
   ```bash
   --num-workers 2  # instead of 4
   ```

### Validation Accuracy Not Improving

**Possible causes:**

1. **Not enough epochs:**
   ```bash
   --epochs 30  # try more epochs
   ```

2. **Learning rate too high/low:**
   ```bash
   --lr 1e-4  # try lower learning rate
   ```

3. **Dataset too small:**
   - Need at least 500-1000 images per class
   - Consider data augmentation (already included)

4. **Class imbalance:**
   - Script uses class weighting automatically
   - Check class distribution in output

### Model Not Saving

**Check:**
1. Output directory exists and is writable
2. Enough disk space
3. Validation accuracy is improving

```bash
# Create output directory
mkdir -p ml/models

# Check disk space
df -h
```

---

## Advanced Options

### Change Model Architecture

Try different ResNet variants:

```bash
# ResNet34 (smaller, faster)
python train_model.py --labels-csv ml/data/train.csv \
                      --img-dir ml/data/train_images \
                      --model-name resnet34

# ResNet101 (larger, slower, potentially more accurate)
python train_model.py --labels-csv ml/data/train.csv \
                      --img-dir ml/data/train_images \
                      --model-name resnet101
```

**Available models:**
- `resnet18` - Smallest, fastest
- `resnet34` - Small, fast
- `resnet50` - **Recommended** (balanced)
- `resnet101` - Large, slow
- `resnet152` - Largest, slowest

### Custom Learning Rate Schedule

The script uses:
- **ReduceLROnPlateau**: Reduces LR when validation loss plateaus
- **Patience**: 2 epochs
- **Factor**: 0.5 (halves the learning rate)

### Early Stopping

Prevents overfitting by stopping when validation doesn't improve:

```bash
# Stop after 10 epochs without improvement
python train_model.py --labels-csv ml/data/train.csv \
                      --img-dir ml/data/train_images \
                      --early-stopping-patience 10
```

### Mixed Precision Training

Automatically enabled when using GPU (faster training, less memory):
- Uses `torch.cuda.amp.autocast()`
- No additional flags needed

### Class Weights

Automatically computed to handle class imbalance:
- Minority classes get higher weight
- Majority classes get lower weight
- Prevents bias toward common classes

---

## After Training

### 1. Verify Model

Check the trained model architecture:

```bash
python ml/check_model.py
```

**Expected output:**
```
Model Architecture: resnet50
Number of Classes: 5
Image Size: 224x224
Best Validation Loss: 0.7234
```

### 2. Evaluate Model

Test on held-out test set:

```bash
python ml/evaluate.py --labels-csv ml/data/train.csv \
                      --img-dir ml/data/train_images \
                      --model-path ml/models/best_model.pth
```

This generates:
- Accuracy, precision, recall, F1
- Confusion matrix
- ROC curves
- All saved in `results/` folder

### 3. Use Model for Inference

Start the backend API:

```bash
cd backend
uvicorn app.main:app --reload
```

Access at: http://localhost:8000

---

## Training Best Practices

### 1. Data Quality

- ✅ High-quality retinal images
- ✅ Balanced class distribution (if possible)
- ✅ Verified labels (manually checked)
- ✅ Sufficient data (1000+ images recommended)

### 2. Hyperparameters

**For most cases, use defaults:**
- Batch size: 16 (GPU) or 4 (CPU/small GPU)
- Learning rate: 2e-4
- Epochs: 20-30
- Image size: 224x224

**Fine-tuning:**
- Lower LR if loss oscillates: `--lr 1e-4`
- Higher LR if training is too slow: `--lr 5e-4`
- More epochs if accuracy still improving: `--epochs 50`

### 3. Monitoring

**Good signs:**
- ✅ Train loss decreasing
- ✅ Val loss decreasing
- ✅ Val accuracy increasing
- ✅ Small gap between train and val loss

**Bad signs:**
- ❌ Val loss increasing (overfitting)
- ❌ Loss not changing (learning rate too low)
- ❌ Loss exploding (learning rate too high)
- ❌ Large gap between train and val loss (overfitting)

### 4. Checkpointing

The script automatically:
- Saves best model (based on validation loss)
- Includes metadata (architecture, epochs, etc.)
- Allows resuming interrupted training

---

## Example: Complete Training Workflow

```bash
# 1. Verify setup
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
ls ml/data/train_images/ | wc -l  # Count images

# 2. Quick test (5 epochs)
python train_model.py --labels-csv ml/data/train.csv \
                      --img-dir ml/data/train_images \
                      --epochs 5 \
                      --batch-size 8

# 3. If test successful, full training
python train_model.py --labels-csv ml/data/train.csv \
                      --img-dir ml/data/train_images \
                      --epochs 30 \
                      --batch-size 16

# 4. Verify trained model
python ml/check_model.py

# 5. Evaluate on test set
python ml/evaluate.py --model-path ml/models/best_model.pth

# 6. Check results
cat results/summary.txt
```

---

## FAQ

**Q: How long does training take?**
A: 3-6 hours on GPU, 20-40 hours on CPU for 20 epochs with ~3000 images.

**Q: How much data do I need?**
A: Minimum 500 images, recommended 1000+ images. More is better.

**Q: Can I use my own dataset?**
A: Yes! Just ensure CSV has `id_code` and `label` columns, and images are in specified folder.

**Q: What accuracy should I expect?**
A: Depends on dataset quality. Typically 60-80% for 5-class DR classification.

**Q: Should I use GPU or CPU?**
A: GPU is 10-20x faster. Highly recommended for practical training.

**Q: Can I train multiple models?**
A: Yes! Change `--output-dir` for each run to save different models.

**Q: How do I know when to stop training?**
A: Early stopping will handle this automatically. Or watch validation loss - stop when it stops decreasing.

---

## Support

If you encounter issues:

1. Check error message carefully
2. Review troubleshooting section
3. Try reducing batch size
4. Set num_workers to 0
5. Verify data paths and format

---

**Good luck with your training!** 🚀

---

**Created:** 2025
**Purpose:** Final Year Project - Diabetic Retinopathy Detection
