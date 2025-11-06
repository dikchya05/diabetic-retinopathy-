# Training Quick Reference

## 🚀 Most Common Commands

### Basic Training (Recommended)
```bash
python train_model.py --labels-csv ml/data/train.csv \
                      --img-dir ml/data/train_images \
                      --epochs 20
```

### Quick Test (5 epochs)
```bash
python train_model.py --labels-csv ml/data/train.csv \
                      --img-dir ml/data/train_images \
                      --epochs 5 \
                      --batch-size 8
```

### Low Memory Training
```bash
python train_model.py --labels-csv ml/data/train.csv \
                      --img-dir ml/data/train_images \
                      --batch-size 4 \
                      --num-workers 0
```

### Production Training (Best Quality)
```bash
python train_model.py --labels-csv ml/data/train.csv \
                      --img-dir ml/data/train_images \
                      --epochs 50 \
                      --batch-size 32
```

---

## 📋 All Available Options

| Option | Default | Description |
|--------|---------|-------------|
| `--labels-csv` | *required* | Path to CSV with labels |
| `--img-dir` | *required* | Directory with images |
| `--model-name` | `resnet50` | Model architecture |
| `--epochs` | `20` | Number of training epochs |
| `--batch-size` | `16` | Batch size |
| `--lr` | `2e-4` | Learning rate |
| `--image-size` | `224` | Input image size |
| `--output-dir` | `ml/models` | Where to save model |
| `--early-stopping-patience` | `5` | Epochs without improvement before stopping |
| `--num-workers` | `0` | Data loading workers |
| `--resume` | `None` | Checkpoint to resume from |

---

## ⚡ Quick Workflows

### Complete Training Pipeline
```bash
# 1. Test training works (5 epochs)
python train_model.py --labels-csv ml/data/train.csv \
                      --img-dir ml/data/train_images \
                      --epochs 5

# 2. Full training (20 epochs)
python train_model.py --labels-csv ml/data/train.csv \
                      --img-dir ml/data/train_images \
                      --epochs 20

# 3. Verify model
python ml/check_model.py

# 4. Evaluate model
python ml/evaluate.py

# 5. View results
cat results/summary.txt
```

### Resume Interrupted Training
```bash
python train_model.py --labels-csv ml/data/train.csv \
                      --img-dir ml/data/train_images \
                      --resume ml/models/best_model.pth \
                      --epochs 30
```

---

## 🐛 Common Issues & Solutions

| Problem | Solution |
|---------|----------|
| Out of memory | `--batch-size 4` or `--batch-size 8` |
| DataLoader error | `--num-workers 0` |
| Too slow | Use GPU or reduce `--image-size 128` |
| Not learning | Try `--lr 1e-4` or `--lr 5e-4` |
| Overfitting | Reduce `--epochs` or use more data |

---

## 📊 After Training

```bash
# Check model architecture
python ml/check_model.py

# Evaluate on test set
python ml/evaluate.py --model-path ml/models/best_model.pth

# View quick summary
cat results/summary.txt

# View all results
ls results/
```

---

## 💡 Tips

✅ **Start with default settings** - they work well for most cases
✅ **Use GPU if available** - 10-20x faster
✅ **Monitor validation loss** - should decrease over time
✅ **Check class distribution** - shown at start of training
✅ **Save checkpoints** - model saved automatically to `ml/models/`

---

For detailed guide, see **TRAINING_GUIDE.md**
