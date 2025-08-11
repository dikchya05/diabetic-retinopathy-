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
pip install -r ml/requirements.txt
```

### 4. Prepare dataset

- Download the Kaggle **APTOS 2019 Blindness Detection** dataset.
- Place images in `ml/data/train_images/` and labels CSV in `ml/data/train.csv`.
- Adjust paths in config files if needed.

---

## Training the Model

Train the model with:

```bash
python -m ml.train --labels-csv ml/data/train.csv --img-dir ml/data/train_images --epochs 5
```

- The best model checkpoint will be saved at `ml/models/best_model.pth`.

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

## Notes

- The API provides Grad-CAM visualizations as a base64 encoded PNG to help interpret model predictions.
- The model and API expect 5 classes labeled `0` through `4`.
- Adjust `IMAGE_SIZE` and other configs in `backend/app/config.py` as needed.
- Use `.gitignore` to exclude large files like model weights and datasets.

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

---

## License

Specify your license here.

