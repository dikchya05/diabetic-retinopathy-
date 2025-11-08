# Diabetic Retinopathy Detection - Setup and Running Guide

This guide provides comprehensive instructions to set up and run the Diabetic Retinopathy Detection system, which consists of three main components:
- **Frontend**: Next.js web application
- **Backend**: FastAPI server with ML inference
- **ML Training**: PyTorch model training pipeline

## Prerequisites

- **Python 3.11+** (Python 3.13 recommended)
- **Node.js 18+** and npm
- **Git**
- **CUDA** (optional, for GPU training and inference)

## Project Structure

```
diabetic-retinopathy/
├── frontend/           # Next.js web application
├── backend/            # FastAPI server
├── ml/                 # ML training and model files
│   ├── models/         # Trained model files
│   ├── train.py        # Training script
│   └── utils.py        # Utility functions
├── requirements.txt    # Python dependencies (root)
├── backend/requirements.txt  # Backend-specific dependencies
└── RUN.md             # This file
```

## Quick Start

### 1. Clone and Setup

```bash
git clone <repository-url>
cd diabetic-retinopathy
```

### 2. Backend Setup (Python Environment)

#### Create and activate virtual environment:
```bash
python -m venv venv

# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

#### Install Python dependencies:
```bash
# Install backend dependencies
pip install -r backend/requirements.txt

# Install additional ML dependencies (if training)
pip install -r requirements.txt
```

#### Required Model File:
- The backend expects a trained model at: `ml/models/best_model-1.pth`
- Make sure you have this file before starting the backend

### 3. Frontend Setup (Node.js)

```bash
cd frontend
npm install
```

## Running the Application

### 1. Start the Backend API

```bash
# Make sure virtual environment is activated
# From project root directory:
venv\Scripts\activate
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The backend will be available at:
- **API**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

### 2. Start the Frontend

In a new terminal:

```bash
cd frontend
npm run dev
```

The frontend will be available at: **http://localhost:3000**

### 3. Access the Application

1. Open your browser and go to http://localhost:3000
2. Upload a retinal image for diabetic retinopathy detection
3. View prediction results with confidence scores and GradCAM visualization

## ML Model Training (Optional)

### Prerequisites for Training

1. **Dataset**: Prepare your diabetic retinopathy dataset
   - Images should be in a folder (e.g., `data/images/`)
   - CSV file with image names and labels (0-4 for severity levels)

2. **Required Python packages**: All dependencies from `requirements.txt`

### Training Command

```bash
# From project root, with virtual environment activated
cd ml
python train.py --data_csv path/to/your/labels.csv --img_dir path/to/your/images --epochs 50 --batch_size 32
```

### Training Parameters

- `--data_csv`: Path to CSV file with image names and labels
- `--img_dir`: Directory containing training images
- `--epochs`: Number of training epochs (default: 20)
- `--batch_size`: Training batch size (default: 16)
- `--lr`: Learning rate (default: 0.001)
- `--model_name`: Model architecture (default: resnet50)

### Model Output

Trained models are saved in `ml/models/` directory as `best_model-1.pth`

## API Endpoints

### Backend API Endpoints

- **GET /** - Welcome message
- **GET /health** - Health check
- **POST /predict** - Image prediction endpoint
  - Upload: multipart/form-data with image file
  - Response: JSON with prediction, confidence, probabilities, and GradCAM

### Example API Usage

```bash
# Health check
curl http://localhost:8000/health

# Predict (replace with actual image path)
curl -X POST "http://localhost:8000/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@path/to/retinal_image.jpg"
```

## Configuration

### Backend Configuration

Edit `backend/app/config.py`:

```python
MODEL_PATH = 'path/to/your/model.pth'  # Model file path
IMAGE_SIZE = 224                       # Input image size
CLASS_NAMES = ['0', '1', '2', '3', '4'] # DR severity levels
DEVICE = 'cuda' or 'cpu'               # Computation device
```

### Environment Variables

```bash
# Optional environment variables
export MODEL_PATH="/path/to/custom/model.pth"
export IMAGE_SIZE=224
export USE_CUDA=1  # Set to 0 to force CPU usage
```

## Diabetic Retinopathy Classes

The model predicts 5 severity levels:
- **0**: No DR (No Diabetic Retinopathy)
- **1**: Mild DR
- **2**: Moderate DR  
- **3**: Severe DR
- **4**: Proliferative DR

## Troubleshooting

### Common Issues

1. **ImportError: grad-cam not found**
   ```bash
   pip install grad-cam
   ```

2. **Scikit-learn compatibility error**
   ```bash
   pip install scikit-learn>=1.3.0
   ```

3. **Model file not found**
   - Ensure `ml/models/best_model-1.pth` exists
   - Or update `MODEL_PATH` in `backend/app/config.py`

4. **CORS issues**
   - Backend is configured for `localhost:3000`
   - Update CORS settings in `backend/app/main.py` if needed

5. **Node.js version issues**
   ```bash
   node --version  # Should be 18+
   npm install -g npm@latest
   ```

6. **Python version compatibility**
   - Use Python 3.11+ (3.13 recommended)
   - Some packages may not support Python 3.14+

### Performance Tips

1. **GPU Usage**: Ensure CUDA is properly installed for faster inference
2. **Model Loading**: Model is loaded once and cached for subsequent predictions
3. **Image Preprocessing**: Images are automatically resized to 224x224

### Development Mode

- Backend runs with `--reload` for auto-reloading on code changes
- Frontend runs with hot-reloading enabled
- Check browser console and terminal for debug information

## Production Deployment

### Backend
```bash
# Install production server
pip install gunicorn

# Run with gunicorn
cd backend
gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### Frontend
```bash
cd frontend
npm run build
npm start
```

## Additional Notes

- **Model Size**: The trained model file can be large (100MB+)
- **Memory Usage**: GPU inference requires adequate VRAM
- **Image Formats**: Supports common formats (JPG, PNG, etc.)
- **Security**: File upload validation is implemented in the backend

For issues and contributions, please refer to the project repository.