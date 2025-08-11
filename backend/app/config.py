import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # points to backend/

MODEL_PATH = os.getenv('MODEL_PATH', os.path.join(BASE_DIR, '..', 'ml', 'models', 'best_model.pth'))

IMAGE_SIZE = int(os.getenv('IMAGE_SIZE', 224))
CLASS_NAMES = [str(i) for i in range(5)]
DEVICE = 'cuda' if (os.getenv('USE_CUDA', '1') == '1' and __import__('torch').cuda.is_available()) else 'cpu'
