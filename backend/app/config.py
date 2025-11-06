import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # points to backend/

MODEL_PATH = os.getenv('MODEL_PATH', os.path.join(BASE_DIR, '..', 'ml', 'models', 'best_model.pth'))
# MODEL_PATH = os.getenv('MODEL_PATH', os.path.join(BASE_DIR, '..', 'ml', 'models', 'best_model-1.pth'))

# Model Configuration
# Using ResNet50 architecture as specified in the final year report
# The model is trained on 5 classes of Diabetic Retinopathy severity (0-4)

print("dbcvkdjbvcdkjvcbdjkv", MODEL_PATH)
IMAGE_SIZE = int(os.getenv('IMAGE_SIZE', 224))
CLASS_NAMES = [str(i) for i in range(5)]  # DR severity: 0=No DR, 1=Mild, 2=Moderate, 3=Severe, 4=Proliferative
DEVICE = 'cuda' if (os.getenv('USE_CUDA', '1') == '1' and __import__('torch').cuda.is_available()) else 'cpu'
