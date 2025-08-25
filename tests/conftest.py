"""
Pytest configuration and fixtures for diabetic retinopathy tests
"""
import pytest
import os
import numpy as np
import pandas as pd
import torch
from PIL import Image
import tempfile
import shutil


@pytest.fixture
def sample_image():
    """Create a sample retinal image for testing"""
    # Create a synthetic retinal image (3-channel RGB, 224x224)
    image_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    # Add some circular pattern to simulate retinal structure
    center = (112, 112)
    y, x = np.ogrid[:224, :224]
    mask = (x - center[0])**2 + (y - center[1])**2 <= 100**2
    image_array[mask] = [255, 0, 0]  # Red center
    return Image.fromarray(image_array)


@pytest.fixture
def sample_batch_images():
    """Create a batch of sample images"""
    images = []
    for i in range(4):
        image_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        images.append(Image.fromarray(image_array))
    return images


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame with image IDs and labels"""
    data = {
        'id_code': [f'image_{i:04d}' for i in range(100)],
        'label': np.random.randint(0, 5, 100)
    }
    return pd.DataFrame(data)


@pytest.fixture
def temp_image_dir(sample_batch_images):
    """Create temporary directory with sample images"""
    temp_dir = tempfile.mkdtemp()
    
    # Save sample images
    for i, img in enumerate(sample_batch_images):
        img.save(os.path.join(temp_dir, f'image_{i:04d}.png'))
    
    yield temp_dir
    
    # Cleanup
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_model():
    """Create a simple model for testing"""
    import timm
    model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=5)
    model.eval()
    return model


@pytest.fixture
def sample_tensor():
    """Create a sample tensor for testing"""
    return torch.randn(1, 3, 224, 224)


@pytest.fixture
def sample_batch_tensor():
    """Create a sample batch tensor for testing"""
    return torch.randn(4, 3, 224, 224)


@pytest.fixture
def sample_predictions():
    """Create sample model predictions"""
    return {
        'y_true': np.array([0, 1, 2, 3, 4, 0, 1, 2]),
        'y_pred': np.array([0, 1, 1, 3, 3, 1, 1, 2]),
        'y_proba': np.random.rand(8, 5)
    }


@pytest.fixture
def temp_model_path():
    """Create a temporary model checkpoint"""
    temp_file = tempfile.NamedTemporaryFile(suffix='.pth', delete=False)
    
    # Create a dummy checkpoint
    checkpoint = {
        'model_state_dict': {'dummy': torch.tensor([1.0])},
        'model_name': 'efficientnet_b0',
        'n_classes': 5,
        'epoch': 10,
        'best_val_loss': 0.5
    }
    
    torch.save(checkpoint, temp_file.name)
    temp_file.close()
    
    yield temp_file.name
    
    # Cleanup
    os.unlink(temp_file.name)


@pytest.fixture
def class_names():
    """Standard class names for diabetic retinopathy"""
    return ['No DR', 'Mild DR', 'Moderate DR', 'Severe DR', 'Proliferative DR']


# Additional fixtures for backend testing
@pytest.fixture
def mock_get_model():
    """Mock get_model function for testing"""
    from unittest.mock import Mock, patch
    mock_model = Mock()
    mock_model.return_value = torch.tensor([[0.1, 0.2, 0.5, 0.15, 0.05]])
    
    with patch('backend.app.predict.get_model', return_value=mock_model):
        yield mock_model


@pytest.fixture
def mock_predict():
    """Mock predict function for testing"""
    from unittest.mock import patch
    
    def mock_predict_func(image):
        return 2, 0.75, [0.1, 0.15, 0.75, 0.0, 0.0]
    
    with patch('backend.app.predict.predict', side_effect=mock_predict_func):
        yield mock_predict_func


# Set up random seeds for reproducible tests
@pytest.fixture(autouse=True)
def setup_random_seeds():
    """Set random seeds for reproducible testing"""
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)