import io
import base64
import numpy as np
from PIL import Image
import torch
import timm
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import cv2
import os
import logging
from pathlib import Path

from .config import settings

logger = logging.getLogger(__name__)


def validate_model_availability():
    """Validate if model is available and loadable"""
    try:
        model_path = Path(settings.MODEL_PATH)
        
        if not model_path.exists():
            return {
                "available": False,
                "error": f"Model file not found at {model_path}"
            }
        
        # Try to load checkpoint metadata
        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            
            # Get state dict
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            
            # Use provided metadata or auto-detect
            if 'model_name' in checkpoint and 'n_classes' in checkpoint:
                model_name = checkpoint['model_name']
                n_classes = checkpoint['n_classes']
            else:
                model_name, n_classes = detect_model_architecture(state_dict)
            
            return {
                "available": True,
                "model_name": model_name,
                "n_classes": n_classes,
                "path": str(model_path),
                "auto_detected": 'model_name' not in checkpoint
            }
        except Exception as e:
            return {
                "available": False,
                "error": f"Cannot load model checkpoint: {str(e)}"
            }
            
    except Exception as e:
        return {
            "available": False,
            "error": f"Model validation failed: {str(e)}"
        }


def detect_model_architecture(state_dict):
    """Detect model architecture from state dict keys"""
    keys = list(state_dict.keys())
    
    # Check for ResNet architecture
    if 'conv1.weight' in keys and any('layer1' in key for key in keys):
        # Determine ResNet variant
        conv1_shape = state_dict['conv1.weight'].shape
        fc_shape = state_dict['fc.weight'].shape if 'fc.weight' in state_dict else None
        
        # Count layer blocks
        layer_counts = {}
        for key in keys:
            if 'layer' in key:
                layer_name = key.split('.')[0]
                if layer_name not in layer_counts:
                    layer_counts[layer_name] = set()
                if '.' in key.split('.', 1)[1]:
                    block_num = key.split('.')[1]
                    layer_counts[layer_name].add(block_num)
        
        # Determine ResNet variant
        if conv1_shape[0] == 64:  # Standard ResNet
            layer_structure = [len(layer_counts.get(f'layer{i}', [])) for i in range(1, 5)]
            if layer_structure == [3, 4, 6, 3]:
                return 'resnet50', fc_shape[0] if fc_shape else 5
            elif layer_structure == [3, 4, 23, 3]:
                return 'resnet101', fc_shape[0] if fc_shape else 5
            elif layer_structure == [3, 8, 36, 3]:
                return 'resnet152', fc_shape[0] if fc_shape else 5
            elif layer_structure == [2, 2, 2, 2]:
                return 'resnet18', fc_shape[0] if fc_shape else 5
            elif layer_structure == [3, 4, 6, 3]:
                return 'resnet34', fc_shape[0] if fc_shape else 5
        
        return 'resnet50', fc_shape[0] if fc_shape else 5  # Default fallback
    
    # Check for EfficientNet architecture
    elif 'conv_stem.weight' in keys and any('blocks.' in key for key in keys):
        classifier_shape = None
        for key in keys:
            if 'classifier.weight' in key:
                classifier_shape = state_dict[key].shape
                break
        return 'efficientnet_b0', classifier_shape[0] if classifier_shape else 5
    
    # Default fallback
    return 'resnet50', 5


def load_model(model_path=None):
    """Load model with automatic architecture detection"""
    if model_path is None:
        model_path = settings.MODEL_PATH
    
    try:
        logger.info(f"Loading model from: {model_path}")
        
        checkpoint = torch.load(model_path, map_location=settings.MODEL_DEVICE)
        logger.debug(f"Checkpoint keys: {list(checkpoint.keys())}")
        
        # Load state dict
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # Auto-detect architecture or use provided info
        if 'model_name' in checkpoint and 'n_classes' in checkpoint:
            model_name = checkpoint['model_name']
            n_classes = checkpoint['n_classes']
            logger.info(f"Using checkpoint metadata: {model_name} with {n_classes} classes")
        else:
            model_name, n_classes = detect_model_architecture(state_dict)
            logger.info(f"Auto-detected architecture: {model_name} with {n_classes} classes")
        
        # Create model
        model = timm.create_model(model_name, pretrained=False, num_classes=n_classes)
        
        # Load with error handling
        try:
            model.load_state_dict(state_dict, strict=True)
            logger.info("Model loaded successfully with strict=True")
        except RuntimeError as e:
            logger.warning(f"Strict loading failed, trying with strict=False: {e}")
            try:
                missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
                
                if missing_keys:
                    logger.warning(f"Missing keys: {missing_keys[:5]}...")  # Show first 5
                if unexpected_keys:
                    logger.warning(f"Unexpected keys: {unexpected_keys[:5]}...")  # Show first 5
                    
                logger.info("Model loaded successfully with strict=False")
            except Exception as e2:
                logger.error(f"Failed to load even with strict=False: {e2}")
                raise RuntimeError(f"Could not load model: {str(e2)}")
        
        # Move to device and set eval mode
        model.to(settings.MODEL_DEVICE)
        model.eval()
        
        logger.info(f"Model loaded successfully on device: {settings.MODEL_DEVICE}")
        return model
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise RuntimeError(f"Model loading failed: {str(e)}")


_MODEL = None


def get_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = load_model()
    return _MODEL


def preprocess_pil(image: Image.Image):
    """Preprocess PIL image for model inference"""
    try:
        # Convert to RGB and numpy array
        img = np.array(image.convert('RGB'))
        
        # Resize to model input size
        img_resized = cv2.resize(img, (settings.IMAGE_SIZE, settings.IMAGE_SIZE))
        
        # Normalize to [0, 1]
        img_norm = img_resized.astype(np.float32) / 255.0
        
        # Apply ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_normalized = (img_norm - mean) / std
        
        # Convert to tensor and add batch dimension
        tensor = torch.tensor(
            np.transpose(img_normalized, (2, 0, 1)), 
            dtype=torch.float
        ).unsqueeze(0).to(settings.MODEL_DEVICE)
        
        return tensor, img_norm
        
    except Exception as e:
        logger.error(f"Image preprocessing failed: {e}")
        raise ValueError(f"Failed to preprocess image: {str(e)}")


def predict(image: Image.Image):
    """Make prediction on image"""
    try:
        model = get_model()
        tensor, _ = preprocess_pil(image)
        
        with torch.no_grad():
            outputs = model(tensor)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
            pred_idx = int(np.argmax(probs))
        
        logger.debug(f"Prediction probabilities: {probs}")
        return pred_idx, float(probs[pred_idx]), probs.tolist()
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise RuntimeError(f"Model prediction failed: {str(e)}")


def find_target_layer(model):
    # Find last Conv2d layer for GradCAM
    for name, module in reversed(list(model.named_modules())):
        if isinstance(module, torch.nn.Conv2d):
            return module
    return model


def gradcam_base64(image: Image.Image):
    """Generate GradCAM visualization as base64 string"""
    try:
        if not settings.ENABLE_GRADCAM:
            logger.debug("GradCAM is disabled")
            return None
            
        model = get_model()
        target_layer = find_target_layer(model)
        
        if target_layer is None:
            logger.warning("No suitable target layer found for GradCAM")
            return None
        
        input_tensor, img_for_overlay = preprocess_pil(image)
        
        # Create GradCAM
        cam = GradCAM(
            model=model, 
            target_layers=[target_layer], 
            use_cuda=(settings.MODEL_DEVICE == 'cuda')
        )
        
        # Generate CAM
        with torch.no_grad():
            outputs = model(input_tensor)
            
        grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0]
        
        # Create visualization
        visualization = show_cam_on_image(img_for_overlay, grayscale_cam, use_rgb=True)
        
        # Encode to base64
        _, buffer = cv2.imencode('.png', cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
        b64 = base64.b64encode(buffer).decode('utf-8')
        
        logger.debug("GradCAM generated successfully")
        return b64
        
    except Exception as e:
        logger.warning(f"GradCAM generation failed: {e}")
        return None
