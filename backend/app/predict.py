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

from .config import MODEL_PATH, IMAGE_SIZE, CLASS_NAMES, DEVICE

def load_model(model_path=MODEL_PATH):
    print(f"[INFO] Loading model from: {model_path}")
    checkpoint = torch.load(model_path, map_location=DEVICE)
    model_name = checkpoint.get('model_name', 'efficientnet_b0')
    n_classes = checkpoint.get('n_classes', len(CLASS_NAMES))
    model = timm.create_model(model_name, pretrained=False, num_classes=n_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    model.eval()
    print("[INFO] Model loaded and set to eval mode")
    return model

_MODEL = None
def get_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = load_model()
    return _MODEL

def preprocess_pil(image: Image.Image):
    img = np.array(image.convert('RGB'))
    img_resized = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    img_norm = img_resized.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img_normalized = (img_norm - mean) / std
    tensor = torch.tensor(np.transpose(img_normalized, (2,0,1)), dtype=torch.float).unsqueeze(0).to(DEVICE)
    return tensor, img_norm

def predict(image: Image.Image):
    model = get_model()
    tensor, _ = preprocess_pil(image)
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
        pred_idx = int(np.argmax(probs))
    return pred_idx, float(probs[pred_idx]), probs.tolist()

def find_target_layer(model):
    for name, module in reversed(list(model.named_modules())):
        if isinstance(module, torch.nn.Conv2d):
            return module
    return model

def gradcam_base64(image: Image.Image):
    model = get_model()
    target_layer = find_target_layer(model)
    input_tensor, img_for_overlay = preprocess_pil(image)
    cam = GradCAM(model=model, target_layers=[target_layer], use_cuda=(DEVICE=='cuda'))
    outputs = model(input_tensor)
    grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0]
    visualization = show_cam_on_image(img_for_overlay, grayscale_cam, use_rgb=True)
    _, buffer = cv2.imencode('.png', cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
    b64 = base64.b64encode(buffer).decode('utf-8')
    return b64
