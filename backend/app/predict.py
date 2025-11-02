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
    checkpoint = torch.load(model_path, map_location=DEVICE)
    print("Checkpoint keys:", checkpoint.keys())
    # model_name = checkpoint.get('model_name', 'resnet34')  # fallback model
    model_name = checkpoint.get('model_name', 'resnet50')  # or whatever you used in training

    n_classes = checkpoint.get('n_classes', len(CLASS_NAMES))

    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    print(f"Loading model: {model_name} with {n_classes} classes")
    model = timm.create_model(model_name, pretrained=False, num_classes=n_classes)

    # Optional: Adjust first conv layer input channels if your data is grayscale (1 channel)
    # if model.conv1.in_channels != 3:
    #     import torch.nn as nn
    #     old_conv = model.conv1
    #     new_conv = nn.Conv2d(
    #         1, old_conv.out_channels,
    #         kernel_size=old_conv.kernel_size,
    #         stride=old_conv.stride,
    #         padding=old_conv.padding,
    #         bias=old_conv.bias is not None
    #     )
    #     model.conv1 = new_conv
    #     print("Replaced conv1 to accept 1 channel input.")

    # Load state dict with strict=False to avoid errors if keys mismatch
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys:
        print("Missing keys when loading state_dict:", missing_keys)
    if unexpected_keys:
        print("Unexpected keys when loading state_dict:", unexpected_keys)

    model.to(DEVICE)
    model.eval()
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
    std = np.array([0.229, 0.224, 0.225])
    img_normalized = (img_norm - mean) / std
    tensor = torch.tensor(np.transpose(img_normalized, (2, 0, 1)), dtype=torch.float).unsqueeze(0).to(DEVICE)
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
    """
    Find the appropriate target layer for GradCAM.
    For ResNet models, we target the last layer of layer4.
    For other models, we find the last Conv2d layer.
    """
    # Try to find layer4 (ResNet models)
    if hasattr(model, 'layer4'):
        print("Found layer4 (ResNet architecture)")
        return model.layer4[-1]

    # Try to find the last convolutional block (for timm models)
    if hasattr(model, 'stages'):
        print("Found stages (timm architecture)")
        return model.stages[-1]

    # Fallback: Find last Conv2d layer
    print("Using fallback: searching for last Conv2d layer")
    for name, module in reversed(list(model.named_modules())):
        if isinstance(module, torch.nn.Conv2d):
            print(f"Found Conv2d layer: {name}")
            return module

    print("WARNING: Could not find suitable target layer, using model itself")
    return model


def gradcam_base64(image: Image.Image):
    try:
        print("=== Starting GradCAM generation ===")
        model = get_model()
        print(f"Model loaded: {type(model).__name__}")

        target_layer = find_target_layer(model)
        print(f"Target layer: {target_layer}")

        input_tensor, img_for_overlay = preprocess_pil(image)
        print(f"Input tensor shape: {input_tensor.shape}")
        print(f"Overlay image shape: {img_for_overlay.shape}")

        cam = GradCAM(model=model, target_layers=[target_layer])
        print("GradCAM object created")

        outputs = model(input_tensor)
        print(f"Forward pass complete, output shape: {outputs.shape}")

        grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0]
        print(f"GradCAM computed, shape: {grayscale_cam.shape}")

        visualization = show_cam_on_image(img_for_overlay, grayscale_cam, use_rgb=True)
        print(f"Visualization created, shape: {visualization.shape}")

        _, buffer = cv2.imencode('.png', cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
        b64 = base64.b64encode(buffer).decode('utf-8')
        print(f"GradCAM base64 encoded, length: {len(b64)}")
        print("=== GradCAM generation successful ===")
        return b64
    except Exception as e:
        print(f"ERROR in gradcam_base64: {str(e)}")
        import traceback
        traceback.print_exc()
        raise
