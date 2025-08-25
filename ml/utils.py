import os
import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import StratifiedKFold
import pandas as pd


def preprocess_retinal_image(image, image_size=224):
    """Advanced preprocessing for retinal images"""
    
    # Convert to numpy array if PIL Image
    if isinstance(image, Image.Image):
        image = np.array(image.convert('RGB'))
    
    # Extract green channel (most informative for retinal images)
    green_channel = image[:, :, 1]
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(green_channel)
    
    # Convert back to RGB by using enhanced green channel
    enhanced_rgb = np.stack([enhanced, enhanced, enhanced], axis=-1)
    
    # Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(enhanced_rgb, (5, 5), 0)
    
    # Crop black borders (common in retinal images)
    cropped = crop_black_borders(blurred)
    
    # Resize to target size
    resized = cv2.resize(cropped, (image_size, image_size))
    
    return resized


def crop_black_borders(image, threshold=10):
    """Remove black borders from retinal images"""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Find non-zero pixels
    coords = cv2.findNonZero(gray > threshold)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        return image[y:y+h, x:x+w]
    return image


def assess_image_quality(image_path):
    """Assess image quality for filtering poor quality images"""
    image = cv2.imread(image_path)
    if image is None:
        return {'valid': False, 'reason': 'Cannot load image'}
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Check if image is too dark (underexposed)
    mean_brightness = np.mean(gray)
    if mean_brightness < 20:
        return {'valid': False, 'reason': 'Too dark/underexposed'}
    
    # Check for blur using Laplacian variance
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    if blur_score < 100:
        return {'valid': False, 'reason': 'Too blurry'}
    
    # Check image size
    height, width = gray.shape
    if min(height, width) < 224:
        return {'valid': False, 'reason': 'Image too small'}
    
    return {'valid': True, 'blur_score': blur_score, 'brightness': mean_brightness}


def create_kfold_splits(df, n_splits=5, random_state=42):
    """Create stratified k-fold splits for cross-validation"""
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    splits = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(df, df['label'])):
        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)
        splits.append({
            'fold': fold,
            'train': train_df,
            'val': val_df
        })
    
    return splits


class RetinopathyDataset(Dataset):
    def __init__(self, df, img_dir, transforms=None, image_column='id_code', 
                 label_column='label', use_advanced_preprocessing=True):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transforms = transforms
        self.image_column = image_column
        self.label_column = label_column
        self.use_advanced_preprocessing = use_advanced_preprocessing

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.loc[idx]
        
        # Try different extensions
        img_path = None
        for ext in ['.png', '.jpg', '.jpeg']:
            potential_path = os.path.join(self.img_dir, f"{row[self.image_column]}{ext}")
            if os.path.exists(potential_path):
                img_path = potential_path
                break
        
        if img_path is None:
            raise FileNotFoundError(f"Image not found for {row[self.image_column]}")
        
        # Load image
        img = Image.open(img_path).convert("RGB")
        
        # Apply advanced preprocessing if enabled
        if self.use_advanced_preprocessing:
            img = preprocess_retinal_image(img)
        else:
            img = np.array(img)
        
        label = int(row[self.label_column])

        # Apply augmentations
        if self.transforms:
            augmented = self.transforms(image=img)
            img = augmented['image']

        return img, label

def get_transforms(image_size=224, advanced=True):
    """Get training and validation transforms"""
    
    if advanced:
        # Advanced augmentations specifically for retinal images
        train_transforms = A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Rotate(limit=10, p=0.3),
            A.OpticalDistortion(distort_limit=0.05, shift_limit=0.05, p=0.3),
            A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.2),
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1),
                A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10),
                A.CLAHE(clip_limit=2.0),
            ], p=0.4),
            A.OneOf([
                A.GaussNoise(var_limit=(10, 50)),
                A.GaussianBlur(blur_limit=(3, 7)),
                A.MotionBlur(blur_limit=5),
            ], p=0.2),
            A.CoarseDropout(max_holes=8, max_height=16, max_width=16, p=0.2),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    else:
        # Basic augmentations
        train_transforms = A.Compose([
            A.RandomResizedCrop(image_size, image_size, scale=(0.8, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.3),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.4),
            A.OneOf([A.GaussNoise(), A.MotionBlur(), A.MedianBlur(blur_limit=3)], p=0.2),
            A.RandomBrightnessContrast(p=0.3),
            A.CLAHE(p=0.2),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    
    valid_transforms = A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    return train_transforms, valid_transforms


def validate_dataset(df, img_dir, sample_size=100):
    """Validate dataset by checking image quality and availability"""
    print(f"Validating dataset with {len(df)} samples...")
    
    # Sample random images for quality check
    sample_df = df.sample(min(sample_size, len(df)), random_state=42)
    
    valid_count = 0
    issues = []
    
    for idx, row in sample_df.iterrows():
        img_id = row['id_code']
        
        # Check if image exists
        img_path = None
        for ext in ['.png', '.jpg', '.jpeg']:
            potential_path = os.path.join(img_dir, f"{img_id}{ext}")
            if os.path.exists(potential_path):
                img_path = potential_path
                break
        
        if img_path is None:
            issues.append(f"Image not found: {img_id}")
            continue
        
        # Assess image quality
        quality = assess_image_quality(img_path)
        if quality['valid']:
            valid_count += 1
        else:
            issues.append(f"Quality issue for {img_id}: {quality['reason']}")
    
    print(f"Validation complete: {valid_count}/{len(sample_df)} images passed quality check")
    
    if issues:
        print("\nFirst 10 issues found:")
        for issue in issues[:10]:
            print(f"  - {issue}")
    
    return {
        'valid_ratio': valid_count / len(sample_df),
        'issues': issues,
        'total_checked': len(sample_df)
    }
