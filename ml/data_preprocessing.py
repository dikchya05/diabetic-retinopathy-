"""
Advanced data preprocessing for diabetic retinopathy detection
"""
import os
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
import logging
from pathlib import Path
from typing import Tuple, Dict, List, Optional, Union
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class APTOSDataset(Dataset):
    """Enhanced APTOS 2019 dataset with advanced preprocessing"""
    
    def __init__(self, 
                 csv_file: str, 
                 img_dir: str, 
                 transform: Optional[A.Compose] = None,
                 image_size: int = 224,
                 use_cache: bool = True,
                 preprocess_mode: str = 'standard'):
        """
        Args:
            csv_file: Path to CSV file with image names and labels
            img_dir: Directory with all images
            transform: Albumentations transform to apply
            image_size: Target image size
            use_cache: Whether to use image caching
            preprocess_mode: 'standard', 'clahe', 'ben_graham'
        """
        self.data = pd.read_csv(csv_file)
        self.img_dir = Path(img_dir)
        self.transform = transform
        self.image_size = image_size
        self.use_cache = use_cache
        self.preprocess_mode = preprocess_mode
        self.image_cache = {}
        
        logger.info(f"Loaded {len(self.data)} samples from {csv_file}")
        logger.info(f"Using preprocessing mode: {preprocess_mode}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Get image info
        img_name = self.data.iloc[idx]['id_code']
        img_path = self.img_dir / f"{img_name}.png"
        
        # Try .jpg if .png doesn't exist
        if not img_path.exists():
            img_path = self.img_dir / f"{img_name}.jpg"
        
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_name}")
        
        # Load and preprocess image
        if self.use_cache and img_name in self.image_cache:
            image = self.image_cache[img_name].copy()
        else:
            image = self._load_and_preprocess_image(img_path)
            if self.use_cache:
                self.image_cache[img_name] = image.copy()
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        
        # Get label
        label = self.data.iloc[idx]['diagnosis']
        
        return image, label
    
    def _load_and_preprocess_image(self, img_path: Path) -> np.ndarray:
        """Load and preprocess image based on mode"""
        # Load image
        image = cv2.imread(str(img_path))
        if image is None:
            raise ValueError(f"Could not load image: {img_path}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply preprocessing based on mode
        if self.preprocess_mode == 'clahe':
            image = self._apply_clahe(image)
        elif self.preprocess_mode == 'ben_graham':
            image = self._apply_ben_graham_preprocessing(image)
        elif self.preprocess_mode == 'crop_resize':
            image = self._crop_and_resize(image)
        
        return image
    
    def _apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)"""
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[..., 0] = clahe.apply(lab[..., 0])
        
        # Convert back to RGB
        image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        return image
    
    def _apply_ben_graham_preprocessing(self, image: np.ndarray) -> np.ndarray:
        """Apply Ben Graham's preprocessing technique"""
        # Crop black borders
        image = self._crop_black_borders(image)
        
        # Apply Gaussian blur to remove noise
        image = cv2.GaussianBlur(image, (0, 0), 2)
        
        # Apply additive gaussian noise
        noise = np.random.normal(0, 5, image.shape).astype(np.uint8)
        image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        return image
    
    def _crop_black_borders(self, image: np.ndarray) -> np.ndarray:
        """Crop black borders around retinal images"""
        # Convert to grayscale for border detection
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Find contours
        _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find largest contour (should be the eye)
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # Add small margin
            margin = 10
            x = max(0, x - margin)
            y = max(0, y - margin)
            w = min(image.shape[1] - x, w + 2 * margin)
            h = min(image.shape[0] - y, h + 2 * margin)
            
            image = image[y:y+h, x:x+w]
        
        return image
    
    def _crop_and_resize(self, image: np.ndarray) -> np.ndarray:
        """Crop center and resize"""
        # Crop center square
        h, w = image.shape[:2]
        size = min(h, w)
        
        start_h = (h - size) // 2
        start_w = (w - size) // 2
        
        image = image[start_h:start_h+size, start_w:start_w+size]
        
        # Resize to target size
        image = cv2.resize(image, (self.image_size, self.image_size))
        
        return image


class DataPreprocessor:
    """Comprehensive data preprocessing pipeline"""
    
    def __init__(self, 
                 data_dir: str = 'ml/data',
                 image_size: int = 224,
                 preprocess_mode: str = 'standard'):
        """
        Initialize data preprocessor
        
        Args:
            data_dir: Directory containing data files
            image_size: Target image size for training
            preprocess_mode: Preprocessing mode ('standard', 'clahe', 'ben_graham', 'crop_resize')
        """
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.preprocess_mode = preprocess_mode
        self.class_names = ['No DR', 'Mild DR', 'Moderate DR', 'Severe DR', 'Proliferative DR']
        
    def analyze_dataset(self, train_csv: str = 'train.csv') -> Dict:
        """Analyze dataset statistics and class distribution"""
        logger.info("Analyzing dataset...")
        
        train_df = pd.read_csv(self.data_dir / train_csv)
        
        # Basic statistics
        stats = {
            'total_samples': len(train_df),
            'unique_images': train_df['id_code'].nunique(),
            'class_distribution': train_df['diagnosis'].value_counts().sort_index().to_dict(),
            'class_percentages': (train_df['diagnosis'].value_counts(normalize=True).sort_index() * 100).to_dict()
        }
        
        # Check for class imbalance
        stats['imbalance_ratio'] = train_df['diagnosis'].value_counts().max() / train_df['diagnosis'].value_counts().min()
        
        # Image quality analysis (sample-based)
        img_stats = self._analyze_image_quality(train_df.head(100))  # Sample for speed
        stats.update(img_stats)
        
        self._plot_class_distribution(stats['class_distribution'])
        
        return stats
    
    def _analyze_image_quality(self, df_sample: pd.DataFrame) -> Dict:
        """Analyze image quality metrics"""
        logger.info("Analyzing image quality (sample)...")
        
        train_img_dir = self.data_dir / 'train_images'
        quality_metrics = []
        
        def analyze_single_image(img_name):
            try:
                img_path = train_img_dir / f"{img_name}.png"
                if not img_path.exists():
                    img_path = train_img_dir / f"{img_name}.jpg"
                
                if img_path.exists():
                    img = cv2.imread(str(img_path))
                    if img is not None:
                        # Calculate quality metrics
                        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        
                        # Variance of Laplacian (blur detection)
                        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
                        
                        # Brightness
                        brightness = np.mean(gray)
                        
                        # Contrast (standard deviation)
                        contrast = np.std(gray)
                        
                        return {
                            'blur_score': blur_score,
                            'brightness': brightness,
                            'contrast': contrast,
                            'resolution': f"{img.shape[1]}x{img.shape[0]}"
                        }
            except Exception as e:
                logger.warning(f"Could not analyze image {img_name}: {e}")
                return None
        
        # Parallel processing for speed
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(analyze_single_image, name): name 
                      for name in df_sample['id_code'].head(50)}  # Further reduced for speed
            
            for future in as_completed(futures):
                result = future.result()
                if result:
                    quality_metrics.append(result)
        
        if quality_metrics:
            return {
                'avg_blur_score': np.mean([m['blur_score'] for m in quality_metrics]),
                'avg_brightness': np.mean([m['brightness'] for m in quality_metrics]),
                'avg_contrast': np.mean([m['contrast'] for m in quality_metrics]),
                'sample_resolutions': list(set([m['resolution'] for m in quality_metrics]))
            }
        else:
            return {'quality_analysis': 'Failed to analyze images'}
    
    def _plot_class_distribution(self, class_dist: Dict):
        """Plot class distribution"""
        plt.figure(figsize=(12, 5))
        
        # Bar plot
        plt.subplot(1, 2, 1)
        labels = [self.class_names[i] for i in sorted(class_dist.keys())]
        counts = [class_dist[i] for i in sorted(class_dist.keys())]
        
        bars = plt.bar(labels, counts, color='skyblue', edgecolor='darkblue', alpha=0.7)
        plt.title('Class Distribution in Training Set')
        plt.xlabel('Diabetic Retinopathy Severity')
        plt.ylabel('Number of Samples')
        plt.xticks(rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    str(count), ha='center', va='bottom')
        
        # Pie chart
        plt.subplot(1, 2, 2)
        plt.pie(counts, labels=labels, autopct='%1.1f%%', startangle=90, colors=plt.cm.Set3.colors)
        plt.title('Class Distribution (Percentage)')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.data_dir.parent / 'class_distribution.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info(f"Class distribution plot saved to {plot_path}")
    
    def compute_class_weights(self, train_csv: str = 'train.csv') -> torch.Tensor:
        """Compute class weights for handling imbalanced data"""
        train_df = pd.read_csv(self.data_dir / train_csv)
        
        # Compute class weights using sklearn
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(train_df['diagnosis']),
            y=train_df['diagnosis']
        )
        
        # Convert to torch tensor
        class_weights_tensor = torch.FloatTensor(class_weights)
        
        logger.info(f"Computed class weights: {class_weights_tensor}")
        return class_weights_tensor
    
    def create_stratified_splits(self, 
                               train_csv: str = 'train.csv',
                               test_size: float = 0.2,
                               n_splits: int = 5,
                               random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, List]:
        """Create stratified train/val splits and K-fold splits"""
        logger.info("Creating stratified data splits...")
        
        df = pd.read_csv(self.data_dir / train_csv)
        
        # Train/validation split
        train_df, val_df = train_test_split(
            df, 
            test_size=test_size, 
            stratify=df['diagnosis'],
            random_state=random_state
        )
        
        # K-fold splits for cross-validation
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        cv_splits = list(skf.split(df['id_code'], df['diagnosis']))
        
        logger.info(f"Created train: {len(train_df)}, val: {len(val_df)} samples")
        logger.info(f"Created {n_splits}-fold CV splits")
        
        return train_df, val_df, cv_splits
    
    def get_transforms(self, mode: str = 'train') -> A.Compose:
        """Get albumentations transforms for different modes"""
        
        if mode == 'train':
            return A.Compose([
                A.Resize(self.image_size, self.image_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2),
                A.RandomRotate90(p=0.5),
                A.Rotate(limit=15, p=0.3),
                A.RandomBrightnessContrast(
                    brightness_limit=0.2, 
                    contrast_limit=0.2, 
                    p=0.3
                ),
                A.HueSaturationValue(
                    hue_shift_limit=10, 
                    sat_shift_limit=10, 
                    val_shift_limit=10, 
                    p=0.3
                ),
                A.ShiftScaleRotate(
                    shift_limit=0.1, 
                    scale_limit=0.1, 
                    rotate_limit=15, 
                    p=0.3
                ),
                A.CoarseDropout(
                    max_holes=8, 
                    max_height=32, 
                    max_width=32, 
                    p=0.3
                ),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2()
            ])
        
        elif mode in ['val', 'test']:
            return A.Compose([
                A.Resize(self.image_size, self.image_size),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2()
            ])
        
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    def create_data_loaders(self,
                          train_df: pd.DataFrame,
                          val_df: pd.DataFrame,
                          batch_size: int = 32,
                          num_workers: int = 4) -> Tuple[DataLoader, DataLoader]:
        """Create PyTorch data loaders"""
        
        # Create datasets
        train_dataset = APTOSDataset(
            csv_file='dummy.csv',  # Will use df directly
            img_dir=str(self.data_dir / 'train_images'),
            transform=self.get_transforms('train'),
            image_size=self.image_size,
            preprocess_mode=self.preprocess_mode
        )
        train_dataset.data = train_df
        
        val_dataset = APTOSDataset(
            csv_file='dummy.csv',  # Will use df directly
            img_dir=str(self.data_dir / 'train_images'),
            transform=self.get_transforms('val'),
            image_size=self.image_size,
            preprocess_mode=self.preprocess_mode
        )
        val_dataset.data = val_df
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        logger.info(f"Created data loaders: train={len(train_loader)}, val={len(val_loader)} batches")
        
        return train_loader, val_loader
    
    def create_test_loader(self,
                          test_csv: str = 'test.csv',
                          batch_size: int = 32,
                          num_workers: int = 4) -> DataLoader:
        """Create test data loader"""
        
        # For test data, we create dummy labels (will be ignored)
        test_df = pd.read_csv(self.data_dir / test_csv)
        test_df['diagnosis'] = 0  # Dummy labels
        
        test_dataset = APTOSDataset(
            csv_file='dummy.csv',
            img_dir=str(self.data_dir / 'test_images'),
            transform=self.get_transforms('test'),
            image_size=self.image_size,
            preprocess_mode=self.preprocess_mode
        )
        test_dataset.data = test_df
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        logger.info(f"Created test loader: {len(test_loader)} batches")
        return test_loader


def main():
    """Test data preprocessing pipeline"""
    preprocessor = DataPreprocessor(preprocess_mode='clahe')
    
    # Analyze dataset
    stats = preprocessor.analyze_dataset()
    print("\nDataset Analysis:")
    print("=" * 50)
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # Create splits
    train_df, val_df, cv_splits = preprocessor.create_stratified_splits()
    
    # Compute class weights
    class_weights = preprocessor.compute_class_weights()
    
    # Create data loaders
    train_loader, val_loader = preprocessor.create_data_loaders(train_df, val_df, batch_size=16)
    
    # Test data loading
    print(f"\nTesting data loading...")
    for i, (images, labels) in enumerate(train_loader):
        print(f"Batch {i}: images shape {images.shape}, labels shape {labels.shape}")
        if i >= 2:  # Test first few batches
            break
    
    print("Data preprocessing test completed!")


if __name__ == '__main__':
    main()