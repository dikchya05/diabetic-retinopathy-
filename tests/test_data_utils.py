"""
Tests for data utilities and preprocessing
"""
import pytest
import numpy as np
import torch
from PIL import Image
import pandas as pd
import os
import sys

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from ml.utils import (
    RetinopathyDataset, get_transforms, preprocess_retinal_image,
    crop_black_borders, assess_image_quality, create_kfold_splits, validate_dataset
)


class TestRetinopathyDataset:
    """Test RetinopathyDataset class"""
    
    def test_dataset_initialization(self, sample_dataframe, temp_image_dir):
        """Test dataset initialization"""
        dataset = RetinopathyDataset(sample_dataframe.head(4), temp_image_dir)
        assert len(dataset) == 4
        assert dataset.df is not None
        assert dataset.img_dir == temp_image_dir
    
    def test_dataset_getitem(self, sample_dataframe, temp_image_dir):
        """Test dataset __getitem__ method"""
        dataset = RetinopathyDataset(sample_dataframe.head(4), temp_image_dir)
        
        # Test getting first item
        image, label = dataset[0]
        assert isinstance(image, np.ndarray)
        assert isinstance(label, int)
        assert 0 <= label <= 4
    
    def test_dataset_with_transforms(self, sample_dataframe, temp_image_dir):
        """Test dataset with transforms"""
        train_transforms, _ = get_transforms(224, advanced=False)
        dataset = RetinopathyDataset(
            sample_dataframe.head(4), 
            temp_image_dir, 
            transforms=train_transforms
        )
        
        image, label = dataset[0]
        assert isinstance(image, torch.Tensor)
        assert image.shape == (3, 224, 224)
    
    def test_dataset_file_not_found(self, sample_dataframe, temp_image_dir):
        """Test dataset handles missing files correctly"""
        # Create dataframe with non-existent image
        bad_df = pd.DataFrame({
            'id_code': ['nonexistent_image'],
            'label': [0]
        })
        
        dataset = RetinopathyDataset(bad_df, temp_image_dir)
        
        with pytest.raises(FileNotFoundError):
            dataset[0]


class TestImagePreprocessing:
    """Test image preprocessing functions"""
    
    def test_preprocess_retinal_image(self, sample_image):
        """Test advanced retinal image preprocessing"""
        processed = preprocess_retinal_image(sample_image, image_size=224)
        
        assert isinstance(processed, np.ndarray)
        assert processed.shape == (224, 224, 3)
        assert processed.dtype == np.uint8
    
    def test_crop_black_borders(self, sample_image):
        """Test black border cropping"""
        # Convert to numpy array
        image_array = np.array(sample_image)
        
        # Add black borders
        bordered_image = np.zeros((300, 300, 3), dtype=np.uint8)
        bordered_image[50:250, 50:250] = image_array[:200, :200]
        
        cropped = crop_black_borders(bordered_image)
        
        # Should be smaller than original
        assert cropped.shape[0] <= bordered_image.shape[0]
        assert cropped.shape[1] <= bordered_image.shape[1]
    
    def test_assess_image_quality_valid_image(self, sample_image, temp_image_dir):
        """Test image quality assessment on valid image"""
        # Save sample image
        img_path = os.path.join(temp_image_dir, 'test_quality.png')
        sample_image.save(img_path)
        
        quality = assess_image_quality(img_path)
        
        assert 'valid' in quality
        if quality['valid']:
            assert 'blur_score' in quality
            assert 'brightness' in quality
    
    def test_assess_image_quality_invalid_path(self):
        """Test image quality assessment on invalid path"""
        quality = assess_image_quality('nonexistent_path.png')
        
        assert quality['valid'] is False
        assert 'reason' in quality


class TestDataTransforms:
    """Test data augmentation transforms"""
    
    def test_get_transforms_basic(self):
        """Test basic transforms"""
        train_transforms, val_transforms = get_transforms(224, advanced=False)
        
        assert train_transforms is not None
        assert val_transforms is not None
    
    def test_get_transforms_advanced(self):
        """Test advanced transforms"""
        train_transforms, val_transforms = get_transforms(224, advanced=True)
        
        assert train_transforms is not None
        assert val_transforms is not None
    
    def test_transforms_output_tensor(self, sample_image):
        """Test that transforms return proper tensors"""
        _, val_transforms = get_transforms(224)
        
        # Convert PIL to numpy for albumentations
        image_array = np.array(sample_image)
        
        transformed = val_transforms(image=image_array)
        tensor_image = transformed['image']
        
        assert isinstance(tensor_image, torch.Tensor)
        assert tensor_image.shape == (3, 224, 224)


class TestDataSplitting:
    """Test data splitting functions"""
    
    def test_create_kfold_splits(self, sample_dataframe):
        """Test k-fold cross-validation splits"""
        splits = create_kfold_splits(sample_dataframe, n_splits=3)
        
        assert len(splits) == 3
        
        for split in splits:
            assert 'fold' in split
            assert 'train' in split
            assert 'val' in split
            assert isinstance(split['train'], pd.DataFrame)
            assert isinstance(split['val'], pd.DataFrame)
            
            # Check no overlap between train and val
            train_ids = set(split['train']['id_code'])
            val_ids = set(split['val']['id_code'])
            assert len(train_ids.intersection(val_ids)) == 0
    
    def test_kfold_splits_preserve_total_size(self, sample_dataframe):
        """Test that k-fold splits preserve total dataset size"""
        splits = create_kfold_splits(sample_dataframe, n_splits=5)
        
        for split in splits:
            total_size = len(split['train']) + len(split['val'])
            assert total_size == len(sample_dataframe)


class TestDataValidation:
    """Test dataset validation functions"""
    
    def test_validate_dataset_basic(self, sample_dataframe, temp_image_dir):
        """Test basic dataset validation"""
        # Use only first 4 samples since we only have 4 images
        validation_result = validate_dataset(
            sample_dataframe.head(4), 
            temp_image_dir, 
            sample_size=4
        )
        
        assert 'valid_ratio' in validation_result
        assert 'issues' in validation_result
        assert 'total_checked' in validation_result
        assert isinstance(validation_result['valid_ratio'], float)
        assert isinstance(validation_result['issues'], list)
        
    def test_validate_dataset_missing_images(self, sample_dataframe, temp_image_dir):
        """Test dataset validation with missing images"""
        # Create dataframe with images that don't exist
        bad_df = pd.DataFrame({
            'id_code': ['missing_1', 'missing_2'],
            'label': [0, 1]
        })
        
        validation_result = validate_dataset(bad_df, temp_image_dir, sample_size=2)
        
        assert validation_result['valid_ratio'] == 0.0
        assert len(validation_result['issues']) > 0


if __name__ == '__main__':
    pytest.main([__file__])