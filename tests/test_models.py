"""
Tests for model functionality
"""
import pytest
import torch
import numpy as np
import os
import sys
import tempfile

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from ml.models.model import create_model, compute_class_weights, set_seed


class TestModelCreation:
    """Test model creation and setup"""
    
    def test_create_model_default(self):
        """Test creating model with default parameters"""
        model = create_model()
        assert model is not None
        
        # Test forward pass
        dummy_input = torch.randn(1, 3, 224, 224)
        output = model(dummy_input)
        assert output.shape == (1, 5)  # 5 classes for DR
    
    def test_create_model_custom_classes(self):
        """Test creating model with custom number of classes"""
        model = create_model(n_classes=3)
        
        dummy_input = torch.randn(1, 3, 224, 224)
        output = model(dummy_input)
        assert output.shape == (1, 3)
    
    def test_create_model_different_architecture(self):
        """Test creating different model architectures"""
        model_names = ['efficientnet_b0', 'resnet50']
        
        for model_name in model_names:
            model = create_model(model_name=model_name, pretrained=False)
            assert model is not None
            
            dummy_input = torch.randn(1, 3, 224, 224)
            output = model(dummy_input)
            assert output.shape == (1, 5)
    
    def test_model_training_mode(self):
        """Test model can switch between train/eval modes"""
        model = create_model()
        
        # Test training mode
        model.train()
        assert model.training is True
        
        # Test eval mode
        model.eval()
        assert model.training is False


class TestModelUtilities:
    """Test model utility functions"""
    
    def test_compute_class_weights(self, sample_dataframe):
        """Test class weights computation"""
        weights = compute_class_weights(sample_dataframe, label_col='label')
        
        assert isinstance(weights, torch.Tensor)
        assert len(weights) == 5  # 5 classes
        assert torch.all(weights > 0)  # All weights should be positive
    
    def test_compute_class_weights_imbalanced(self):
        """Test class weights with imbalanced dataset"""
        import pandas as pd
        
        # Create imbalanced dataset
        imbalanced_data = {
            'id_code': [f'img_{i}' for i in range(100)],
            'label': [0] * 80 + [1] * 15 + [2] * 3 + [3] * 1 + [4] * 1
        }
        df = pd.DataFrame(imbalanced_data)
        
        weights = compute_class_weights(df)
        
        # Rare classes should have higher weights
        assert weights[0] < weights[4]  # Class 0 is common, class 4 is rare
        assert weights[1] < weights[3]  # Class 1 is more common than class 3
    
    def test_set_seed_reproducibility(self):
        """Test that set_seed produces reproducible results"""
        # Set seed and generate random tensor
        set_seed(42)
        tensor1 = torch.randn(3, 3)
        
        # Set same seed and generate another tensor
        set_seed(42)
        tensor2 = torch.randn(3, 3)
        
        # They should be identical
        assert torch.allclose(tensor1, tensor2)
    
    def test_set_seed_different_seeds(self):
        """Test that different seeds produce different results"""
        set_seed(42)
        tensor1 = torch.randn(3, 3)
        
        set_seed(123)
        tensor2 = torch.randn(3, 3)
        
        # They should be different
        assert not torch.allclose(tensor1, tensor2)


class TestModelTraining:
    """Test model training utilities"""
    
    def test_model_forward_pass(self, sample_model, sample_batch_tensor):
        """Test model forward pass"""
        output = sample_model(sample_batch_tensor)
        
        assert isinstance(output, torch.Tensor)
        assert output.shape == (4, 5)  # batch_size=4, n_classes=5
    
    def test_model_gradient_computation(self, sample_model, sample_batch_tensor):
        """Test gradient computation"""
        sample_model.train()
        
        # Create dummy labels
        labels = torch.randint(0, 5, (4,))
        
        # Forward pass
        outputs = sample_model(sample_batch_tensor)
        
        # Compute loss
        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        
        # Check that gradients are computed
        has_gradients = False
        for param in sample_model.parameters():
            if param.grad is not None:
                has_gradients = True
                break
        
        assert has_gradients, "Model parameters should have gradients after backward pass"
    
    def test_model_parameter_count(self, sample_model):
        """Test that model has reasonable number of parameters"""
        total_params = sum(p.numel() for p in sample_model.parameters())
        trainable_params = sum(p.numel() for p in sample_model.parameters() if p.requires_grad)
        
        # EfficientNet-B0 should have around 5M parameters
        assert 1_000_000 < total_params < 10_000_000
        assert trainable_params > 0


class TestModelPersistence:
    """Test model saving and loading"""
    
    def test_model_checkpoint_save_load(self, sample_model):
        """Test saving and loading model checkpoint"""
        # Use a temporary file with delete=False to avoid Windows issues
        temp_file = tempfile.NamedTemporaryFile(suffix='.pth', delete=False)
        temp_file.close()  # Close the file handle
        
        try:
            # Save model
            checkpoint = {
                'model_state_dict': sample_model.state_dict(),
                'model_name': 'efficientnet_b0',
                'n_classes': 5,
                'epoch': 10,
                'best_val_loss': 0.5
            }
            torch.save(checkpoint, temp_file.name)
            
            # Load model
            loaded_checkpoint = torch.load(temp_file.name, map_location='cpu')
            
            # Verify checkpoint contents
            assert 'model_state_dict' in loaded_checkpoint
            assert 'model_name' in loaded_checkpoint
            assert loaded_checkpoint['model_name'] == 'efficientnet_b0'
            assert loaded_checkpoint['n_classes'] == 5
        finally:
            # Cleanup
            os.unlink(temp_file.name)
    
    def test_model_state_dict_consistency(self, sample_model):
        """Test model state dict consistency"""
        # Get original state dict
        original_state_dict = sample_model.state_dict()
        
        # Create new model and load state dict
        new_model = create_model(model_name='efficientnet_b0', n_classes=5, pretrained=False)
        new_model.load_state_dict(original_state_dict)
        
        # Compare parameters
        for (name1, param1), (name2, param2) in zip(
            sample_model.named_parameters(), 
            new_model.named_parameters()
        ):
            assert name1 == name2
            assert torch.equal(param1, param2)


if __name__ == '__main__':
    pytest.main([__file__])