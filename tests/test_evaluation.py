"""
Tests for model evaluation functionality
"""
import pytest
import numpy as np
import torch
import pandas as pd
import os
import sys
import tempfile
import json

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from ml.evaluation import ModelEvaluator


class TestModelEvaluator:
    """Test ModelEvaluator class"""
    
    def test_evaluator_initialization(self, class_names):
        """Test evaluator initialization"""
        evaluator = ModelEvaluator(class_names)
        
        assert evaluator.class_names == class_names
        assert evaluator.n_classes == len(class_names)
    
    def test_evaluator_default_class_names(self):
        """Test evaluator with default class names"""
        evaluator = ModelEvaluator()
        
        assert evaluator.n_classes == 5
        assert len(evaluator.class_names) == 5
    
    def test_calculate_metrics(self, sample_predictions, class_names):
        """Test metrics calculation"""
        evaluator = ModelEvaluator(class_names)
        
        y_true = sample_predictions['y_true']
        y_pred = sample_predictions['y_pred']
        y_proba = sample_predictions['y_proba']
        
        metrics = evaluator._calculate_metrics(y_true, y_pred, y_proba)
        
        # Check required metrics are present
        assert 'accuracy' in metrics
        assert 'f1_macro' in metrics
        assert 'f1_weighted' in metrics
        assert 'kappa' in metrics
        assert 'auc_macro' in metrics
        assert 'auc_per_class' in metrics
        assert 'clinical_metrics' in metrics
        assert 'confusion_matrix' in metrics
        
        # Check metric value ranges
        assert 0.0 <= metrics['accuracy'] <= 1.0
        assert 0.0 <= metrics['f1_macro'] <= 1.0
        assert 0.0 <= metrics['f1_weighted'] <= 1.0
        assert -1.0 <= metrics['kappa'] <= 1.0
    
    def test_clinical_metrics_calculation(self, class_names):
        """Test clinical metrics (sensitivity, specificity) calculation"""
        evaluator = ModelEvaluator(class_names)
        
        # Create simple test case
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 1, 1, 1, 2, 0])
        y_proba = np.random.rand(6, 5)
        
        metrics = evaluator._calculate_metrics(y_true, y_pred, y_proba)
        
        clinical_metrics = metrics['clinical_metrics']
        
        # Check all classes have clinical metrics
        for class_name in class_names:
            assert class_name in clinical_metrics
            assert 'sensitivity' in clinical_metrics[class_name]
            assert 'specificity' in clinical_metrics[class_name]
            assert 'precision' in clinical_metrics[class_name]
            
            # Check value ranges
            assert 0.0 <= clinical_metrics[class_name]['sensitivity'] <= 1.0
            assert 0.0 <= clinical_metrics[class_name]['specificity'] <= 1.0
            assert 0.0 <= clinical_metrics[class_name]['precision'] <= 1.0


class TestVisualizationGeneration:
    """Test visualization generation"""
    
    def test_generate_confusion_matrix(self, sample_predictions, class_names):
        """Test confusion matrix generation"""
        evaluator = ModelEvaluator(class_names)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            evaluator._generate_confusion_matrix(
                sample_predictions['y_true'],
                sample_predictions['y_pred'],
                temp_dir
            )
            
            # Check files are created
            assert os.path.exists(os.path.join(temp_dir, 'confusion_matrix.png'))
            assert os.path.exists(os.path.join(temp_dir, 'confusion_matrix_normalized.png'))
    
    def test_generate_classification_report(self, sample_predictions, class_names):
        """Test classification report generation"""
        evaluator = ModelEvaluator(class_names)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            evaluator._generate_classification_report(
                sample_predictions['y_true'],
                sample_predictions['y_pred'],
                temp_dir
            )
            
            # Check file is created
            report_path = os.path.join(temp_dir, 'classification_report.txt')
            assert os.path.exists(report_path)
            
            # Check file content
            with open(report_path, 'r') as f:
                content = f.read()
                assert 'DIABETIC RETINOPATHY MODEL EVALUATION REPORT' in content
    
    def test_generate_roc_curves(self, sample_predictions, class_names):
        """Test ROC curves generation"""
        evaluator = ModelEvaluator(class_names)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            evaluator._generate_roc_curves(
                sample_predictions['y_true'],
                sample_predictions['y_proba'],
                temp_dir
            )
            
            # Check file is created
            assert os.path.exists(os.path.join(temp_dir, 'roc_curves.png'))
    
    def test_generate_class_distribution_analysis(self, sample_predictions, class_names):
        """Test class distribution analysis"""
        evaluator = ModelEvaluator(class_names)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            evaluator._generate_class_distribution_analysis(
                sample_predictions['y_true'],
                sample_predictions['y_pred'],
                temp_dir
            )
            
            # Check file is created
            assert os.path.exists(os.path.join(temp_dir, 'class_distribution.png'))


class TestResultsSaving:
    """Test results saving functionality"""
    
    def test_save_detailed_results(self, sample_predictions, class_names):
        """Test detailed results saving"""
        evaluator = ModelEvaluator(class_names)
        
        # Calculate metrics first
        metrics = evaluator._calculate_metrics(
            sample_predictions['y_true'],
            sample_predictions['y_pred'],
            sample_predictions['y_proba']
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            evaluator._save_detailed_results(
                metrics,
                sample_predictions['y_true'],
                sample_predictions['y_pred'],
                sample_predictions['y_proba'],
                temp_dir
            )
            
            # Check JSON file is created
            json_path = os.path.join(temp_dir, 'detailed_results.json')
            assert os.path.exists(json_path)
            
            # Check numpy files are created
            assert os.path.exists(os.path.join(temp_dir, 'y_true.npy'))
            assert os.path.exists(os.path.join(temp_dir, 'y_pred.npy'))
            assert os.path.exists(os.path.join(temp_dir, 'y_proba.npy'))
            
            # Verify JSON content
            with open(json_path, 'r') as f:
                results = json.load(f)
                
                assert 'timestamp' in results
                assert 'model_performance' in results
                assert 'per_class_metrics' in results
                assert 'clinical_metrics' in results
                assert 'confusion_matrix' in results
                assert 'class_names' in results
    
    def test_numpy_files_correctness(self, sample_predictions, class_names):
        """Test that saved numpy files contain correct data"""
        evaluator = ModelEvaluator(class_names)
        
        metrics = evaluator._calculate_metrics(
            sample_predictions['y_true'],
            sample_predictions['y_pred'],
            sample_predictions['y_proba']
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            evaluator._save_detailed_results(
                metrics,
                sample_predictions['y_true'],
                sample_predictions['y_pred'],
                sample_predictions['y_proba'],
                temp_dir
            )
            
            # Load and verify numpy files
            loaded_y_true = np.load(os.path.join(temp_dir, 'y_true.npy'))
            loaded_y_pred = np.load(os.path.join(temp_dir, 'y_pred.npy'))
            loaded_y_proba = np.load(os.path.join(temp_dir, 'y_proba.npy'))
            
            np.testing.assert_array_equal(loaded_y_true, sample_predictions['y_true'])
            np.testing.assert_array_equal(loaded_y_pred, sample_predictions['y_pred'])
            np.testing.assert_array_almost_equal(loaded_y_proba, sample_predictions['y_proba'])


class TestEdgeCases:
    """Test edge cases and error handling"""
    
    def test_single_class_predictions(self, class_names):
        """Test evaluation with single class predictions"""
        evaluator = ModelEvaluator(class_names)
        
        # All predictions are the same class
        y_true = np.array([0, 0, 0, 0])
        y_pred = np.array([0, 0, 0, 0])
        y_proba = np.array([[1.0, 0, 0, 0, 0]] * 4)
        
        metrics = evaluator._calculate_metrics(y_true, y_pred, y_proba)
        
        # Should still return valid metrics
        assert metrics['accuracy'] == 1.0
        assert 'clinical_metrics' in metrics
    
    def test_empty_predictions(self, class_names):
        """Test evaluation with empty predictions"""
        evaluator = ModelEvaluator(class_names)
        
        # Empty arrays
        y_true = np.array([])
        y_pred = np.array([])
        y_proba = np.array([]).reshape(0, 5)
        
        # Should handle gracefully (might raise exception or return NaN values)
        try:
            metrics = evaluator._calculate_metrics(y_true, y_pred, y_proba)
            # If it succeeds, check that NaN values are handled
            assert isinstance(metrics, dict)
        except (ValueError, IndexError):
            # It's acceptable for empty arrays to raise an exception
            pass
    
    def test_mismatched_array_sizes(self, class_names):
        """Test evaluation with mismatched array sizes"""
        evaluator = ModelEvaluator(class_names)
        
        y_true = np.array([0, 1, 2])
        y_pred = np.array([0, 1])  # Different size
        y_proba = np.random.rand(3, 5)
        
        # Should raise an error
        with pytest.raises((ValueError, IndexError)):
            evaluator._calculate_metrics(y_true, y_pred, y_proba)


if __name__ == '__main__':
    pytest.main([__file__])