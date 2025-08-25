"""
Fixed Advanced training script with proper imports and method calls
"""
import argparse
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
import numpy as np
import logging
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Tuple

from .data_preprocessing import DataPreprocessor, APTOSDataset
from .models.advanced_architectures import (
    create_advanced_model, get_model_recipe, MODEL_RECIPES
)
from .evaluation import ModelEvaluator

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrainingConfig:
    """Training configuration with proper defaults"""
    
    def __init__(self, **kwargs):
        # Model configuration
        self.model_type = kwargs.get('model_type', 'advanced_cnn')
        self.model_recipe = kwargs.get('model_recipe', None)
        self.backbone = kwargs.get('backbone', 'efficientnet_b0')
        self.num_classes = kwargs.get('num_classes', 5)
        
        # Data configuration
        self.data_dir = kwargs.get('data_dir', 'ml/data')
        self.image_size = kwargs.get('image_size', 224)
        self.batch_size = kwargs.get('batch_size', 32)
        self.num_workers = kwargs.get('num_workers', 4)
        self.preprocess_mode = kwargs.get('preprocess_mode', 'standard')
        
        # Training configuration
        self.epochs = kwargs.get('epochs', 50)
        self.learning_rate = kwargs.get('learning_rate', 1e-4)
        self.weight_decay = kwargs.get('weight_decay', 1e-5)
        self.optimizer = kwargs.get('optimizer', 'adamw')
        self.scheduler = kwargs.get('scheduler', 'cosine')
        
        # Cross-validation configuration
        self.use_cv = kwargs.get('use_cv', False)
        self.n_folds = kwargs.get('n_folds', 5)
        
        # Advanced options
        self.use_class_weights = kwargs.get('use_class_weights', True)
        self.use_mixed_precision = kwargs.get('use_mixed_precision', False)
        self.early_stopping_patience = kwargs.get('early_stopping_patience', 10)
        
        # Paths
        self.save_dir = kwargs.get('save_dir', 'ml/experiments')
        self.experiment_name = kwargs.get('experiment_name', f'exp_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
        
        # Device
        self.device = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')


class AdvancedTrainer:
    """Fixed advanced trainer with proper method calls"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Create experiment directory
        self.experiment_dir = Path(config.save_dir) / config.experiment_name
        self.models_dir = self.experiment_dir / 'models'
        self.logs_dir = self.experiment_dir / 'logs'
        self.results_dir = self.experiment_dir / 'results'
        
        for dir_path in [self.models_dir, self.logs_dir, self.results_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize data preprocessor
        self.preprocessor = DataPreprocessor(
            data_dir=config.data_dir,
            image_size=config.image_size,
            preprocess_mode=config.preprocess_mode
        )
        
        # Initialize evaluator
        self.evaluator = ModelEvaluator()
        
        logger.info(f"Initialized trainer")
        logger.info(f"Device: {self.device}")
        logger.info(f"Experiment: {self.experiment_dir}")
    
    def _prepare_data(self) -> Tuple[DataLoader, DataLoader]:
        """Prepare training and validation data loaders"""
        logger.info("Preparing data...")
        
        # Create stratified splits
        train_df, val_df, cv_splits = self.preprocessor.create_stratified_splits(
            test_size=0.2,
            n_splits=self.config.n_folds,
            random_state=42
        )
        
        # Store for cross-validation if needed
        self.cv_splits = cv_splits
        self.full_train_df = pd.read_csv(Path(self.config.data_dir) / 'train.csv')
        
        # Create data loaders
        train_loader, val_loader = self.preprocessor.create_data_loaders(
            train_df, val_df,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers
        )
        
        logger.info(f"Training samples: {len(train_df)}")
        logger.info(f"Validation samples: {len(val_df)}")
        
        return train_loader, val_loader
    
    def _setup_model_and_training(self) -> Tuple[nn.Module, nn.Module, optim.Optimizer, Any]:
        """Setup model, loss, optimizer, and scheduler"""
        logger.info("Setting up model and training components...")
        
        # Create model
        if self.config.model_recipe:
            logger.info(f"Using model recipe: {self.config.model_recipe}")
            model = get_model_recipe(self.config.model_recipe, num_classes=self.config.num_classes)
        else:
            logger.info(f"Creating model: {self.config.model_type}")
            model_kwargs = {}
            if hasattr(self.config, 'backbone') and self.config.backbone:
                model_kwargs['backbone'] = self.config.backbone
            
            model = create_advanced_model(
                self.config.model_type,
                num_classes=self.config.num_classes,
                **model_kwargs
            )
        
        model = model.to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Model parameters - Total: {total_params:,}, Trainable: {trainable_params:,}")
        
        # Setup loss function with class weights if requested
        if self.config.use_class_weights:
            class_weights = self.preprocessor.compute_class_weights().to(self.device)
            criterion = nn.CrossEntropyLoss(weight=class_weights)
            logger.info(f"Using class weights: {class_weights.cpu().numpy()}")
        else:
            criterion = nn.CrossEntropyLoss()
        
        # Setup optimizer
        if self.config.optimizer == 'adamw':
            optimizer = optim.AdamW(
                model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer == 'sgd':
            optimizer = optim.SGD(
                model.parameters(),
                lr=self.config.learning_rate,
                momentum=0.9,
                weight_decay=self.config.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")
        
        # Setup scheduler
        scheduler = None
        if self.config.scheduler == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.config.epochs
            )
        elif self.config.scheduler == 'reduce_on_plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', patience=5, factor=0.5
            )
        
        return model, criterion, optimizer, scheduler
    
    def _train_epoch(self, model: nn.Module, dataloader: DataLoader, 
                    optimizer: optim.Optimizer, criterion: nn.Module) -> float:
        """Train for one epoch"""
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(dataloader, desc="Training", leave=False)
        
        for batch_idx, (images, labels) in enumerate(progress_bar):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss/num_batches:.4f}'
            })
        
        return total_loss / num_batches
    
    def _validate_epoch(self, model: nn.Module, dataloader: DataLoader, 
                       criterion: nn.Module) -> Dict[str, float]:
        """Validate for one epoch"""
        model.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for images, labels in tqdm(dataloader, desc="Validation", leave=False):
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                
                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                # Get predictions and probabilities
                probabilities = torch.softmax(outputs, dim=1)
                predictions = torch.argmax(probabilities, dim=1)
                
                # Store results
                total_loss += loss.item()
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # Calculate metrics
        avg_loss = total_loss / len(dataloader)
        accuracy = np.mean(np.array(all_predictions) == np.array(all_labels))
        
        # Calculate additional metrics using evaluator
        metrics = self.evaluator._calculate_metrics(
            np.array(all_labels),
            np.array(all_predictions), 
            np.array(all_probabilities)
        )
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'f1_macro': metrics['f1_macro'],
            'f1_weighted': metrics['f1_weighted'],
            'kappa': metrics['kappa']
        }
    
    def train_single_fold(self, fold_idx: int = 0) -> Dict[str, Any]:
        """Train a single fold"""
        logger.info(f"Starting training for fold {fold_idx}")
        
        # Prepare data
        if self.config.use_cv and hasattr(self, 'cv_splits'):
            # Use cross-validation split
            train_idx, val_idx = self.cv_splits[fold_idx]
            train_df = self.full_train_df.iloc[train_idx].reset_index(drop=True)
            val_df = self.full_train_df.iloc[val_idx].reset_index(drop=True)
            
            train_loader, val_loader = self.preprocessor.create_data_loaders(
                train_df, val_df,
                batch_size=self.config.batch_size,
                num_workers=self.config.num_workers
            )
        else:
            # Use regular train/val split
            train_loader, val_loader = self._prepare_data()
        
        # Setup model and training components
        model, criterion, optimizer, scheduler = self._setup_model_and_training()
        
        # Training history
        history = {
            'train_loss': [], 'val_loss': [], 'val_accuracy': [],
            'val_f1_macro': [], 'val_f1_weighted': [], 'val_kappa': [], 'learning_rate': []
        }
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_path = self.models_dir / f'best_model_fold_{fold_idx}.pth'
        
        # Training loop
        for epoch in range(self.config.epochs):
            logger.info(f"Epoch {epoch + 1}/{self.config.epochs}")
            
            # Training
            train_loss = self._train_epoch(model, train_loader, optimizer, criterion)
            
            # Validation
            val_metrics = self._validate_epoch(model, val_loader, criterion)
            
            # Update history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_metrics['loss'])
            history['val_accuracy'].append(val_metrics['accuracy'])
            history['val_f1_macro'].append(val_metrics['f1_macro'])
            history['val_f1_weighted'].append(val_metrics['f1_weighted'])
            history['val_kappa'].append(val_metrics['kappa'])
            
            # Learning rate
            current_lr = optimizer.param_groups[0]['lr']
            history['learning_rate'].append(current_lr)
            
            # Scheduler step
            if scheduler:
                if self.config.scheduler == 'reduce_on_plateau':
                    scheduler.step(val_metrics['loss'])
                else:
                    scheduler.step()
            
            # Logging
            logger.info(
                f"Fold {fold_idx}, Epoch {epoch+1}: "
                f"Train Loss: {train_loss:.4f}, Val Loss: {val_metrics['loss']:.4f}, "
                f"Val Acc: {val_metrics['accuracy']:.4f}, Val F1: {val_metrics['f1_macro']:.4f}, "
                f"LR: {current_lr:.6f}"
            )
            
            # Early stopping and model saving
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                patience_counter = 0
                
                # Save best model
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_metrics['loss'],
                    'val_accuracy': val_metrics['accuracy'],
                    'val_f1_macro': val_metrics['f1_macro'],
                    'config': vars(self.config),
                    'model_name': getattr(self.config, 'backbone', 'efficientnet_b0'),
                    'n_classes': self.config.num_classes
                }
                
                torch.save(checkpoint, best_model_path)
                logger.info(f"Saved best model with val_loss: {val_metrics['loss']:.4f}")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= self.config.early_stopping_patience:
                logger.info(f"Early stopping after {epoch + 1} epochs")
                break
        
        # Save training history
        history_path = self.results_dir / f'history_fold_{fold_idx}.json'
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        # Final evaluation on best model
        logger.info("Loading best model for final evaluation...")
        checkpoint = torch.load(best_model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        final_metrics = self._validate_epoch(model, val_loader, criterion)
        
        return {
            'fold': fold_idx,
            'best_val_loss': best_val_loss,
            'final_metrics': final_metrics,
            'history': history,
            'model_path': str(best_model_path)
        }
    
    def train(self) -> Dict[str, Any]:
        """Main training function"""
        logger.info("Starting training...")
        
        if self.config.use_cv:
            # Cross-validation training
            logger.info(f"Running {self.config.n_folds}-fold cross-validation")
            
            # Prepare CV splits
            train_df, _, cv_splits = self.preprocessor.create_stratified_splits(
                n_splits=self.config.n_folds,
                random_state=42
            )
            self.cv_splits = cv_splits
            self.full_train_df = pd.read_csv(Path(self.config.data_dir) / 'train.csv')
            
            fold_results = []
            for fold_idx in range(self.config.n_folds):
                logger.info(f"\n{'='*50}")
                logger.info(f"FOLD {fold_idx + 1}/{self.config.n_folds}")
                logger.info(f"{'='*50}")
                
                fold_result = self.train_single_fold(fold_idx)
                fold_results.append(fold_result)
            
            # Aggregate results
            avg_val_loss = np.mean([r['best_val_loss'] for r in fold_results])
            avg_val_acc = np.mean([r['final_metrics']['accuracy'] for r in fold_results])
            avg_val_f1 = np.mean([r['final_metrics']['f1_macro'] for r in fold_results])
            
            logger.info(f"\nCross-Validation Results:")
            logger.info(f"Average Val Loss: {avg_val_loss:.4f}")
            logger.info(f"Average Val Accuracy: {avg_val_acc:.4f}")
            logger.info(f"Average Val F1: {avg_val_f1:.4f}")
            
            # Save aggregate results
            cv_results = {
                'cv_results': fold_results,
                'average_metrics': {
                    'val_loss': avg_val_loss,
                    'val_accuracy': avg_val_acc,
                    'val_f1_macro': avg_val_f1
                },
                'config': vars(self.config)
            }
            
            results_path = self.results_dir / 'cv_results.json'
            with open(results_path, 'w') as f:
                json.dump(cv_results, f, indent=2, default=str)
            
            return cv_results
        
        else:
            # Single fold training
            result = self.train_single_fold(fold_idx=0)
            
            # Save results
            results_path = self.results_dir / 'training_results.json'
            with open(results_path, 'w') as f:
                json.dump(result, f, indent=2, default=str)
            
            return result


def create_config_from_args(args) -> TrainingConfig:
    """Create training config from command line arguments"""
    config_dict = {}
    
    # Model arguments
    if args.model_type:
        config_dict['model_type'] = args.model_type
    if args.recipe:
        config_dict['model_recipe'] = args.recipe
    if args.backbone:
        config_dict['backbone'] = args.backbone
    
    # Training arguments
    if args.epochs:
        config_dict['epochs'] = args.epochs
    if args.batch_size:
        config_dict['batch_size'] = args.batch_size
    if args.learning_rate:
        config_dict['learning_rate'] = args.learning_rate
    if args.optimizer:
        config_dict['optimizer'] = args.optimizer
    if args.scheduler:
        config_dict['scheduler'] = args.scheduler
    
    # Cross-validation
    if args.use_cv:
        config_dict['use_cv'] = True
        config_dict['n_folds'] = args.n_folds or 5
    
    # Other options
    if args.data_dir:
        config_dict['data_dir'] = args.data_dir
    if args.save_dir:
        config_dict['save_dir'] = args.save_dir
    if args.image_size:
        config_dict['image_size'] = args.image_size
    
    return TrainingConfig(**config_dict)


def main():
    parser = argparse.ArgumentParser(description='Advanced Training for Diabetic Retinopathy Detection')
    
    # Model arguments
    parser.add_argument('--model-type', choices=['advanced_cnn', 'vision_transformer', 'hybrid', 'ensemble'],
                       help='Model architecture type')
    parser.add_argument('--recipe', choices=['lightweight', 'high_performance', 'transformer_based', 'hybrid_best'],
                       help='Pre-configured model recipe')
    parser.add_argument('--backbone', help='Backbone model name (e.g., efficientnet_b0)')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, help='Training batch size')
    parser.add_argument('--learning-rate', type=float, help='Initial learning rate')
    parser.add_argument('--optimizer', choices=['adamw', 'sgd'], help='Optimizer type')
    parser.add_argument('--scheduler', choices=['cosine', 'reduce_on_plateau'], help='Learning rate scheduler')
    
    # Cross-validation
    parser.add_argument('--use-cv', action='store_true', help='Use cross-validation')
    parser.add_argument('--n-folds', type=int, help='Number of CV folds')
    
    # Data and paths
    parser.add_argument('--data-dir', help='Directory containing dataset')
    parser.add_argument('--save-dir', help='Directory to save results')
    parser.add_argument('--image-size', type=int, help='Input image size')
    
    args = parser.parse_args()
    
    # Create configuration
    config = create_config_from_args(args)
    
    # Initialize trainer
    trainer = AdvancedTrainer(config)
    
    # Start training
    results = trainer.train()
    
    logger.info("Training completed successfully!")
    logger.info(f"Results saved to: {trainer.results_dir}")


if __name__ == '__main__':
    main()