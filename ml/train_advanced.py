"""
Advanced training script with ensemble models, cross-validation, and advanced techniques
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

from utils import RetinopathyDataset, get_transforms, create_kfold_splits
from models.advanced_architectures import (
    create_advanced_model, get_model_recipe, MODEL_RECIPES
)
from evaluation import ModelEvaluator


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AdvancedTrainer:
    """Advanced trainer with multiple features"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Create output directories
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.models_dir = self.output_dir / 'models'
        self.models_dir.mkdir(exist_ok=True)
        
        self.logs_dir = self.output_dir / 'logs'
        self.logs_dir.mkdir(exist_ok=True)
        
    def setup_data(self, df, img_dir):
        """Setup data loaders"""
        if self.config.use_kfold:
            # K-fold cross-validation
            self.fold_splits = create_kfold_splits(df, n_splits=self.config.n_folds)
            logger.info(f"Created {self.config.n_folds}-fold cross-validation splits")
        else:
            # Simple train-val split
            from sklearn.model_selection import train_test_split
            train_df, val_df = train_test_split(
                df, test_size=0.2, stratify=df['label'], random_state=42
            )
            self.fold_splits = [{'fold': 0, 'train': train_df, 'val': val_df}]
        
        self.img_dir = img_dir
    
    def create_data_loaders(self, train_df, val_df):
        """Create data loaders for a fold"""
        train_transforms, val_transforms = get_transforms(
            self.config.image_size, 
            advanced=self.config.advanced_augmentation
        )
        
        train_dataset = RetinopathyDataset(
            train_df, self.img_dir, transforms=train_transforms,
            use_advanced_preprocessing=self.config.advanced_preprocessing
        )
        
        val_dataset = RetinopathyDataset(
            val_df, self.img_dir, transforms=val_transforms,
            use_advanced_preprocessing=self.config.advanced_preprocessing
        )
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True
        )
        
        return train_loader, val_loader
    
    def create_model(self):
        """Create model based on configuration"""
        if self.config.model_recipe:
            logger.info(f"Using model recipe: {self.config.model_recipe}")
            model = get_model_recipe(self.config.model_recipe, num_classes=self.config.num_classes)
        else:
            logger.info(f"Creating custom model: {self.config.model_type}")
            model_kwargs = {}
            if self.config.backbone:
                model_kwargs['backbone'] = self.config.backbone
            
            model = create_advanced_model(
                self.config.model_type,
                num_classes=self.config.num_classes,
                **model_kwargs
            )
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Model parameters - Total: {total_params:,}, Trainable: {trainable_params:,}")
        
        return model.to(self.device)
    
    def setup_training(self, model, train_size):
        """Setup optimizer, scheduler, and loss function"""
        # Optimizer
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
        
        # Scheduler
        if self.config.scheduler == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.config.epochs
            )
        elif self.config.scheduler == 'reduce_on_plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=3, verbose=True
            )
        elif self.config.scheduler == 'step':
            scheduler = optim.lr_scheduler.StepLR(
                optimizer, step_size=10, gamma=0.1
            )
        else:
            scheduler = None
        
        # Loss function
        if self.config.use_class_weights:
            # Calculate class weights from data
            from ml.models.model import compute_class_weights
            class_weights = compute_class_weights(
                pd.DataFrame({'label': range(self.config.num_classes)}),
                'label'
            ).to(self.device)
            criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            criterion = nn.CrossEntropyLoss()
        
        # Mixed precision training
        scaler = torch.cuda.amp.GradScaler(enabled=self.config.use_amp)
        
        return optimizer, scheduler, criterion, scaler
    
    def train_fold(self, fold_idx, train_df, val_df):
        """Train a single fold"""
        logger.info(f"Training fold {fold_idx}...")
        
        # Create data loaders
        train_loader, val_loader = self.create_data_loaders(train_df, val_df)
        
        # Create model
        model = self.create_model()
        
        # Setup training components
        optimizer, scheduler, criterion, scaler = self.setup_training(model, len(train_df))
        
        # Training history
        history = {
            'train_loss': [], 'val_loss': [], 'val_acc': [], 
            'val_f1': [], 'val_kappa': [], 'learning_rate': []
        }
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_path = self.models_dir / f'best_model_fold_{fold_idx}.pth'
        
        for epoch in range(self.config.epochs):
            # Training
            train_loss = self.train_epoch(model, train_loader, optimizer, criterion, scaler)
            
            # Validation
            val_metrics = self.validate_epoch(model, val_loader, criterion)
            
            # Update history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_metrics['loss'])
            history['val_acc'].append(val_metrics['accuracy'])
            history['val_f1'].append(val_metrics['f1'])
            history['val_kappa'].append(val_metrics['kappa'])
            
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
                f"Fold {fold_idx}, Epoch {epoch+1}/{self.config.epochs}: "
                f"Train Loss: {train_loss:.4f}, Val Loss: {val_metrics['loss']:.4f}, "
                f"Val Acc: {val_metrics['accuracy']:.4f}, Val F1: {val_metrics['f1']:.4f}, "
                f"Val Kappa: {val_metrics['kappa']:.4f}, LR: {current_lr:.6f}"
            )
            
            # Save best model
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                patience_counter = 0
                
                torch.save({
                    'fold': fold_idx,
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_loss': best_val_loss,
                    'val_metrics': val_metrics,
                    'config': self.config.__dict__,
                    'history': history
                }, best_model_path)
                
                logger.info(f"Saved best model for fold {fold_idx}")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= self.config.early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        return history, best_model_path
    
    def train_epoch(self, model, train_loader, optimizer, criterion, scaler):
        """Train one epoch"""
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        with tqdm(train_loader, desc="Training") as pbar:
            for images, labels in pbar:
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                
                with torch.cuda.amp.autocast(enabled=self.config.use_amp):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                
                scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.config.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.grad_clip)
                
                scaler.step(optimizer)
                scaler.update()
                
                total_loss += loss.item()
                num_batches += 1
                
                pbar.set_postfix({'loss': loss.item()})
        
        return total_loss / num_batches
    
    def validate_epoch(self, model, val_loader, criterion):
        """Validate one epoch"""
        model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                with torch.cuda.amp.autocast(enabled=self.config.use_amp):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                
                total_loss += loss.item()
                
                preds = outputs.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
        
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        kappa = cohen_kappa_score(all_labels, all_preds)
        
        return {
            'loss': total_loss / len(val_loader),
            'accuracy': accuracy,
            'f1': f1,
            'kappa': kappa
        }
    
    def train(self, df, img_dir):
        """Main training function"""
        self.setup_data(df, img_dir)
        
        fold_histories = {}
        best_models = {}
        
        for fold_data in self.fold_splits:
            fold_idx = fold_data['fold']
            train_df = fold_data['train']
            val_df = fold_data['val']
            
            logger.info(f"\n{'='*50}")
            logger.info(f"TRAINING FOLD {fold_idx}")
            logger.info(f"Train samples: {len(train_df)}, Val samples: {len(val_df)}")
            logger.info(f"{'='*50}")
            
            history, best_model_path = self.train_fold(fold_idx, train_df, val_df)
            
            fold_histories[fold_idx] = history
            best_models[fold_idx] = str(best_model_path)
        
        # Save training summary
        summary = {
            'config': self.config.__dict__,
            'fold_histories': fold_histories,
            'best_models': best_models,
            'training_completed_at': datetime.now().isoformat()
        }
        
        summary_path = self.output_dir / 'training_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Training completed! Summary saved to: {summary_path}")
        
        # Evaluate each fold
        self.evaluate_folds(best_models)
        
        return fold_histories, best_models
    
    def evaluate_folds(self, best_models):
        """Evaluate all trained folds"""
        logger.info("Evaluating trained models...")
        
        evaluator = ModelEvaluator()
        
        for fold_idx, model_path in best_models.items():
            logger.info(f"Evaluating fold {fold_idx}...")
            
            # Load model
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Create model and load weights
            model = self.create_model()
            model.load_state_dict(checkpoint['model_state_dict'])
            
            # Get validation data for this fold
            fold_data = self.fold_splits[fold_idx]
            val_df = fold_data['val']
            
            # Create validation loader
            _, val_loader = self.create_data_loaders(fold_data['train'], val_df)
            
            # Evaluate
            eval_dir = self.output_dir / f'evaluation_fold_{fold_idx}'
            metrics = evaluator.evaluate_model(model, val_loader, self.device, str(eval_dir))
            
            logger.info(f"Fold {fold_idx} - Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1_macro']:.4f}")


def main():
    parser = argparse.ArgumentParser(description='Advanced training for diabetic retinopathy detection')
    
    # Data arguments
    parser.add_argument('--labels-csv', type=str, required=True, help='Path to labels CSV')
    parser.add_argument('--img-dir', type=str, required=True, help='Directory with images')
    parser.add_argument('--output-dir', type=str, default='training_output', help='Output directory')
    
    # Model arguments
    parser.add_argument('--model-type', type=str, choices=['advanced_cnn', 'vision_transformer', 'hybrid', 'ensemble'],
                       help='Model architecture type')
    parser.add_argument('--model-recipe', type=str, choices=list(MODEL_RECIPES.keys()),
                       help='Pre-configured model recipe')
    parser.add_argument('--backbone', type=str, default='efficientnet_b0', help='CNN backbone')
    parser.add_argument('--num-classes', type=int, default=5, help='Number of classes')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--optimizer', type=str, choices=['adamw', 'sgd'], default='adamw')
    parser.add_argument('--scheduler', type=str, choices=['cosine', 'reduce_on_plateau', 'step'], default='cosine')
    
    # Data processing arguments
    parser.add_argument('--image-size', type=int, default=224, help='Image size')
    parser.add_argument('--advanced-augmentation', action='store_true', help='Use advanced augmentation')
    parser.add_argument('--advanced-preprocessing', action='store_true', help='Use advanced preprocessing')
    
    # Training features
    parser.add_argument('--use-kfold', action='store_true', help='Use k-fold cross-validation')
    parser.add_argument('--n-folds', type=int, default=5, help='Number of folds for cross-validation')
    parser.add_argument('--use-amp', action='store_true', help='Use automatic mixed precision')
    parser.add_argument('--use-class-weights', action='store_true', help='Use class weights for imbalanced data')
    parser.add_argument('--grad-clip', type=float, default=0.0, help='Gradient clipping (0 to disable)')
    parser.add_argument('--early-stopping-patience', type=int, default=10, help='Early stopping patience')
    
    # System arguments
    parser.add_argument('--num-workers', type=int, default=4, help='Number of data loading workers')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.model_type and not args.model_recipe:
        parser.error("Must specify either --model-type or --model-recipe")
    
    # Load data
    df = pd.read_csv(args.labels_csv)
    if 'diagnosis' in df.columns:
        df = df.rename(columns={'diagnosis': 'label'})
    
    logger.info(f"Loaded {len(df)} samples")
    logger.info(f"Label distribution:\n{df['label'].value_counts().sort_index()}")
    
    # Create trainer and start training
    trainer = AdvancedTrainer(args)
    fold_histories, best_models = trainer.train(df, args.img_dir)
    
    logger.info("Advanced training completed successfully!")


if __name__ == '__main__':
    main()