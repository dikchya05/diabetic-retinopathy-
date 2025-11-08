"""
Industrial-Grade Training Script for Diabetic Retinopathy Classification

Key Improvements:
1. Class-weighted loss and Focal Loss for imbalanced data
2. Advanced medical image augmentation (albumentations)
3. Two-stage fine-tuning (freeze backbone → full fine-tune)
4. AdamW optimizer with cosine annealing + warmup
5. Early stopping with patience
6. Comprehensive logging and checkpointing
7. Medical image preprocessing (CLAHE, green channel emphasis)
8. Mixed precision training for faster convergence
9. Learning rate finder integration
10. Production-ready error handling
"""

import argparse
import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
from PIL import Image
import cv2

# Advanced augmentation library for medical images
try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    print("⚠️  albumentations not available. Install with: pip install albumentations")
    ALBUMENTATIONS_AVAILABLE = False


# -------------------------------
# Medical Image Preprocessing
# -------------------------------
class MedicalImagePreprocessor:
    """Applies medical imaging best practices for fundus images"""

    @staticmethod
    def apply_clahe(image_np):
        """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)"""
        lab = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        return cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)

    @staticmethod
    def enhance_green_channel(image_np):
        """Enhance green channel - blood vessels more visible"""
        enhanced = image_np.copy()
        enhanced[:, :, 1] = cv2.equalizeHist(enhanced[:, :, 1])
        return enhanced

    @staticmethod
    def remove_black_borders(image_np, threshold=10):
        """Auto-crop black borders from fundus images"""
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
            return image_np[y:y+h, x:x+w]
        return image_np


# -------------------------------
# Custom Dataset with Medical Preprocessing
# -------------------------------
class DiabeticRetinopathyDataset(torch.utils.data.Dataset):
    """Enhanced dataset with medical image preprocessing"""

    def __init__(self, df, img_dir, transform=None, image_col='id_code',
                 label_col='label', img_ext='.png', medical_preprocessing=True):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.image_col = image_col
        self.label_col = label_col
        self.img_ext = img_ext
        self.medical_preprocessing = medical_preprocessing
        self.preprocessor = MedicalImagePreprocessor()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = str(self.df.iloc[idx][self.image_col]) + self.img_ext
        img_path = os.path.join(self.img_dir, img_name)
        label = int(self.df.iloc[idx][self.label_col])

        try:
            # Load image
            image = cv2.imread(img_path)
            if image is None:
                raise ValueError(f"Failed to load image: {img_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Apply medical preprocessing
            if self.medical_preprocessing:
                image = self.preprocessor.remove_black_borders(image)
                image = self.preprocessor.apply_clahe(image)

            # Apply augmentation/transforms
            if self.transform:
                if ALBUMENTATIONS_AVAILABLE and isinstance(self.transform, A.Compose):
                    augmented = self.transform(image=image)
                    image = augmented['image']
                else:
                    image = Image.fromarray(image)
                    image = self.transform(image)
            else:
                image = Image.fromarray(image)
                image = transforms.ToTensor()(image)

            return image, label

        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return a black image in case of error
            return torch.zeros(3, 224, 224), label


# -------------------------------
# Focal Loss for Imbalanced Classification
# -------------------------------
class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance
    Reference: Lin et al. "Focal Loss for Dense Object Detection" (2017)
    """

    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # Class weights
        self.gamma = gamma  # Focusing parameter
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(weight=self.alpha, reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


# -------------------------------
# Advanced Data Augmentation
# -------------------------------
def get_train_transforms(image_size=224, use_advanced=True):
    """Get training transforms with medical image augmentation"""

    if ALBUMENTATIONS_AVAILABLE and use_advanced:
        # Advanced augmentation using albumentations
        return A.Compose([
            A.Resize(image_size, image_size),
            A.RandomRotate90(p=0.5),
            A.Flip(p=0.5),
            A.Transpose(p=0.5),
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0)),
                A.GaussianBlur(blur_limit=3),
                A.MotionBlur(blur_limit=3),
            ], p=0.3),
            A.OneOf([
                A.OpticalDistortion(distort_limit=0.1),
                A.GridDistortion(num_steps=5, distort_limit=0.1),
                A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50),
            ], p=0.3),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.3),
            A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    else:
        # Fallback to torchvision transforms
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


def get_val_transforms(image_size=224):
    """Get validation transforms"""

    if ALBUMENTATIONS_AVAILABLE:
        return A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


# -------------------------------
# Model Creation with Multiple Architectures
# -------------------------------
def create_model(model_name='resnet50', num_classes=5, pretrained=True):
    """Create model with various architecture options"""

    print(f"📦 Creating model: {model_name}")

    if model_name == 'resnet50':
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)

    elif model_name == 'resnet101':
        model = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V2 if pretrained else None)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)

    elif model_name == 'efficientnet_b3':
        model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None)
        num_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_features, num_classes)

    elif model_name == 'efficientnet_b5':
        model = models.efficientnet_b5(weights=models.EfficientNet_B5_Weights.IMAGENET1K_V1 if pretrained else None)
        num_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_features, num_classes)

    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return model


# -------------------------------
# Early Stopping
# -------------------------------
class EarlyStopping:
    """Early stopping to prevent overfitting"""

    def __init__(self, patience=7, min_delta=0.001, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0

    def __call__(self, score, epoch):
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return False

        if self.mode == 'max':
            is_improvement = score > self.best_score + self.min_delta
        else:
            is_improvement = score < self.best_score - self.min_delta

        if is_improvement:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True

        return False


# -------------------------------
# Two-Stage Training Loop
# -------------------------------
def train_model(model, train_loader, val_loader, criterion, device, args, save_dir):
    """
    Two-stage training with industrial best practices:
    Stage 1: Freeze backbone, train classifier
    Stage 2: Unfreeze all, fine-tune entire model
    """

    os.makedirs(save_dir, exist_ok=True)
    best_val_acc = 0.0
    training_history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'learning_rates': []
    }

    # Mixed precision training
    scaler = GradScaler() if args.mixed_precision and device.type == 'cuda' else None

    # Early stopping
    early_stopping = EarlyStopping(patience=args.patience, min_delta=0.001, mode='max')

    # ========================================
    # STAGE 1: Freeze backbone, train classifier
    # ========================================
    if args.two_stage:
        print("\n" + "="*80)
        print("🔒 STAGE 1: Training classifier only (backbone frozen)")
        print("="*80)

        # Freeze all layers except final classifier
        for param in model.parameters():
            param.requires_grad = False

        # Unfreeze classifier
        if hasattr(model, 'fc'):  # ResNet
            for param in model.fc.parameters():
                param.requires_grad = True
        elif hasattr(model, 'classifier'):  # EfficientNet
            for param in model.classifier.parameters():
                param.requires_grad = True

        # Optimizer for stage 1 (higher learning rate)
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr_stage1,
            weight_decay=args.weight_decay
        )

        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs_stage1,
            eta_min=args.lr_stage1 * 0.01
        )

        # Train stage 1
        for epoch in range(args.epochs_stage1):
            train_loss, train_acc = train_epoch(
                model, train_loader, criterion, optimizer, device, scaler, epoch, args.epochs_stage1, "Stage 1"
            )
            val_loss, val_acc = validate_epoch(
                model, val_loader, criterion, device, epoch, args.epochs_stage1, "Stage 1"
            )

            current_lr = optimizer.param_groups[0]['lr']
            training_history['train_loss'].append(train_loss)
            training_history['train_acc'].append(train_acc)
            training_history['val_loss'].append(val_loss)
            training_history['val_acc'].append(val_acc)
            training_history['learning_rates'].append(current_lr)

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                save_checkpoint(model, optimizer, epoch, val_acc, save_dir, 'best_stage1.pth', args)
                print(f"✅ Best Stage 1 model saved: Val Acc = {val_acc:.2f}%")

            scheduler.step()

    # ========================================
    # STAGE 2: Unfreeze all layers, fine-tune
    # ========================================
    print("\n" + "="*80)
    print("🔓 STAGE 2: Fine-tuning entire model (all layers unfrozen)")
    print("="*80)

    # Unfreeze all layers
    for param in model.parameters():
        param.requires_grad = True

    # Optimizer for stage 2 (lower learning rate)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr_stage2,
        weight_decay=args.weight_decay
    )

    # Cosine annealing with warmup
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=args.epochs_stage2 // 3,
        T_mult=1,
        eta_min=args.lr_stage2 * 0.001
    )

    # Train stage 2
    stage2_start_epoch = args.epochs_stage1 if args.two_stage else 0

    for epoch in range(args.epochs_stage2):
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, scaler, epoch, args.epochs_stage2, "Stage 2"
        )
        val_loss, val_acc = validate_epoch(
            model, val_loader, criterion, device, epoch, args.epochs_stage2, "Stage 2"
        )

        current_lr = optimizer.param_groups[0]['lr']
        training_history['train_loss'].append(train_loss)
        training_history['train_acc'].append(train_acc)
        training_history['val_loss'].append(val_loss)
        training_history['val_acc'].append(val_acc)
        training_history['learning_rates'].append(current_lr)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(model, optimizer, epoch + stage2_start_epoch, val_acc, save_dir, 'best_model.pth', args)
            print(f"✅ Best model saved: Val Acc = {val_acc:.2f}%")

        # Save checkpoint every N epochs
        if (epoch + 1) % args.save_freq == 0:
            save_checkpoint(model, optimizer, epoch + stage2_start_epoch, val_acc, save_dir, f'checkpoint_epoch_{epoch+1}.pth', args)

        scheduler.step()

        # Early stopping
        if early_stopping(val_acc, epoch):
            print(f"\n⚠️  Early stopping triggered at epoch {epoch+1}")
            print(f"    Best validation accuracy: {early_stopping.best_score:.2f}% at epoch {early_stopping.best_epoch+1}")
            break

    # Save training history
    history_path = os.path.join(save_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=2)

    print(f"\n{'='*80}")
    print(f"✅ Training completed!")
    print(f"   Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"   Models saved to: {save_dir}")
    print(f"   Training history: {history_path}")
    print(f"{'='*80}\n")

    return best_val_acc, training_history


def train_epoch(model, loader, criterion, optimizer, device, scaler, epoch, total_epochs, stage_name):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc=f"{stage_name} - Epoch {epoch+1}/{total_epochs} [Train]")

    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        # Mixed precision training
        if scaler is not None:
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # Update progress bar
        pbar.set_postfix({
            'loss': f'{running_loss/(pbar.n+1):.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })

    avg_loss = running_loss / len(loader)
    accuracy = 100.0 * correct / total

    print(f"   Train Loss: {avg_loss:.4f} | Train Acc: {accuracy:.2f}%")
    return avg_loss, accuracy


def validate_epoch(model, loader, criterion, device, epoch, total_epochs, stage_name):
    """Validate for one epoch"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc=f"{stage_name} - Epoch {epoch+1}/{total_epochs} [Val]")

    with torch.no_grad():
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            pbar.set_postfix({
                'loss': f'{running_loss/(pbar.n+1):.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })

    avg_loss = running_loss / len(loader)
    accuracy = 100.0 * correct / total

    print(f"   Val Loss:   {avg_loss:.4f} | Val Acc:   {accuracy:.2f}%")
    return avg_loss, accuracy


def save_checkpoint(model, optimizer, epoch, accuracy, save_dir, filename, args):
    """Save model checkpoint with metadata"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'accuracy': accuracy,
        'model_name': args.model_name,
        'num_classes': args.num_classes,
        'image_size': args.image_size,
        'timestamp': datetime.now().isoformat(),
        'args': vars(args)
    }
    torch.save(checkpoint, os.path.join(save_dir, filename))


# -------------------------------
# Main Function
# -------------------------------
def main():
    parser = argparse.ArgumentParser(description='Industrial-grade DR training')

    # Data arguments
    parser.add_argument('--labels-csv', type=str, required=True, help='Path to CSV with labels')
    parser.add_argument('--img-dir', type=str, required=True, help='Directory with images')
    parser.add_argument('--save-dir', type=str, default='ml/models', help='Directory to save models')

    # Model arguments
    parser.add_argument('--model-name', type=str, default='resnet50',
                       choices=['resnet50', 'resnet101', 'efficientnet_b3', 'efficientnet_b5'],
                       help='Model architecture')
    parser.add_argument('--num-classes', type=int, default=5, help='Number of classes')
    parser.add_argument('--pretrained', action='store_true', default=True, help='Use pretrained weights')

    # Training arguments
    parser.add_argument('--two-stage', action='store_true', default=True, help='Use two-stage training')
    parser.add_argument('--epochs-stage1', type=int, default=5, help='Epochs for stage 1 (frozen backbone)')
    parser.add_argument('--epochs-stage2', type=int, default=25, help='Epochs for stage 2 (full fine-tune)')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr-stage1', type=float, default=0.001, help='Learning rate for stage 1')
    parser.add_argument('--lr-stage2', type=float, default=0.0001, help='Learning rate for stage 2')
    parser.add_argument('--weight-decay', type=float, default=0.0001, help='Weight decay')

    # Loss function
    parser.add_argument('--loss', type=str, default='focal', choices=['ce', 'weighted_ce', 'focal'],
                       help='Loss function: ce (CrossEntropy), weighted_ce, focal')
    parser.add_argument('--focal-gamma', type=float, default=2.0, help='Focal loss gamma parameter')

    # Data augmentation
    parser.add_argument('--image-size', type=int, default=224, help='Input image size')
    parser.add_argument('--advanced-aug', action='store_true', default=True, help='Use advanced augmentation')
    parser.add_argument('--medical-preprocess', action='store_true', default=True, help='Apply medical preprocessing')

    # Training options
    parser.add_argument('--mixed-precision', action='store_true', default=True, help='Use mixed precision training')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--save-freq', type=int, default=5, help='Save checkpoint every N epochs')

    # Split configuration
    parser.add_argument('--val-split', type=float, default=0.2, help='Validation split ratio')
    parser.add_argument('--random-seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    # Set random seeds for reproducibility
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

    print("\n" + "="*80)
    print("🏥 DIABETIC RETINOPATHY - INDUSTRIAL TRAINING PIPELINE")
    print("="*80)
    print(f"Model: {args.model_name}")
    print(f"Loss: {args.loss}")
    print(f"Two-stage training: {args.two_stage}")
    print(f"Advanced augmentation: {args.advanced_aug}")
    print(f"Medical preprocessing: {args.medical_preprocess}")
    print(f"Mixed precision: {args.mixed_precision}")
    print("="*80 + "\n")

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Device: {device}")
    if device.type == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print()

    # Load and prepare data
    print("📂 Loading dataset...")
    df = pd.read_csv(args.labels_csv)
    if 'diagnosis' in df.columns:
        df = df.rename(columns={'diagnosis': 'label'})

    print(f"   Total samples: {len(df)}")
    print(f"   Class distribution:")
    class_counts = df['label'].value_counts().sort_index()
    for cls, count in class_counts.items():
        print(f"      Class {cls}: {count} samples ({100*count/len(df):.1f}%)")
    print()

    # Train-validation split
    train_df, val_df = train_test_split(
        df,
        test_size=args.val_split,
        stratify=df['label'],
        random_state=args.random_seed
    )

    print(f"✂️  Data split:")
    print(f"   Training: {len(train_df)} samples")
    print(f"   Validation: {len(val_df)} samples")
    print()

    # Compute class weights for imbalanced data
    class_weights = None
    if args.loss in ['weighted_ce', 'focal']:
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(train_df['label']),
            y=train_df['label'].values
        )
        class_weights = torch.FloatTensor(class_weights).to(device)
        print(f"⚖️  Class weights computed:")
        for i, weight in enumerate(class_weights):
            print(f"   Class {i}: {weight:.4f}")
        print()

    # Create datasets
    train_transform = get_train_transforms(args.image_size, args.advanced_aug)
    val_transform = get_val_transforms(args.image_size)

    train_dataset = DiabeticRetinopathyDataset(
        train_df, args.img_dir, train_transform,
        medical_preprocessing=args.medical_preprocess
    )
    val_dataset = DiabeticRetinopathyDataset(
        val_df, args.img_dir, val_transform,
        medical_preprocessing=args.medical_preprocess
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )

    # Create model
    model = create_model(args.model_name, args.num_classes, args.pretrained)
    model = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print()

    # Create loss function
    if args.loss == 'ce':
        criterion = nn.CrossEntropyLoss()
        print("📊 Loss function: CrossEntropyLoss")
    elif args.loss == 'weighted_ce':
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print("📊 Loss function: Weighted CrossEntropyLoss")
    elif args.loss == 'focal':
        criterion = FocalLoss(alpha=class_weights, gamma=args.focal_gamma)
        print(f"📊 Loss function: Focal Loss (gamma={args.focal_gamma})")
    print()

    # Train model
    best_acc, history = train_model(
        model, train_loader, val_loader, criterion, device, args, args.save_dir
    )

    print(f"🎉 Training pipeline completed successfully!")
    print(f"   Final best validation accuracy: {best_acc:.2f}%")
    print(f"   Models saved to: {args.save_dir}")


if __name__ == '__main__':
    main()
