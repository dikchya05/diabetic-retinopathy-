"""
Comprehensive Training Script for Diabetic Retinopathy Detection

This script trains a ResNet50 model with:
- Proper train/validation split (80/20)
- Class-weighted loss (handles imbalanced data)
- Mixed precision training (faster on GPU)
- Early stopping (prevents overfitting)
- Best model checkpointing
- Comprehensive logging

Usage:
    python train_model.py --labels-csv ml/data/train.csv --img-dir ml/data/train_images --epochs 20

Author: Final Year Project
Date: 2025
"""

import os
import argparse
import pandas as pd
import torch

# Import training function from ml.models.model
from ml.models.model import train_loop


def main():
    """Main training function"""

    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Train ResNet50 model for Diabetic Retinopathy Detection',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments
    parser.add_argument(
        '--labels-csv',
        type=str,
        required=True,
        help='Path to CSV file with image labels (e.g., ml/data/train.csv)'
    )

    parser.add_argument(
        '--img-dir',
        type=str,
        required=True,
        help='Directory containing training images (e.g., ml/data/train_images)'
    )

    # Optional arguments
    parser.add_argument(
        '--model-name',
        type=str,
        default='resnet50',
        help='Model architecture (resnet50, resnet34, etc.)'
    )

    parser.add_argument(
        '--epochs',
        type=int,
        default=20,
        help='Number of training epochs'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=16,
        help='Batch size for training (reduce if out of memory)'
    )

    parser.add_argument(
        '--lr',
        type=float,
        default=2e-4,
        help='Learning rate (0.0002 works well for ResNet50)'
    )

    parser.add_argument(
        '--image-size',
        type=int,
        default=224,
        help='Image size (224 is standard for ResNet)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='ml/models',
        help='Directory to save trained models'
    )

    parser.add_argument(
        '--early-stopping-patience',
        type=int,
        default=5,
        help='Stop training if no improvement after N epochs'
    )

    parser.add_argument(
        '--num-workers',
        type=int,
        default=0,
        help='Number of data loading workers (0 for Windows, 4+ for Linux/Mac)'
    )

    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume training from'
    )

    args = parser.parse_args()

    # Print configuration
    print("=" * 80)
    print("DIABETIC RETINOPATHY MODEL TRAINING")
    print("=" * 80)
    print(f"\n📋 Configuration:")
    print(f"   Model: {args.model_name}")
    print(f"   Dataset: {args.labels_csv}")
    print(f"   Images: {args.img_dir}")
    print(f"   Epochs: {args.epochs}")
    print(f"   Batch Size: {args.batch_size}")
    print(f"   Learning Rate: {args.lr}")
    print(f"   Image Size: {args.image_size}x{args.image_size}")
    print(f"   Output: {args.output_dir}")
    print(f"   Early Stopping Patience: {args.early_stopping_patience}")

    # Check CUDA availability
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"   Device: {device}")
    if device == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 80)

    # Load and validate dataset
    print("\n📊 Loading dataset...")

    if not os.path.exists(args.labels_csv):
        print(f"❌ ERROR: CSV file not found: {args.labels_csv}")
        return

    if not os.path.exists(args.img_dir):
        print(f"❌ ERROR: Image directory not found: {args.img_dir}")
        return

    df = pd.read_csv(args.labels_csv)

    # Handle different CSV column names
    if 'diagnosis' in df.columns and 'label' not in df.columns:
        df = df.rename(columns={'diagnosis': 'label'})
        print("   Renamed 'diagnosis' column to 'label'")

    # Validate required columns
    if 'label' not in df.columns:
        print(f"❌ ERROR: CSV must have 'label' column")
        print(f"   Available columns: {df.columns.tolist()}")
        return

    if 'id_code' not in df.columns and df.columns[0] != 'id_code':
        # Try to use first column as image ID
        print(f"   Warning: Using first column '{df.columns[0]}' as image ID")
        df = df.rename(columns={df.columns[0]: 'id_code'})

    print(f"   ✅ Loaded {len(df)} images")
    print(f"   Columns: {df.columns.tolist()}")

    # Show class distribution
    print("\n📈 Class Distribution:")
    class_names = ['No DR', 'Mild DR', 'Moderate DR', 'Severe DR', 'Proliferative DR']
    class_counts = df['label'].value_counts().sort_index()

    for label, count in class_counts.items():
        if label < len(class_names):
            class_name = class_names[label]
        else:
            class_name = f"Class {label}"
        percentage = (count / len(df)) * 100
        print(f"   {class_name}: {count} images ({percentage:.1f}%)")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Start training
    print("\n🚀 Starting training...\n")
    print("=" * 80)

    try:
        model_path = train_loop(
            labels_df=df,
            img_dir=args.img_dir,
            model_name=args.model_name,
            image_size=args.image_size,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=device,
            out_dir=args.output_dir,
            num_workers=args.num_workers,
            resume_checkpoint=args.resume,
            early_stopping_patience=args.early_stopping_patience
        )

        print("\n" + "=" * 80)
        print("✅ TRAINING COMPLETE!")
        print("=" * 80)
        print(f"\n📁 Model saved to: {model_path}")
        print("\n📋 Next Steps:")
        print("   1. Verify model: python ml/check_model.py")
        print(f"   2. Evaluate model: python ml/evaluate.py --model-path {model_path}")
        print("   3. Run inference: Start the backend API")
        print("\n" + "=" * 80)

    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        print("   Partial model may be saved in:", args.output_dir)

    except Exception as e:
        print(f"\n\n❌ ERROR during training: {str(e)}")
        import traceback
        traceback.print_exc()
        print("\n💡 Troubleshooting:")
        print("   - If out of memory: reduce --batch-size (try 8 or 4)")
        print("   - If num_workers error: set --num-workers 0")
        print("   - If image not found: verify --img-dir path")
        print("   - If CUDA error: training will continue on CPU")


if __name__ == '__main__':
    main()
