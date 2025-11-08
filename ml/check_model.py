"""
Model Architecture Verification Script

This script checks what model architecture is stored in the trained checkpoint.
Run this to verify the model architecture.

Usage:
    python ml/check_model.py
"""

import os
import torch
import sys

def check_model_architecture(model_path):
    """
    Inspect the model checkpoint and print architecture details

    Args:
        model_path: Path to the model checkpoint file (.pth)
    """
    print("=" * 70)
    print("MODEL ARCHITECTURE VERIFICATION")
    print("=" * 70)

    if not os.path.exists(model_path):
        print(f"❌ ERROR: Model file not found at: {model_path}")
        print("\nPlease ensure the model file exists before running this script.")
        return

    print(f"\n📁 Loading checkpoint from: {model_path}")
    print(f"📊 File size: {os.path.getsize(model_path) / (1024*1024):.2f} MB\n")

    try:
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')

        print("✅ Checkpoint loaded successfully!\n")
        print("-" * 70)
        print("CHECKPOINT CONTENTS:")
        print("-" * 70)

        # Check what keys are in the checkpoint
        if isinstance(checkpoint, dict):
            print(f"Checkpoint keys: {list(checkpoint.keys())}\n")

            # Extract model architecture info
            model_name = checkpoint.get('model_name', 'NOT FOUND')
            n_classes = checkpoint.get('n_classes', 'NOT FOUND')
            image_size = checkpoint.get('image_size', 'NOT FOUND')
            epoch = checkpoint.get('epoch', 'NOT FOUND')
            best_val_loss = checkpoint.get('best_val_loss', 'NOT FOUND')

            print(f"🏗️  Model Architecture: {model_name}")
            print(f"📊 Number of Classes: {n_classes}")
            print(f"🖼️  Image Size: {image_size}x{image_size}")
            print(f"📈 Training Epoch: {epoch}")
            print(f"📉 Best Validation Loss: {best_val_loss}")

            # Determine if model state dict is nested
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                print(f"\n✅ Model weights found in 'model_state_dict'")
            else:
                state_dict = checkpoint
                print(f"\n⚠️  Model weights are at root level (older format)")

            # Analyze state dict structure
            print(f"📦 Total parameters in state_dict: {len(state_dict)}")

            # Show first few layer names to identify architecture
            print("\n🔍 First 10 layer names:")
            for i, key in enumerate(list(state_dict.keys())[:10]):
                print(f"   {i+1}. {key}")

            # Architecture identification
            print("\n" + "=" * 70)
            print("ARCHITECTURE IDENTIFICATION:")
            print("=" * 70)

            first_keys = list(state_dict.keys())[:20]
            first_keys_str = ' '.join(first_keys)

            if model_name != 'NOT FOUND':
                if 'resnet' in model_name.lower():
                    print(f"✅ Model is: ResNet (specifically {model_name})")
                    print("   This MATCHES your report (ResNet50)")
                else:
                    print(f"⚠️  Model is: {model_name}")
                    print("   This DOES NOT MATCH your report (should be ResNet50)")
                    print("\n   RECOMMENDATION: Update model to ResNet50 to match report")
            else:
                # Try to identify from layer names
                if any('layer1' in key or 'layer2' in key or 'layer3' in key or 'layer4' in key for key in first_keys):
                    print("✅ Detected: ResNet architecture (based on layer names)")
                    print("   Likely ResNet50 - MATCHES your report")
                else:
                    print("⚠️  Detected: Different architecture (based on layer names)")
                    print("   DOES NOT MATCH your report (should be ResNet50)")

            print("=" * 70)

        else:
            print("⚠️  Checkpoint is a state_dict only (no metadata)")
            print(f"Total parameters: {len(checkpoint)}")
            print("\nFirst 10 layer names:")
            for i, key in enumerate(list(checkpoint.keys())[:10]):
                print(f"   {i+1}. {key}")

    except Exception as e:
        print(f"❌ ERROR loading checkpoint: {str(e)}")
        import traceback
        traceback.print_exc()
        return

    print("\n" + "=" * 70)
    print("NEXT STEPS:")
    print("=" * 70)

    if model_name != 'NOT FOUND':
        if 'resnet' in model_name.lower():
            print("✅ Your model is ResNet - you're good to go!")
            print("   Run: python ml/evaluate.py to test the model")
        else:
            print("⚠️  Your model is NOT ResNet50")
            print("\nOptions:")
            print("   1. Retrain with ResNet50 (recommended for report consistency)")
            print("   2. Update report to reflect actual architecture")
            print("   3. Proceed with evaluation anyway")
    else:
        print("⚠️  Model metadata not found in checkpoint")
        print("   The model might still work, but architecture is unclear")

    print("=" * 70)


if __name__ == "__main__":
    # Default model path
    default_path = os.path.join('ml', 'models', 'best_model-1.pth')

    # Check if alternative path exists
    alt_path = os.path.join('best_model.pth')

    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    elif os.path.exists(default_path):
        model_path = default_path
    elif os.path.exists(alt_path):
        model_path = alt_path
    else:
        print("❌ No model file found!")
        print(f"   Looked for: {default_path}")
        print(f"   Looked for: {alt_path}")
        print("\nUsage: python ml/check_model.py [path/to/model.pth]")
        sys.exit(1)

    check_model_architecture(model_path)
