"""
Script to run comprehensive model evaluation
"""
import argparse
import os
import pandas as pd
import torch
from torch.utils.data import DataLoader

from utils import RetinopathyDataset, get_transforms
from evaluation import ModelEvaluator, evaluate_saved_model


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained diabetic retinopathy model')
    parser.add_argument('--model-path', type=str, required=True, 
                       help='Path to trained model checkpoint')
    parser.add_argument('--test-csv', type=str, required=True, 
                       help='Path to test CSV file')
    parser.add_argument('--img-dir', type=str, required=True, 
                       help='Directory containing test images')
    parser.add_argument('--batch-size', type=int, default=32, 
                       help='Batch size for evaluation')
    parser.add_argument('--output-dir', type=str, default='ml/evaluation_results', 
                       help='Directory to save evaluation results')
    parser.add_argument('--image-size', type=int, default=224, 
                       help='Image size for preprocessing')
    parser.add_argument('--num-workers', type=int, default=4, 
                       help='Number of workers for data loading')
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load test data
    print("Loading test data...")
    test_df = pd.read_csv(args.test_csv)
    
    # Rename diagnosis column to label if needed
    if 'diagnosis' in test_df.columns:
        test_df = test_df.rename(columns={'diagnosis': 'label'})
    
    print(f"Test dataset: {len(test_df)} samples")
    print("Label distribution:")
    print(test_df['label'].value_counts().sort_index())
    
    # Create test dataset and dataloader
    _, test_transforms = get_transforms(args.image_size, advanced=True)
    
    test_dataset = RetinopathyDataset(
        test_df, 
        args.img_dir, 
        transforms=test_transforms,
        use_advanced_preprocessing=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Class names for diabetic retinopathy
    class_names = ['No DR', 'Mild DR', 'Moderate DR', 'Severe DR', 'Proliferative DR']
    
    # Run evaluation
    print(f"\nStarting evaluation of model: {args.model_path}")
    metrics = evaluate_saved_model(
        args.model_path, 
        test_loader, 
        device, 
        class_names
    )
    
    # Print summary results
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"Overall Accuracy: {metrics['accuracy']:.4f}")
    print(f"F1-Score (Macro): {metrics['f1_macro']:.4f}")
    print(f"F1-Score (Weighted): {metrics['f1_weighted']:.4f}")
    print(f"Cohen's Kappa: {metrics['kappa']:.4f}")
    print(f"AUC (Macro): {metrics['auc_macro']:.4f}")
    print("\nPer-Class AUC Scores:")
    for class_name, auc in metrics['auc_per_class'].items():
        print(f"  {class_name}: {auc:.4f}")
    
    print("\nClinical Metrics (Sensitivity/Specificity):")
    for class_name, clinical in metrics['clinical_metrics'].items():
        print(f"  {class_name}:")
        print(f"    Sensitivity: {clinical['sensitivity']:.4f}")
        print(f"    Specificity: {clinical['specificity']:.4f}")
        print(f"    Precision: {clinical['precision']:.4f}")
    
    print(f"\nDetailed results saved to: {args.output_dir}")
    print("Files generated:")
    print("  - confusion_matrix.png")
    print("  - confusion_matrix_normalized.png")
    print("  - roc_curves.png")
    print("  - precision_recall_curves.png")
    print("  - class_distribution.png")
    print("  - evaluation_dashboard.html")
    print("  - classification_report.txt")
    print("  - detailed_results.json")


if __name__ == '__main__':
    main()