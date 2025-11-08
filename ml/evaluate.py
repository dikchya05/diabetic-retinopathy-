"""
Comprehensive Model Evaluation Script for Diabetic Retinopathy Detection

This script implements the TESTING PHASE that was missing from the original code.
It evaluates the trained model on a held-out test set with comprehensive metrics.

Features:
- Proper train/val/test split (70/15/15) with stratification
- Comprehensive evaluation metrics for academic reporting
- Visualization of results (confusion matrix, ROC curves)
- Per-class performance analysis
- Generates report-ready outputs

Usage:
    python ml/evaluate.py --labels-csv ml/data/train.csv --img-dir ml/data/train_images --model-path ml/models/best_model-1.pth

Author: Generated for Final Year Project
Date: 2025
"""

import os
import sys
import argparse
import json
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report,
    roc_curve, auc, cohen_kappa_score
)
from sklearn.preprocessing import label_binarize
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import timm

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import custom modules
from ml.utils import RetinopathyDataset, get_transforms


class ModelEvaluator:
    """
    Comprehensive model evaluation class for Diabetic Retinopathy detection
    """

    def __init__(self, model_path, labels_csv, img_dir, output_dir='results'):
        """
        Initialize the evaluator

        Args:
            model_path: Path to trained model checkpoint
            labels_csv: Path to labels CSV file
            img_dir: Directory containing images
            output_dir: Directory to save evaluation results
        """
        self.model_path = model_path
        self.labels_csv = labels_csv
        self.img_dir = img_dir
        self.output_dir = output_dir
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # DR severity class names
        self.class_names = ['No DR', 'Mild DR', 'Moderate DR', 'Severe DR', 'Proliferative DR']

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        print("=" * 80)
        print("DIABETIC RETINOPATHY MODEL EVALUATION")
        print("=" * 80)
        print(f"Model: {model_path}")
        print(f"Device: {self.device}")
        print(f"Output: {output_dir}")
        print("=" * 80)

    def load_model(self):
        """Load the trained model from checkpoint"""
        print("\n📦 Loading model...")

        checkpoint = torch.load(self.model_path, map_location=self.device)

        # Extract model metadata
        if isinstance(checkpoint, dict):
            model_name = checkpoint.get('model_name', 'resnet50')
            n_classes = checkpoint.get('n_classes', 5)
            state_dict = checkpoint.get('model_state_dict', checkpoint)

            print(f"   Architecture: {model_name}")
            print(f"   Classes: {n_classes}")
        else:
            model_name = 'resnet50'  # Default
            n_classes = 5
            state_dict = checkpoint

        # Create model
        model = timm.create_model(model_name, pretrained=False, num_classes=n_classes)
        model.load_state_dict(state_dict, strict=False)
        model = model.to(self.device)
        model.eval()

        print(f"   ✅ Model loaded successfully!\n")
        return model

    def create_data_splits(self, test_size=0.15, val_size=0.15, random_state=42):
        """
        Create train/validation/test splits (70/15/15)

        Args:
            test_size: Proportion for test set (0.15 = 15%)
            val_size: Proportion for validation set (0.15 = 15%)
            random_state: Random seed for reproducibility

        Returns:
            train_df, val_df, test_df
        """
        print("📊 Creating data splits...")

        # Load data
        df = pd.read_csv(self.labels_csv)

        # Handle different CSV column names (rename 'diagnosis' to 'label' if needed)
        if 'diagnosis' in df.columns and 'label' not in df.columns:
            df = df.rename(columns={'diagnosis': 'label'})
            print("   Renamed 'diagnosis' column to 'label'")

        print(f"   Total samples: {len(df)}")

        # Class distribution
        print("\n   Class distribution:")
        class_counts = df['label'].value_counts().sort_index()
        for label, count in class_counts.items():
            print(f"      Class {label} ({self.class_names[label]}): {count} ({count/len(df)*100:.1f}%)")

        # First split: separate test set (15%)
        train_val_df, test_df = train_test_split(
            df,
            test_size=test_size,
            stratify=df['label'],
            random_state=random_state
        )

        # Second split: separate validation from training
        # val_size relative to remaining data
        val_relative_size = val_size / (1 - test_size)
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_relative_size,
            stratify=train_val_df['label'],
            random_state=random_state
        )

        print(f"\n   ✅ Train set: {len(train_df)} samples ({len(train_df)/len(df)*100:.1f}%)")
        print(f"   ✅ Validation set: {len(val_df)} samples ({len(val_df)/len(df)*100:.1f}%)")
        print(f"   ✅ Test set: {len(test_df)} samples ({len(test_df)/len(df)*100:.1f}%)")

        return train_df, val_df, test_df

    def evaluate_on_test_set(self, model, test_df, batch_size=32):
        """
        Evaluate model on test set and collect predictions

        Args:
            model: Trained PyTorch model
            test_df: Test set DataFrame
            batch_size: Batch size for evaluation

        Returns:
            y_true, y_pred, y_probs, image_ids
        """
        print("\n🔬 Evaluating on test set...")

        # Create test dataset and loader
        _, valid_transforms = get_transforms(image_size=224)
        test_dataset = RetinopathyDataset(
            test_df,
            self.img_dir,
            transforms=valid_transforms,
            image_column='id_code',
            label_column='label'
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,  # Set to 0 for Windows compatibility
            pin_memory=True if self.device == 'cuda' else False
        )

        # Collect predictions
        y_true = []
        y_pred = []
        y_probs = []

        model.eval()
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc="   Testing"):
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = model(images)
                probabilities = torch.softmax(outputs, dim=1)
                predictions = outputs.argmax(dim=1)

                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predictions.cpu().numpy())
                y_probs.extend(probabilities.cpu().numpy())

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_probs = np.array(y_probs)
        image_ids = test_df['id_code'].values

        print(f"   ✅ Evaluation complete! {len(y_true)} samples processed.\n")

        return y_true, y_pred, y_probs, image_ids

    def calculate_metrics(self, y_true, y_pred, y_probs):
        """
        Calculate comprehensive evaluation metrics

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_probs: Prediction probabilities

        Returns:
            Dictionary of metrics
        """
        print("📈 Calculating metrics...")

        metrics = {}

        # Overall accuracy
        accuracy = accuracy_score(y_true, y_pred)
        metrics['overall_accuracy'] = float(accuracy)

        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )

        metrics['per_class'] = {}
        for i in range(len(self.class_names)):
            metrics['per_class'][self.class_names[i]] = {
                'precision': float(precision[i]),
                'recall': float(recall[i]),
                'f1_score': float(f1[i]),
                'support': int(support[i])
            }

        # Macro and weighted averages
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            y_true, y_pred, average='macro', zero_division=0
        )
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )

        metrics['macro_avg'] = {
            'precision': float(precision_macro),
            'recall': float(recall_macro),
            'f1_score': float(f1_macro)
        }
        metrics['weighted_avg'] = {
            'precision': float(precision_weighted),
            'recall': float(recall_weighted),
            'f1_score': float(f1_weighted)
        }

        # Cohen's Kappa (important for medical imaging)
        kappa = cohen_kappa_score(y_true, y_pred)
        metrics['cohen_kappa'] = float(kappa)

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()

        # Calculate sensitivity and specificity for each class
        metrics['sensitivity_specificity'] = {}
        for i in range(len(self.class_names)):
            # Sensitivity (True Positive Rate / Recall)
            sensitivity = recall[i]

            # Specificity (True Negative Rate)
            tn = np.sum(cm) - (np.sum(cm[i, :]) + np.sum(cm[:, i]) - cm[i, i])
            fp = np.sum(cm[:, i]) - cm[i, i]
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

            metrics['sensitivity_specificity'][self.class_names[i]] = {
                'sensitivity': float(sensitivity),
                'specificity': float(specificity)
            }

        print("   ✅ Metrics calculated!\n")
        return metrics

    def plot_confusion_matrix(self, y_true, y_pred):
        """Create and save confusion matrix visualization"""
        print("📊 Creating confusion matrix...")

        cm = confusion_matrix(y_true, y_pred)

        # Create figure
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            cbar_kws={'label': 'Count'}
        )
        plt.title('Confusion Matrix - Diabetic Retinopathy Classification', fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()

        # Save
        output_path = os.path.join(self.output_dir, 'confusion_matrix.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"   ✅ Saved to: {output_path}\n")

    def plot_roc_curves(self, y_true, y_probs):
        """Create and save ROC curves for each class"""
        print("📈 Creating ROC curves...")

        # Binarize labels for multi-class ROC
        y_true_bin = label_binarize(y_true, classes=[0, 1, 2, 3, 4])

        # Calculate ROC curve and AUC for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        plt.figure(figsize=(10, 8))

        colors = ['#10b981', '#3b82f6', '#f59e0b', '#f97316', '#ef4444']

        for i in range(len(self.class_names)):
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

            plt.plot(
                fpr[i], tpr[i],
                color=colors[i],
                lw=2,
                label=f'{self.class_names[i]} (AUC = {roc_auc[i]:.3f})'
            )

        # Plot diagonal
        plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves - Diabetic Retinopathy Classification', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()

        # Save
        output_path = os.path.join(self.output_dir, 'roc_curves.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"   ✅ Saved to: {output_path}\n")

        # Return AUC scores
        return {self.class_names[i]: float(roc_auc[i]) for i in range(len(self.class_names))}

    def save_results(self, metrics, auc_scores, y_true, y_pred, y_probs, image_ids):
        """Save all results to files"""
        print("💾 Saving results...")

        # 1. Save comprehensive metrics to JSON
        results = {
            'evaluation_date': datetime.now().isoformat(),
            'model_path': self.model_path,
            'test_set_size': len(y_true),
            'metrics': metrics,
            'auc_scores': auc_scores
        }

        metrics_path = os.path.join(self.output_dir, 'metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"   ✅ Metrics saved to: {metrics_path}")

        # 2. Save classification report
        report = classification_report(
            y_true, y_pred,
            target_names=self.class_names,
            digits=4
        )
        report_path = os.path.join(self.output_dir, 'classification_report.txt')
        with open(report_path, 'w') as f:
            f.write("DIABETIC RETINOPATHY CLASSIFICATION REPORT\n")
            f.write("=" * 80 + "\n\n")
            f.write(report)
            f.write("\n\n" + "=" * 80 + "\n")
            f.write(f"Overall Accuracy: {metrics['overall_accuracy']:.4f}\n")
            f.write(f"Cohen's Kappa: {metrics['cohen_kappa']:.4f}\n")
        print(f"   ✅ Classification report saved to: {report_path}")

        # 3. Save detailed predictions CSV
        predictions_df = pd.DataFrame({
            'image_id': image_ids,
            'true_label': y_true,
            'predicted_label': y_pred,
            'true_class': [self.class_names[i] for i in y_true],
            'predicted_class': [self.class_names[i] for i in y_pred],
            'correct': y_true == y_pred,
            'prob_no_dr': y_probs[:, 0],
            'prob_mild': y_probs[:, 1],
            'prob_moderate': y_probs[:, 2],
            'prob_severe': y_probs[:, 3],
            'prob_proliferative': y_probs[:, 4],
            'confidence': np.max(y_probs, axis=1)
        })

        predictions_path = os.path.join(self.output_dir, 'test_predictions.csv')
        predictions_df.to_csv(predictions_path, index=False)
        print(f"   ✅ Predictions saved to: {predictions_path}")

        # 4. Save summary statistics
        summary_path = os.path.join(self.output_dir, 'summary.txt')
        with open(summary_path, 'w') as f:
            f.write("DIABETIC RETINOPATHY MODEL EVALUATION SUMMARY\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Model: {os.path.basename(self.model_path)}\n")
            f.write(f"Test Set Size: {len(y_true)} images\n\n")

            f.write("OVERALL PERFORMANCE:\n")
            f.write("-" * 80 + "\n")
            f.write(f"Accuracy:      {metrics['overall_accuracy']:.4f} ({metrics['overall_accuracy']*100:.2f}%)\n")
            f.write(f"Cohen's Kappa: {metrics['cohen_kappa']:.4f}\n")
            f.write(f"Macro F1:      {metrics['macro_avg']['f1_score']:.4f}\n")
            f.write(f"Weighted F1:   {metrics['weighted_avg']['f1_score']:.4f}\n\n")

            f.write("PER-CLASS PERFORMANCE:\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'Class':<20} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'AUC':<12} {'Support'}\n")
            f.write("-" * 80 + "\n")

            for class_name in self.class_names:
                class_metrics = metrics['per_class'][class_name]
                auc_score = auc_scores.get(class_name, 0.0)
                f.write(f"{class_name:<20} {class_metrics['precision']:<12.4f} {class_metrics['recall']:<12.4f} "
                       f"{class_metrics['f1_score']:<12.4f} {auc_score:<12.4f} {class_metrics['support']}\n")

            f.write("\n" + "=" * 80 + "\n")
            f.write("Files generated:\n")
            f.write("  - metrics.json (comprehensive metrics)\n")
            f.write("  - classification_report.txt (detailed report)\n")
            f.write("  - test_predictions.csv (all predictions)\n")
            f.write("  - confusion_matrix.png (visualization)\n")
            f.write("  - roc_curves.png (ROC analysis)\n")
            f.write("  - summary.txt (this file)\n")

        print(f"   ✅ Summary saved to: {summary_path}\n")

    def run_evaluation(self):
        """Run complete evaluation pipeline"""
        print("\n🚀 Starting evaluation pipeline...\n")

        # Load model
        model = self.load_model()

        # Create data splits
        train_df, val_df, test_df = self.create_data_splits()

        # Evaluate on test set
        y_true, y_pred, y_probs, image_ids = self.evaluate_on_test_set(model, test_df)

        # Calculate metrics
        metrics = self.calculate_metrics(y_true, y_pred, y_probs)

        # Create visualizations
        self.plot_confusion_matrix(y_true, y_pred)
        auc_scores = self.plot_roc_curves(y_true, y_probs)

        # Save results
        self.save_results(metrics, auc_scores, y_true, y_pred, y_probs, image_ids)

        # Print summary
        print("=" * 80)
        print("EVALUATION COMPLETE!")
        print("=" * 80)
        print(f"\n📊 Overall Accuracy: {metrics['overall_accuracy']:.4f} ({metrics['overall_accuracy']*100:.2f}%)")
        print(f"📊 Cohen's Kappa: {metrics['cohen_kappa']:.4f}")
        print(f"📊 Macro F1-Score: {metrics['macro_avg']['f1_score']:.4f}")
        print(f"\n📁 All results saved to: {self.output_dir}/")
        print("\nFiles generated:")
        print("  ✅ metrics.json")
        print("  ✅ classification_report.txt")
        print("  ✅ test_predictions.csv")
        print("  ✅ confusion_matrix.png")
        print("  ✅ roc_curves.png")
        print("  ✅ summary.txt")
        print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate Diabetic Retinopathy Detection Model on Test Set'
    )

    parser.add_argument(
        '--labels-csv',
        type=str,
        default='ml/data/train.csv',
        help='Path to labels CSV file'
    )

    parser.add_argument(
        '--img-dir',
        type=str,
        default='ml/data/train_images',
        help='Directory containing images'
    )

    parser.add_argument(
        '--model-path',
        type=str,
        default='ml/models/best_model-1.pth',
        help='Path to trained model checkpoint'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='results',
        help='Directory to save evaluation results'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for evaluation'
    )

    args = parser.parse_args()

    # Create evaluator and run
    evaluator = ModelEvaluator(
        model_path=args.model_path,
        labels_csv=args.labels_csv,
        img_dir=args.img_dir,
        output_dir=args.output_dir
    )

    evaluator.run_evaluation()


if __name__ == '__main__':
    main()
