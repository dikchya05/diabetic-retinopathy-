"""
Comprehensive model evaluation for diabetic retinopathy detection
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_auc_score, 
    roc_curve, precision_recall_curve, f1_score, cohen_kappa_score
)
from sklearn.preprocessing import label_binarize
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
from datetime import datetime


class ModelEvaluator:
    """Comprehensive model evaluation class"""
    
    def __init__(self, class_names=None):
        if class_names is None:
            self.class_names = ['No DR', 'Mild DR', 'Moderate DR', 'Severe DR', 'Proliferative DR']
        else:
            self.class_names = class_names
        self.n_classes = len(self.class_names)
    
    def evaluate_model(self, model, dataloader, device='cpu', save_dir='ml/evaluation_results'):
        """Complete model evaluation pipeline"""
        print("Starting comprehensive model evaluation...")
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Get predictions and ground truth
        y_true, y_pred, y_proba = self._get_predictions(model, dataloader, device)
        
        # Basic metrics
        metrics = self._calculate_metrics(y_true, y_pred, y_proba)
        
        # Generate all visualizations
        self._generate_confusion_matrix(y_true, y_pred, save_dir)
        self._generate_classification_report(y_true, y_pred, save_dir)
        self._generate_roc_curves(y_true, y_proba, save_dir)
        self._generate_precision_recall_curves(y_true, y_proba, save_dir)
        self._generate_class_distribution_analysis(y_true, y_pred, save_dir)
        self._generate_interactive_dashboard(y_true, y_pred, y_proba, metrics, save_dir)
        
        # Save detailed results
        self._save_detailed_results(metrics, y_true, y_pred, y_proba, save_dir)
        
        print(f"Evaluation complete! Results saved to {save_dir}")
        return metrics
    
    def _get_predictions(self, model, dataloader, device):
        """Get model predictions and probabilities"""
        model.eval()
        y_true = []
        y_pred = []
        y_proba = []
        
        print("Generating predictions...")
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(dataloader):
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = model(images)
                probabilities = F.softmax(outputs, dim=1)
                predictions = torch.argmax(probabilities, dim=1)
                
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predictions.cpu().numpy())
                y_proba.extend(probabilities.cpu().numpy())
                
                if batch_idx % 10 == 0:
                    print(f"Processed {batch_idx * len(images)}/{len(dataloader.dataset)} samples")
        
        return np.array(y_true), np.array(y_pred), np.array(y_proba)
    
    def _calculate_metrics(self, y_true, y_pred, y_proba):
        """Calculate comprehensive metrics"""
        print("Calculating metrics...")
        
        # Basic metrics
        accuracy = np.mean(y_true == y_pred)
        f1_macro = f1_score(y_true, y_pred, average='macro')
        f1_weighted = f1_score(y_true, y_pred, average='weighted')
        kappa = cohen_kappa_score(y_true, y_pred)
        
        # Per-class metrics
        report = classification_report(y_true, y_pred, target_names=self.class_names, output_dict=True)
        
        # ROC AUC scores
        y_true_bin = label_binarize(y_true, classes=range(self.n_classes))
        
        # Macro AUC
        try:
            auc_macro = roc_auc_score(y_true_bin, y_proba, average='macro', multi_class='ovr')
        except:
            auc_macro = 0.0
        
        # Per-class AUC
        auc_per_class = {}
        for i in range(self.n_classes):
            try:
                if len(np.unique(y_true_bin[:, i])) > 1:  # Check if class exists in data
                    auc_per_class[self.class_names[i]] = roc_auc_score(y_true_bin[:, i], y_proba[:, i])
                else:
                    auc_per_class[self.class_names[i]] = 0.0
            except:
                auc_per_class[self.class_names[i]] = 0.0
        
        # Clinical metrics (sensitivity and specificity for each class)
        clinical_metrics = {}
        cm = confusion_matrix(y_true, y_pred)
        
        for i, class_name in enumerate(self.class_names):
            tp = cm[i, i]
            fp = np.sum(cm[:, i]) - tp
            fn = np.sum(cm[i, :]) - tp
            tn = np.sum(cm) - tp - fp - fn
            
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            
            clinical_metrics[class_name] = {
                'sensitivity': sensitivity,
                'specificity': specificity,
                'precision': precision
            }
        
        return {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'kappa': kappa,
            'auc_macro': auc_macro,
            'auc_per_class': auc_per_class,
            'classification_report': report,
            'clinical_metrics': clinical_metrics,
            'confusion_matrix': cm
        }
    
    def _generate_confusion_matrix(self, y_true, y_pred, save_dir):
        """Generate confusion matrix visualization"""
        cm = confusion_matrix(y_true, y_pred)
        
        # Matplotlib version
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Normalized confusion matrix
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', 
                   xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title('Normalized Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'confusion_matrix_normalized.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_classification_report(self, y_true, y_pred, save_dir):
        """Generate and save classification report"""
        report = classification_report(y_true, y_pred, target_names=self.class_names)
        
        with open(os.path.join(save_dir, 'classification_report.txt'), 'w') as f:
            f.write("DIABETIC RETINOPATHY MODEL EVALUATION REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(report)
    
    def _generate_roc_curves(self, y_true, y_proba, save_dir):
        """Generate ROC curves for each class"""
        y_true_bin = label_binarize(y_true, classes=range(self.n_classes))
        
        plt.figure(figsize=(12, 8))
        
        for i in range(self.n_classes):
            if len(np.unique(y_true_bin[:, i])) > 1:
                fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
                auc = roc_auc_score(y_true_bin[:, i], y_proba[:, i])
                plt.plot(fpr, tpr, label=f'{self.class_names[i]} (AUC = {auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves for Each Class')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'roc_curves.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_precision_recall_curves(self, y_true, y_proba, save_dir):
        """Generate Precision-Recall curves for each class"""
        y_true_bin = label_binarize(y_true, classes=range(self.n_classes))
        
        plt.figure(figsize=(12, 8))
        
        for i in range(self.n_classes):
            if len(np.unique(y_true_bin[:, i])) > 1:
                precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_proba[:, i])
                plt.plot(recall, precision, label=f'{self.class_names[i]}')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves for Each Class')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'precision_recall_curves.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_class_distribution_analysis(self, y_true, y_pred, save_dir):
        """Analyze class distribution in predictions vs ground truth"""
        true_counts = np.bincount(y_true, minlength=self.n_classes)
        pred_counts = np.bincount(y_pred, minlength=self.n_classes)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # True distribution
        ax1.bar(range(self.n_classes), true_counts, color='skyblue', alpha=0.7)
        ax1.set_title('True Label Distribution')
        ax1.set_xlabel('Class')
        ax1.set_ylabel('Count')
        ax1.set_xticks(range(self.n_classes))
        ax1.set_xticklabels(self.class_names, rotation=45)
        
        # Predicted distribution
        ax2.bar(range(self.n_classes), pred_counts, color='lightcoral', alpha=0.7)
        ax2.set_title('Predicted Label Distribution')
        ax2.set_xlabel('Class')
        ax2.set_ylabel('Count')
        ax2.set_xticks(range(self.n_classes))
        ax2.set_xticklabels(self.class_names, rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'class_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_interactive_dashboard(self, y_true, y_pred, y_proba, metrics, save_dir):
        """Generate interactive dashboard using Plotly"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Confusion Matrix', 'ROC Curves', 'Class Distribution', 'Metrics Summary'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        fig.add_trace(
            go.Heatmap(z=cm, x=self.class_names, y=self.class_names, colorscale='Blues'),
            row=1, col=1
        )
        
        # ROC Curves
        y_true_bin = label_binarize(y_true, classes=range(self.n_classes))
        for i in range(self.n_classes):
            if len(np.unique(y_true_bin[:, i])) > 1:
                fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
                fig.add_trace(
                    go.Scatter(x=fpr, y=tpr, name=self.class_names[i], mode='lines'),
                    row=1, col=2
                )
        
        # Class Distribution
        true_counts = np.bincount(y_true, minlength=self.n_classes)
        pred_counts = np.bincount(y_pred, minlength=self.n_classes)
        
        fig.add_trace(
            go.Bar(x=self.class_names, y=true_counts, name='True', opacity=0.7),
            row=2, col=1
        )
        fig.add_trace(
            go.Bar(x=self.class_names, y=pred_counts, name='Predicted', opacity=0.7),
            row=2, col=1
        )
        
        # Metrics table
        metrics_text = f"""
        Accuracy: {metrics['accuracy']:.4f}
        F1 (Macro): {metrics['f1_macro']:.4f}
        F1 (Weighted): {metrics['f1_weighted']:.4f}
        Cohen's Kappa: {metrics['kappa']:.4f}
        AUC (Macro): {metrics['auc_macro']:.4f}
        """
        
        fig.add_trace(
            go.Scatter(x=[0], y=[0], mode='text', text=[metrics_text], 
                      textfont_size=12, showlegend=False),
            row=2, col=2
        )
        
        fig.update_layout(height=800, title_text="Diabetic Retinopathy Model Evaluation Dashboard")
        fig.write_html(os.path.join(save_dir, 'evaluation_dashboard.html'))
    
    def _save_detailed_results(self, metrics, y_true, y_pred, y_proba, save_dir):
        """Save all detailed results to JSON"""
        # Convert numpy arrays to lists for JSON serialization
        results = {
            'timestamp': datetime.now().isoformat(),
            'model_performance': {
                'accuracy': float(metrics['accuracy']),
                'f1_macro': float(metrics['f1_macro']),
                'f1_weighted': float(metrics['f1_weighted']),
                'kappa': float(metrics['kappa']),
                'auc_macro': float(metrics['auc_macro'])
            },
            'per_class_metrics': {},
            'auc_scores': metrics['auc_per_class'],
            'clinical_metrics': metrics['clinical_metrics'],
            'confusion_matrix': metrics['confusion_matrix'].tolist(),
            'class_names': self.class_names
        }
        
        # Add per-class metrics from classification report
        for class_name in self.class_names:
            if class_name in metrics['classification_report']:
                results['per_class_metrics'][class_name] = {
                    'precision': float(metrics['classification_report'][class_name]['precision']),
                    'recall': float(metrics['classification_report'][class_name]['recall']),
                    'f1_score': float(metrics['classification_report'][class_name]['f1-score'])
                }
        
        with open(os.path.join(save_dir, 'detailed_results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save raw predictions for further analysis
        np.save(os.path.join(save_dir, 'y_true.npy'), y_true)
        np.save(os.path.join(save_dir, 'y_pred.npy'), y_pred)
        np.save(os.path.join(save_dir, 'y_proba.npy'), y_proba)


def evaluate_saved_model(model_path, test_dataloader, device='cpu', class_names=None):
    """Convenience function to evaluate a saved model"""
    import torch
    import timm
    
    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    model_name = checkpoint.get('model_name', 'efficientnet_b0')
    n_classes = checkpoint.get('n_classes', 5)
    
    model = timm.create_model(model_name, pretrained=False, num_classes=n_classes)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    # Evaluate
    evaluator = ModelEvaluator(class_names)
    return evaluator.evaluate_model(model, test_dataloader, device)