"""
Model benchmarking and comparison script for diabetic retinopathy detection
"""
import argparse
import time
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import json
from pathlib import Path
import logging
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

from utils import RetinopathyDataset, get_transforms
from models.advanced_architectures import MODEL_RECIPES, get_model_recipe, create_advanced_model
from evaluation import ModelEvaluator


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelBenchmark:
    """Comprehensive model benchmarking suite"""
    
    def __init__(self, test_df, img_dir, output_dir='benchmark_results'):
        self.test_df = test_df
        self.img_dir = img_dir
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Setup test data loader
        _, test_transforms = get_transforms(224, advanced=True)
        test_dataset = RetinopathyDataset(
            test_df, img_dir, transforms=test_transforms, use_advanced_preprocessing=True
        )
        
        self.test_loader = DataLoader(
            test_dataset, batch_size=32, shuffle=False, 
            num_workers=4, pin_memory=True
        )
        
        self.evaluator = ModelEvaluator()
        self.results = {}
    
    def benchmark_model(self, model_name: str, model: nn.Module, 
                       model_path: str = None) -> Dict:
        """Benchmark a single model"""
        logger.info(f"Benchmarking {model_name}...")
        
        model.to(self.device)
        model.eval()
        
        # Performance metrics
        results = {
            'model_name': model_name,
            'model_path': model_path,
            'parameters': self.count_parameters(model),
            'model_size_mb': self.get_model_size(model),
        }
        
        # Inference speed test
        inference_times = self.measure_inference_speed(model)
        results.update(inference_times)
        
        # Memory usage
        memory_usage = self.measure_memory_usage(model)
        results.update(memory_usage)
        
        # Accuracy metrics
        accuracy_metrics = self.evaluate_accuracy(model, model_name)
        results.update(accuracy_metrics)
        
        logger.info(f"Completed benchmarking {model_name}")
        return results
    
    def count_parameters(self, model: nn.Module) -> Dict[str, int]:
        """Count model parameters"""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'non_trainable_parameters': total_params - trainable_params
        }
    
    def get_model_size(self, model: nn.Module) -> float:
        """Get model size in MB"""
        param_size = 0
        buffer_size = 0
        
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        size_all_mb = (param_size + buffer_size) / 1024**2
        return round(size_all_mb, 2)
    
    def measure_inference_speed(self, model: nn.Module, 
                              num_warmup: int = 10, num_runs: int = 100) -> Dict[str, float]:
        """Measure inference speed"""
        dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(num_warmup):
                _ = model(dummy_input)
        
        # Synchronize GPU if available
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # Measure time
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(num_runs):
                _ = model(dummy_input)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        end_time = time.time()
        
        total_time = end_time - start_time
        avg_time = total_time / num_runs
        
        # Batch inference speed
        batch_input = torch.randn(32, 3, 224, 224).to(self.device)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        batch_start = time.time()
        
        with torch.no_grad():
            _ = model(batch_input)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        batch_end = time.time()
        batch_time = batch_end - batch_start
        
        return {
            'inference_time_single_ms': round(avg_time * 1000, 3),
            'inference_time_batch32_ms': round(batch_time * 1000, 3),
            'throughput_images_per_second': round(1 / avg_time, 2),
            'throughput_batch32_images_per_second': round(32 / batch_time, 2)
        }
    
    def measure_memory_usage(self, model: nn.Module) -> Dict[str, float]:
        """Measure GPU memory usage"""
        if not torch.cuda.is_available():
            return {'gpu_memory_allocated_mb': 0, 'gpu_memory_reserved_mb': 0}
        
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        dummy_input = torch.randn(32, 3, 224, 224).to(self.device)
        
        with torch.no_grad():
            _ = model(dummy_input)
        
        allocated = torch.cuda.max_memory_allocated() / 1024**2
        reserved = torch.cuda.max_memory_reserved() / 1024**2
        
        return {
            'gpu_memory_allocated_mb': round(allocated, 2),
            'gpu_memory_reserved_mb': round(reserved, 2)
        }
    
    def evaluate_accuracy(self, model: nn.Module, model_name: str) -> Dict:
        """Evaluate model accuracy"""
        eval_dir = self.output_dir / f'evaluation_{model_name.replace(" ", "_")}'
        
        try:
            metrics = self.evaluator.evaluate_model(
                model, self.test_loader, self.device, str(eval_dir)
            )
            
            return {
                'accuracy': round(metrics['accuracy'], 4),
                'f1_macro': round(metrics['f1_macro'], 4),
                'f1_weighted': round(metrics['f1_weighted'], 4),
                'kappa': round(metrics['kappa'], 4),
                'auc_macro': round(metrics['auc_macro'], 4)
            }
        except Exception as e:
            logger.error(f"Error evaluating {model_name}: {e}")
            return {
                'accuracy': 0.0,
                'f1_macro': 0.0,
                'f1_weighted': 0.0,
                'kappa': 0.0,
                'auc_macro': 0.0
            }
    
    def benchmark_all_recipes(self) -> Dict[str, Dict]:
        """Benchmark all pre-configured model recipes"""
        results = {}
        
        for recipe_name in MODEL_RECIPES:
            try:
                logger.info(f"Creating model from recipe: {recipe_name}")
                model = get_model_recipe(recipe_name, num_classes=5)
                
                result = self.benchmark_model(f"Recipe: {recipe_name}", model)
                results[recipe_name] = result
                
                # Save individual result
                result_file = self.output_dir / f'result_{recipe_name}.json'
                with open(result_file, 'w') as f:
                    json.dump(result, f, indent=2)
                
            except Exception as e:
                logger.error(f"Failed to benchmark recipe {recipe_name}: {e}")
                continue
        
        self.results.update(results)
        return results
    
    def benchmark_custom_models(self, model_configs: List[Dict]) -> Dict[str, Dict]:
        """Benchmark custom model configurations"""
        results = {}
        
        for config in model_configs:
            try:
                model_name = config['name']
                model_type = config['type']
                model_kwargs = config.get('kwargs', {})
                
                logger.info(f"Creating custom model: {model_name}")
                model = create_advanced_model(model_type, num_classes=5, **model_kwargs)
                
                result = self.benchmark_model(f"Custom: {model_name}", model)
                results[model_name] = result
                
                # Save individual result
                result_file = self.output_dir / f'result_{model_name.replace(" ", "_")}.json'
                with open(result_file, 'w') as f:
                    json.dump(result, f, indent=2)
                
            except Exception as e:
                logger.error(f"Failed to benchmark custom model {config['name']}: {e}")
                continue
        
        self.results.update(results)
        return results
    
    def benchmark_pretrained_models(self, model_paths: Dict[str, str]) -> Dict[str, Dict]:
        """Benchmark pre-trained model checkpoints"""
        results = {}
        
        for model_name, model_path in model_paths.items():
            try:
                logger.info(f"Loading pre-trained model: {model_name}")
                
                # Load checkpoint
                checkpoint = torch.load(model_path, map_location=self.device)
                
                # Determine model architecture
                model_config = checkpoint.get('config', {})
                
                if 'model_recipe' in model_config and model_config['model_recipe']:
                    model = get_model_recipe(model_config['model_recipe'], num_classes=5)
                elif 'model_type' in model_config:
                    model_kwargs = {k: v for k, v in model_config.items() 
                                  if k in ['backbone', 'pretrained', 'use_attention']}
                    model = create_advanced_model(
                        model_config['model_type'], 
                        num_classes=5, 
                        **model_kwargs
                    )
                else:
                    # Fallback to basic model
                    logger.warning(f"Using fallback model for {model_name}")
                    model = create_advanced_model('advanced_cnn', num_classes=5)
                
                # Load weights
                model.load_state_dict(checkpoint['model_state_dict'])
                
                result = self.benchmark_model(f"Pretrained: {model_name}", model, model_path)
                results[model_name] = result
                
                # Save individual result
                result_file = self.output_dir / f'result_pretrained_{model_name.replace(" ", "_")}.json'
                with open(result_file, 'w') as f:
                    json.dump(result, f, indent=2)
                
            except Exception as e:
                logger.error(f"Failed to benchmark pre-trained model {model_name}: {e}")
                continue
        
        self.results.update(results)
        return results
    
    def generate_comparison_report(self) -> str:
        """Generate comprehensive comparison report"""
        if not self.results:
            return "No benchmark results available."
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame.from_dict(self.results, orient='index')
        
        # Generate report
        report = ["# Model Benchmark Report\n"]
        report.append(f"**Test Dataset:** {len(self.test_df)} samples\n")
        report.append(f"**Device:** {self.device}\n\n")
        
        # Performance Summary
        report.append("## Performance Summary\n")
        
        # Sort by accuracy
        accuracy_ranking = df.sort_values('accuracy', ascending=False)
        report.append("### Top Models by Accuracy\n")
        for i, (model_name, row) in enumerate(accuracy_ranking.head(5).iterrows(), 1):
            report.append(f"{i}. **{model_name}**: {row['accuracy']:.4f} accuracy, {row['f1_macro']:.4f} F1\n")
        
        report.append("\n### Speed Comparison\n")
        speed_ranking = df.sort_values('inference_time_single_ms')
        report.append("**Fastest Models (single image):**\n")
        for i, (model_name, row) in enumerate(speed_ranking.head(3).iterrows(), 1):
            report.append(f"{i}. **{model_name}**: {row['inference_time_single_ms']:.1f}ms\n")
        
        # Model Size Comparison
        report.append("\n### Model Size Comparison\n")
        size_ranking = df.sort_values('model_size_mb')
        report.append("**Smallest Models:**\n")
        for i, (model_name, row) in enumerate(size_ranking.head(3).iterrows(), 1):
            report.append(f"{i}. **{model_name}**: {row['model_size_mb']:.1f}MB, {row['total_parameters']:,} params\n")
        
        # Detailed Comparison Table
        report.append("\n## Detailed Comparison\n")
        report.append("| Model | Accuracy | F1 | Speed (ms) | Size (MB) | Parameters |\n")
        report.append("|-------|----------|----|-----------:|----------:|-----------:|\n")
        
        for model_name, row in accuracy_ranking.iterrows():
            report.append(
                f"| {model_name} | {row['accuracy']:.4f} | {row['f1_macro']:.4f} | "
                f"{row['inference_time_single_ms']:.1f} | {row['model_size_mb']:.1f} | "
                f"{row['total_parameters']:,} |\n"
            )
        
        # Best model recommendations
        report.append("\n## Recommendations\n")
        
        best_accuracy = accuracy_ranking.iloc[0]
        report.append(f"**Best Accuracy:** {best_accuracy.name} ({best_accuracy['accuracy']:.4f})\n")
        
        best_speed = speed_ranking.iloc[0]
        report.append(f"**Fastest:** {best_speed.name} ({best_speed['inference_time_single_ms']:.1f}ms)\n")
        
        best_size = size_ranking.iloc[0]
        report.append(f"**Smallest:** {best_size.name} ({best_size['model_size_mb']:.1f}MB)\n")
        
        # Balance recommendation
        df['efficiency_score'] = df['accuracy'] / (df['inference_time_single_ms'] / 1000)
        best_balance = df.sort_values('efficiency_score', ascending=False).iloc[0]
        report.append(f"**Best Balance (Accuracy/Speed):** {best_balance.name}\n")
        
        report_text = ''.join(report)
        
        # Save report
        report_file = self.output_dir / 'benchmark_report.md'
        with open(report_file, 'w') as f:
            f.write(report_text)
        
        logger.info(f"Benchmark report saved to: {report_file}")
        return report_text
    
    def create_visualizations(self):
        """Create benchmark visualization plots"""
        if not self.results:
            return
        
        df = pd.DataFrame.from_dict(self.results, orient='index')
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Benchmark Comparison', fontsize=16, fontweight='bold')
        
        # Accuracy vs Speed scatter plot
        ax1 = axes[0, 0]
        scatter = ax1.scatter(df['inference_time_single_ms'], df['accuracy'], 
                            s=df['model_size_mb']*5, alpha=0.6, c=df['f1_macro'], 
                            cmap='viridis')
        ax1.set_xlabel('Inference Time (ms)')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Accuracy vs Speed\n(Size = bubble size, Color = F1 score)')
        plt.colorbar(scatter, ax=ax1)
        
        # Model size comparison
        ax2 = axes[0, 1]
        model_names = [name.split(': ')[-1] for name in df.index]
        bars = ax2.bar(range(len(model_names)), df['model_size_mb'])
        ax2.set_xlabel('Models')
        ax2.set_ylabel('Model Size (MB)')
        ax2.set_title('Model Size Comparison')
        ax2.set_xticks(range(len(model_names)))
        ax2.set_xticklabels(model_names, rotation=45, ha='right')
        
        # Performance metrics comparison
        ax3 = axes[1, 0]
        metrics = ['accuracy', 'f1_macro', 'kappa']
        x = np.arange(len(model_names))
        width = 0.25
        
        for i, metric in enumerate(metrics):
            ax3.bar(x + i*width, df[metric], width, label=metric.replace('_', ' ').title())
        
        ax3.set_xlabel('Models')
        ax3.set_ylabel('Score')
        ax3.set_title('Performance Metrics Comparison')
        ax3.set_xticks(x + width)
        ax3.set_xticklabels(model_names, rotation=45, ha='right')
        ax3.legend()
        
        # Speed comparison
        ax4 = axes[1, 1]
        ax4.barh(model_names, df['inference_time_single_ms'])
        ax4.set_xlabel('Inference Time (ms)')
        ax4.set_title('Inference Speed Comparison')
        
        plt.tight_layout()
        
        # Save plot
        plot_file = self.output_dir / 'benchmark_comparison.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        logger.info(f"Benchmark plots saved to: {plot_file}")
        
        plt.show()
    
    def save_results(self):
        """Save all benchmark results"""
        # Save complete results
        results_file = self.output_dir / 'benchmark_results.json'
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Save as CSV for easy analysis
        if self.results:
            df = pd.DataFrame.from_dict(self.results, orient='index')
            csv_file = self.output_dir / 'benchmark_results.csv'
            df.to_csv(csv_file)
            
            logger.info(f"Results saved to: {results_file} and {csv_file}")


def main():
    parser = argparse.ArgumentParser(description='Benchmark diabetic retinopathy models')
    
    parser.add_argument('--test-csv', type=str, required=True, help='Test dataset CSV')
    parser.add_argument('--img-dir', type=str, required=True, help='Images directory')
    parser.add_argument('--output-dir', type=str, default='benchmark_results', help='Output directory')
    
    # What to benchmark
    parser.add_argument('--benchmark-recipes', action='store_true', 
                       help='Benchmark all model recipes')
    parser.add_argument('--benchmark-custom', type=str, 
                       help='JSON file with custom model configurations')
    parser.add_argument('--benchmark-pretrained', type=str,
                       help='JSON file with pretrained model paths')
    
    # Options
    parser.add_argument('--generate-report', action='store_true',
                       help='Generate comparison report')
    parser.add_argument('--create-plots', action='store_true',
                       help='Create visualization plots')
    
    args = parser.parse_args()
    
    # Load test data
    test_df = pd.read_csv(args.test_csv)
    if 'diagnosis' in test_df.columns:
        test_df = test_df.rename(columns={'diagnosis': 'label'})
    
    logger.info(f"Loaded test dataset with {len(test_df)} samples")
    
    # Create benchmark suite
    benchmark = ModelBenchmark(test_df, args.img_dir, args.output_dir)
    
    # Run benchmarks
    if args.benchmark_recipes:
        logger.info("Benchmarking model recipes...")
        benchmark.benchmark_all_recipes()
    
    if args.benchmark_custom:
        logger.info("Benchmarking custom models...")
        with open(args.benchmark_custom, 'r') as f:
            custom_configs = json.load(f)
        benchmark.benchmark_custom_models(custom_configs)
    
    if args.benchmark_pretrained:
        logger.info("Benchmarking pretrained models...")
        with open(args.benchmark_pretrained, 'r') as f:
            pretrained_paths = json.load(f)
        benchmark.benchmark_pretrained_models(pretrained_paths)
    
    # Generate outputs
    benchmark.save_results()
    
    if args.generate_report:
        logger.info("Generating comparison report...")
        report = benchmark.generate_comparison_report()
        print("\n" + report)
    
    if args.create_plots:
        logger.info("Creating visualization plots...")
        benchmark.create_visualizations()
    
    logger.info("Benchmarking completed!")


if __name__ == '__main__':
    main()