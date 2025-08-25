"""
Advanced dataset analysis and visualization tools for diabetic retinopathy detection
"""
import os
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class DatasetAnalyzer:
    """Comprehensive dataset analysis and visualization tool"""
    
    def __init__(self, data_dir: str = 'ml/data'):
        """
        Initialize dataset analyzer
        
        Args:
            data_dir: Directory containing dataset files
        """
        self.data_dir = Path(data_dir)
        self.class_names = ['No DR', 'Mild DR', 'Moderate DR', 'Severe DR', 'Proliferative DR']
        self.severity_colors = {
            0: '#10b981',  # Green - No DR
            1: '#f59e0b',  # Yellow - Mild
            2: '#f97316',  # Orange - Moderate  
            3: '#ef4444',  # Red - Severe
            4: '#dc2626'   # Dark Red - Proliferative
        }
        
    def generate_comprehensive_report(self, save_dir: str = 'ml/dataset_analysis'):
        """Generate comprehensive dataset analysis report"""
        print("🔍 Generating comprehensive dataset analysis report...")
        
        # Create output directory
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Load data
        train_df = pd.read_csv(self.data_dir / 'train.csv')
        
        # Generate all analyses
        basic_stats = self._analyze_basic_statistics(train_df)
        class_analysis = self._analyze_class_distribution(train_df, save_path)
        image_analysis = self._analyze_image_properties(train_df, save_path)
        quality_analysis = self._analyze_image_quality(train_df, save_path)
        correlation_analysis = self._analyze_correlations(train_df, save_path)
        
        # Generate visualizations
        self._create_comprehensive_dashboard(
            basic_stats, class_analysis, image_analysis, save_path
        )
        
        # Generate HTML report
        self._generate_html_report(
            basic_stats, class_analysis, image_analysis, quality_analysis, save_path
        )
        
        # Save detailed results
        self._save_analysis_results(
            basic_stats, class_analysis, image_analysis, quality_analysis, save_path
        )
        
        print(f"✅ Analysis complete! Results saved to {save_path}")
        return save_path
    
    def _analyze_basic_statistics(self, df: pd.DataFrame) -> dict:
        """Analyze basic dataset statistics"""
        print("📊 Analyzing basic statistics...")
        
        stats = {
            'total_samples': len(df),
            'unique_images': df['id_code'].nunique(),
            'duplicate_images': len(df) - df['id_code'].nunique(),
            'classes': sorted(df['diagnosis'].unique()),
            'class_counts': df['diagnosis'].value_counts().sort_index().to_dict(),
            'class_percentages': (df['diagnosis'].value_counts(normalize=True).sort_index() * 100).to_dict(),
            'imbalance_ratio': df['diagnosis'].value_counts().max() / df['diagnosis'].value_counts().min(),
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        return stats
    
    def _analyze_class_distribution(self, df: pd.DataFrame, save_path: Path) -> dict:
        """Detailed class distribution analysis"""
        print("🏷️ Analyzing class distribution...")
        
        # Basic distribution
        class_dist = df['diagnosis'].value_counts().sort_index()
        class_percentages = (df['diagnosis'].value_counts(normalize=True).sort_index() * 100)
        
        # Statistical measures
        entropy = -np.sum(class_percentages / 100 * np.log2(class_percentages / 100 + 1e-10))
        gini_impurity = 1 - np.sum((class_percentages / 100) ** 2)
        
        # Create visualizations
        self._plot_class_distribution(class_dist, class_percentages, save_path)
        
        analysis = {
            'distribution': class_dist.to_dict(),
            'percentages': class_percentages.to_dict(),
            'entropy': entropy,
            'gini_impurity': gini_impurity,
            'most_common_class': class_dist.idxmax(),
            'least_common_class': class_dist.idxmin(),
            'imbalance_recommendations': self._get_imbalance_recommendations(class_dist)
        }
        
        return analysis
    
    def _analyze_image_properties(self, df: pd.DataFrame, save_path: Path, sample_size: int = 500) -> dict:
        """Analyze image properties (dimensions, file sizes, etc.)"""
        print(f"🖼️ Analyzing image properties (sample size: {sample_size})...")
        
        train_img_dir = self.data_dir / 'train_images'
        sample_df = df.sample(n=min(sample_size, len(df)), random_state=42)
        
        properties = []
        
        def analyze_image(img_name):
            try:
                # Try both .png and .jpg extensions
                img_path = train_img_dir / f"{img_name}.png"
                if not img_path.exists():
                    img_path = train_img_dir / f"{img_name}.jpg"
                
                if img_path.exists():
                    # File size
                    file_size = img_path.stat().st_size
                    
                    # Image dimensions and properties
                    with Image.open(img_path) as img:
                        width, height = img.size
                        mode = img.mode
                        
                        # Additional properties
                        aspect_ratio = width / height
                        megapixels = (width * height) / 1_000_000
                        
                        return {
                            'img_name': img_name,
                            'file_size_mb': file_size / (1024 * 1024),
                            'width': width,
                            'height': height,
                            'aspect_ratio': aspect_ratio,
                            'megapixels': megapixels,
                            'color_mode': mode,
                            'resolution_category': self._categorize_resolution(width, height)
                        }
            except Exception as e:
                print(f"Error analyzing {img_name}: {e}")
                return None
        
        # Parallel processing
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {executor.submit(analyze_image, name): name 
                      for name in sample_df['id_code']}
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Analyzing images"):
                result = future.result()
                if result:
                    properties.append(result)
        
        if not properties:
            return {'error': 'No images could be analyzed'}
        
        # Convert to DataFrame for analysis
        props_df = pd.DataFrame(properties)
        
        # Statistical analysis
        analysis = {
            'sample_size': len(properties),
            'dimensions': {
                'width_stats': props_df['width'].describe().to_dict(),
                'height_stats': props_df['height'].describe().to_dict(),
                'aspect_ratio_stats': props_df['aspect_ratio'].describe().to_dict(),
            },
            'file_sizes': {
                'size_mb_stats': props_df['file_size_mb'].describe().to_dict(),
                'total_size_gb': props_df['file_size_mb'].sum() / 1024 * (len(df) / len(properties))
            },
            'resolution_categories': props_df['resolution_category'].value_counts().to_dict(),
            'color_modes': props_df['color_mode'].value_counts().to_dict(),
            'common_resolutions': self._find_common_resolutions(props_df)
        }
        
        # Create visualizations
        self._plot_image_properties(props_df, save_path)
        
        return analysis
    
    def _analyze_image_quality(self, df: pd.DataFrame, save_path: Path, sample_size: int = 200) -> dict:
        """Analyze image quality metrics"""
        print(f"✨ Analyzing image quality (sample size: {sample_size})...")
        
        train_img_dir = self.data_dir / 'train_images'
        sample_df = df.sample(n=min(sample_size, len(df)), random_state=42)
        
        quality_metrics = []
        
        def analyze_quality(row):
            try:
                img_name = row['id_code']
                diagnosis = row['diagnosis']
                
                # Load image
                img_path = train_img_dir / f"{img_name}.png"
                if not img_path.exists():
                    img_path = train_img_dir / f"{img_name}.jpg"
                
                if img_path.exists():
                    img = cv2.imread(str(img_path))
                    if img is not None:
                        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        
                        # Quality metrics
                        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()  # Higher = less blurry
                        brightness = np.mean(gray)
                        contrast = np.std(gray)
                        
                        # Edge density
                        edges = cv2.Canny(gray, 50, 150)
                        edge_density = np.sum(edges > 0) / edges.size
                        
                        # Signal-to-noise ratio estimation
                        signal = np.mean(gray)
                        noise = np.std(gray)
                        snr = signal / noise if noise > 0 else 0
                        
                        return {
                            'img_name': img_name,
                            'diagnosis': diagnosis,
                            'blur_score': blur_score,
                            'brightness': brightness,
                            'contrast': contrast,
                            'edge_density': edge_density,
                            'snr': snr,
                            'quality_category': self._categorize_quality(blur_score, brightness, contrast)
                        }
            except Exception as e:
                print(f"Error analyzing quality for {row['id_code']}: {e}")
                return None
        
        # Process in parallel
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(analyze_quality, row) for _, row in sample_df.iterrows()]
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Analyzing quality"):
                result = future.result()
                if result:
                    quality_metrics.append(result)
        
        if not quality_metrics:
            return {'error': 'No quality metrics could be computed'}
        
        # Convert to DataFrame
        quality_df = pd.DataFrame(quality_metrics)
        
        # Analysis by severity class
        quality_by_class = {}
        for diagnosis in sorted(quality_df['diagnosis'].unique()):
            class_data = quality_df[quality_df['diagnosis'] == diagnosis]
            quality_by_class[diagnosis] = {
                'count': len(class_data),
                'blur_score_mean': class_data['blur_score'].mean(),
                'brightness_mean': class_data['brightness'].mean(),
                'contrast_mean': class_data['contrast'].mean(),
                'edge_density_mean': class_data['edge_density'].mean()
            }
        
        analysis = {
            'sample_size': len(quality_metrics),
            'overall_metrics': {
                'blur_score_stats': quality_df['blur_score'].describe().to_dict(),
                'brightness_stats': quality_df['brightness'].describe().to_dict(),
                'contrast_stats': quality_df['contrast'].describe().to_dict(),
                'edge_density_stats': quality_df['edge_density'].describe().to_dict()
            },
            'quality_by_severity': quality_by_class,
            'quality_categories': quality_df['quality_category'].value_counts().to_dict(),
            'recommendations': self._get_quality_recommendations(quality_df)
        }
        
        # Create visualizations
        self._plot_quality_analysis(quality_df, save_path)
        
        return analysis
    
    def _analyze_correlations(self, df: pd.DataFrame, save_path: Path) -> dict:
        """Analyze correlations and patterns in the data"""
        print("🔗 Analyzing correlations and patterns...")
        
        # This would be expanded with additional metadata if available
        analysis = {
            'class_correlations': 'Limited to class distribution analysis with current data',
            'temporal_patterns': 'Not available - no timestamp data',
            'spatial_patterns': 'Would require image location metadata'
        }
        
        return analysis
    
    def _categorize_resolution(self, width: int, height: int) -> str:
        """Categorize image resolution"""
        total_pixels = width * height
        
        if total_pixels < 500_000:
            return 'Low (<0.5MP)'
        elif total_pixels < 2_000_000:
            return 'Medium (0.5-2MP)'
        elif total_pixels < 8_000_000:
            return 'High (2-8MP)'
        else:
            return 'Very High (>8MP)'
    
    def _categorize_quality(self, blur_score: float, brightness: float, contrast: float) -> str:
        """Categorize image quality"""
        # These thresholds would ideally be determined from medical image standards
        if blur_score > 500 and brightness > 50 and contrast > 30:
            return 'High Quality'
        elif blur_score > 100 and brightness > 30 and contrast > 20:
            return 'Medium Quality'
        else:
            return 'Low Quality'
    
    def _find_common_resolutions(self, props_df: pd.DataFrame) -> dict:
        """Find most common image resolutions"""
        resolution_counts = props_df.groupby(['width', 'height']).size().sort_values(ascending=False)
        return {f"{w}x{h}": count for (w, h), count in resolution_counts.head(10).items()}
    
    def _get_imbalance_recommendations(self, class_dist: pd.Series) -> list:
        """Get recommendations for handling class imbalance"""
        imbalance_ratio = class_dist.max() / class_dist.min()
        
        recommendations = []
        
        if imbalance_ratio > 10:
            recommendations.append("Severe class imbalance detected. Consider using class weights or resampling techniques.")
        elif imbalance_ratio > 5:
            recommendations.append("Moderate class imbalance detected. Class weights recommended.")
        else:
            recommendations.append("Class distribution is relatively balanced.")
        
        # Specific techniques
        recommendations.extend([
            "Use stratified sampling for train/validation splits",
            "Consider focal loss for handling imbalanced classes",
            "Apply appropriate evaluation metrics (F1-score, AUC-ROC) instead of just accuracy",
            "Use ensemble methods to improve minority class prediction"
        ])
        
        return recommendations
    
    def _get_quality_recommendations(self, quality_df: pd.DataFrame) -> list:
        """Get recommendations based on image quality analysis"""
        recommendations = []
        
        avg_blur = quality_df['blur_score'].mean()
        avg_brightness = quality_df['brightness'].mean()
        avg_contrast = quality_df['contrast'].mean()
        
        if avg_blur < 100:
            recommendations.append("Many images appear blurry. Consider blur detection and filtering.")
        
        if avg_brightness < 30:
            recommendations.append("Images are quite dark. Consider brightness normalization or histogram equalization.")
        elif avg_brightness > 200:
            recommendations.append("Images are quite bright. Consider exposure correction.")
        
        if avg_contrast < 20:
            recommendations.append("Low contrast detected. Consider CLAHE (Contrast Limited Adaptive Histogram Equalization).")
        
        recommendations.extend([
            "Apply consistent preprocessing across all images",
            "Consider image quality as a feature for model training",
            "Implement quality-based data filtering if necessary"
        ])
        
        return recommendations
    
    def _plot_class_distribution(self, class_dist: pd.Series, class_percentages: pd.Series, save_path: Path):
        """Create class distribution visualizations"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Bar plot
        ax1 = axes[0, 0]
        bars = ax1.bar(range(len(class_dist)), class_dist.values, 
                      color=[self.severity_colors[i] for i in class_dist.index])
        ax1.set_title('Class Distribution (Counts)', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Severity Level')
        ax1.set_ylabel('Number of Samples')
        ax1.set_xticks(range(len(self.class_names)))
        ax1.set_xticklabels(self.class_names, rotation=45, ha='right')
        
        # Add value labels
        for bar, count in zip(bars, class_dist.values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                    str(count), ha='center', va='bottom', fontweight='bold')
        
        # Pie chart
        ax2 = axes[0, 1]
        colors = [self.severity_colors[i] for i in class_dist.index]
        wedges, texts, autotexts = ax2.pie(class_dist.values, labels=self.class_names, 
                                          autopct='%1.1f%%', colors=colors, startangle=90)
        ax2.set_title('Class Distribution (Percentage)', fontsize=14, fontweight='bold')
        
        # Log scale bar plot
        ax3 = axes[1, 0]
        bars_log = ax3.bar(range(len(class_dist)), class_dist.values, 
                          color=[self.severity_colors[i] for i in class_dist.index])
        ax3.set_yscale('log')
        ax3.set_title('Class Distribution (Log Scale)', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Severity Level')
        ax3.set_ylabel('Number of Samples (Log Scale)')
        ax3.set_xticks(range(len(self.class_names)))
        ax3.set_xticklabels(self.class_names, rotation=45, ha='right')
        
        # Imbalance visualization
        ax4 = axes[1, 1]
        imbalance_ratios = [class_dist.max() / count for count in class_dist.values]
        bars_imb = ax4.bar(range(len(class_dist)), imbalance_ratios, 
                          color=[self.severity_colors[i] for i in class_dist.index])
        ax4.set_title('Class Imbalance Ratios', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Severity Level')
        ax4.set_ylabel('Imbalance Ratio (vs Most Common)')
        ax4.set_xticks(range(len(self.class_names)))
        ax4.set_xticklabels(self.class_names, rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(save_path / 'class_distribution_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_image_properties(self, props_df: pd.DataFrame, save_path: Path):
        """Create image properties visualizations"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Width distribution
        axes[0, 0].hist(props_df['width'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('Image Width Distribution')
        axes[0, 0].set_xlabel('Width (pixels)')
        axes[0, 0].set_ylabel('Frequency')
        
        # Height distribution
        axes[0, 1].hist(props_df['height'], bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
        axes[0, 1].set_title('Image Height Distribution')
        axes[0, 1].set_xlabel('Height (pixels)')
        axes[0, 1].set_ylabel('Frequency')
        
        # Aspect ratio distribution
        axes[0, 2].hist(props_df['aspect_ratio'], bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[0, 2].set_title('Aspect Ratio Distribution')
        axes[0, 2].set_xlabel('Aspect Ratio (W/H)')
        axes[0, 2].set_ylabel('Frequency')
        
        # File size distribution
        axes[1, 0].hist(props_df['file_size_mb'], bins=30, alpha=0.7, color='gold', edgecolor='black')
        axes[1, 0].set_title('File Size Distribution')
        axes[1, 0].set_xlabel('File Size (MB)')
        axes[1, 0].set_ylabel('Frequency')
        
        # Width vs Height scatter
        axes[1, 1].scatter(props_df['width'], props_df['height'], alpha=0.6, color='purple')
        axes[1, 1].set_title('Width vs Height')
        axes[1, 1].set_xlabel('Width (pixels)')
        axes[1, 1].set_ylabel('Height (pixels)')
        
        # Resolution categories
        resolution_counts = props_df['resolution_category'].value_counts()
        axes[1, 2].bar(range(len(resolution_counts)), resolution_counts.values, 
                      color='orange', alpha=0.7)
        axes[1, 2].set_title('Resolution Categories')
        axes[1, 2].set_xlabel('Category')
        axes[1, 2].set_ylabel('Count')
        axes[1, 2].set_xticks(range(len(resolution_counts)))
        axes[1, 2].set_xticklabels(resolution_counts.index, rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(save_path / 'image_properties_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_quality_analysis(self, quality_df: pd.DataFrame, save_path: Path):
        """Create image quality visualizations"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Quality metrics by diagnosis
        for i, metric in enumerate(['blur_score', 'brightness', 'contrast']):
            ax = axes[0, i]
            
            # Box plot by diagnosis
            diagnosis_groups = [quality_df[quality_df['diagnosis'] == d][metric].values 
                               for d in sorted(quality_df['diagnosis'].unique())]
            
            box_plot = ax.boxplot(diagnosis_groups, labels=[self.class_names[d] for d in sorted(quality_df['diagnosis'].unique())],
                                 patch_artist=True)
            
            # Color boxes
            for patch, d in zip(box_plot['boxes'], sorted(quality_df['diagnosis'].unique())):
                patch.set_facecolor(self.severity_colors[d])
                patch.set_alpha(0.7)
            
            ax.set_title(f'{metric.replace("_", " ").title()} by Diagnosis')
            ax.set_xlabel('Diagnosis')
            ax.set_ylabel(metric.replace("_", " ").title())
            ax.tick_params(axis='x', rotation=45)
        
        # Quality distributions
        for i, metric in enumerate(['edge_density', 'snr']):
            if i < 2:  # Only plot if we have space
                ax = axes[1, i]
                ax.hist(quality_df[metric], bins=30, alpha=0.7, color='teal', edgecolor='black')
                ax.set_title(f'{metric.replace("_", " ").title()} Distribution')
                ax.set_xlabel(metric.replace("_", " ").title())
                ax.set_ylabel('Frequency')
        
        # Quality categories
        ax = axes[1, 2]
        quality_counts = quality_df['quality_category'].value_counts()
        bars = ax.bar(range(len(quality_counts)), quality_counts.values, 
                     color=['green', 'orange', 'red'][:len(quality_counts)], alpha=0.7)
        ax.set_title('Image Quality Categories')
        ax.set_xlabel('Quality Category')
        ax.set_ylabel('Count')
        ax.set_xticks(range(len(quality_counts)))
        ax.set_xticklabels(quality_counts.index, rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(save_path / 'image_quality_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_comprehensive_dashboard(self, basic_stats, class_analysis, image_analysis, save_path):
        """Create interactive dashboard using Plotly"""
        print("📊 Creating interactive dashboard...")
        
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Class Distribution', 'Class Imbalance Visualization',
                'Image Resolution Distribution', 'File Size Analysis',
                'Dataset Overview', 'Quality Metrics Summary'
            ),
            specs=[
                [{"type": "bar"}, {"type": "bar"}],
                [{"type": "histogram"}, {"type": "box"}],
                [{"type": "table"}, {"type": "scatter"}]
            ]
        )
        
        # Class distribution
        classes = list(class_analysis['distribution'].keys())
        counts = list(class_analysis['distribution'].values())
        colors = [self.severity_colors[i] for i in classes]
        
        fig.add_trace(
            go.Bar(x=[self.class_names[i] for i in classes], y=counts, 
                   marker_color=colors, name="Class Counts"),
            row=1, col=1
        )
        
        # Class imbalance
        max_count = max(counts)
        imbalance_ratios = [max_count / count for count in counts]
        fig.add_trace(
            go.Bar(x=[self.class_names[i] for i in classes], y=imbalance_ratios, 
                   marker_color=colors, name="Imbalance Ratio"),
            row=1, col=2
        )
        
        # Add more visualizations...
        
        fig.update_layout(
            height=1200,
            title_text="Diabetic Retinopathy Dataset Analysis Dashboard",
            showlegend=False
        )
        
        fig.write_html(save_path / 'dataset_analysis_dashboard.html')
    
    def _generate_html_report(self, basic_stats, class_analysis, image_analysis, quality_analysis, save_path):
        """Generate comprehensive HTML report"""
        print("📝 Generating HTML report...")
        
        html_template = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Diabetic Retinopathy Dataset Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f8ff; padding: 20px; border-radius: 10px; }}
                .section {{ margin: 20px 0; padding: 15px; border-left: 4px solid #4CAF50; }}
                .metric {{ background-color: #f9f9f9; padding: 10px; margin: 5px 0; border-radius: 5px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .recommendation {{ background-color: #fff3cd; border: 1px solid #ffeaa7; padding: 10px; border-radius: 5px; margin: 5px 0; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🏥 Diabetic Retinopathy Dataset Analysis Report</h1>
                <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="section">
                <h2>📊 Basic Dataset Statistics</h2>
                <div class="metric"><strong>Total Samples:</strong> {basic_stats['total_samples']:,}</div>
                <div class="metric"><strong>Unique Images:</strong> {basic_stats['unique_images']:,}</div>
                <div class="metric"><strong>Class Imbalance Ratio:</strong> {basic_stats['imbalance_ratio']:.2f}</div>
            </div>
            
            <div class="section">
                <h2>🏷️ Class Distribution Analysis</h2>
                <table>
                    <tr><th>Severity Level</th><th>Class Name</th><th>Count</th><th>Percentage</th></tr>
        """
        
        # Add class distribution table
        for i, class_name in enumerate(self.class_names):
            if i in class_analysis['distribution']:
                count = class_analysis['distribution'][i]
                percentage = class_analysis['percentages'][i]
                html_template += f"<tr><td>{i}</td><td>{class_name}</td><td>{count:,}</td><td>{percentage:.1f}%</td></tr>"
        
        html_template += """
                </table>
                
                <h3>📋 Imbalance Handling Recommendations</h3>
        """
        
        for rec in class_analysis['imbalance_recommendations']:
            html_template += f'<div class="recommendation">• {rec}</div>'
        
        # Add image analysis if available
        if 'error' not in image_analysis:
            html_template += f"""
                <div class="section">
                    <h2>🖼️ Image Properties Analysis</h2>
                    <div class="metric"><strong>Sample Size Analyzed:</strong> {image_analysis['sample_size']:,}</div>
                    <div class="metric"><strong>Average Width:</strong> {image_analysis['dimensions']['width_stats']['mean']:.0f} pixels</div>
                    <div class="metric"><strong>Average Height:</strong> {image_analysis['dimensions']['height_stats']['mean']:.0f} pixels</div>
                    <div class="metric"><strong>Estimated Total Dataset Size:</strong> {image_analysis['file_sizes']['total_size_gb']:.1f} GB</div>
                </div>
            """
        
        # Add quality analysis if available
        if 'error' not in quality_analysis:
            html_template += f"""
                <div class="section">
                    <h2>✨ Image Quality Analysis</h2>
                    <div class="metric"><strong>Quality Sample Size:</strong> {quality_analysis['sample_size']:,}</div>
                    
                    <h3>📋 Quality Recommendations</h3>
            """
            
            for rec in quality_analysis['recommendations']:
                html_template += f'<div class="recommendation">• {rec}</div>'
        
        html_template += """
            </div>
            
            <div class="section">
                <h2>📈 Visualizations</h2>
                <p>The following visualization files have been generated:</p>
                <ul>
                    <li>class_distribution_analysis.png - Detailed class distribution plots</li>
                    <li>image_properties_analysis.png - Image dimension and size analysis</li>
                    <li>image_quality_analysis.png - Quality metrics visualization</li>
                    <li>dataset_analysis_dashboard.html - Interactive dashboard</li>
                </ul>
            </div>
        </body>
        </html>
        """
        
        with open(save_path / 'dataset_analysis_report.html', 'w', encoding='utf-8') as f:
            f.write(html_template)
    
    def _save_analysis_results(self, basic_stats, class_analysis, image_analysis, quality_analysis, save_path):
        """Save all analysis results to JSON"""
        results = {
            'basic_statistics': basic_stats,
            'class_analysis': class_analysis,
            'image_analysis': image_analysis,
            'quality_analysis': quality_analysis,
            'generated_files': [
                'class_distribution_analysis.png',
                'image_properties_analysis.png', 
                'image_quality_analysis.png',
                'dataset_analysis_dashboard.html',
                'dataset_analysis_report.html',
                'analysis_results.json'
            ]
        }
        
        with open(save_path / 'analysis_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)


def main():
    """Run comprehensive dataset analysis"""
    print("🔍 Starting comprehensive dataset analysis...")
    
    analyzer = DatasetAnalyzer()
    results_path = analyzer.generate_comprehensive_report()
    
    print(f"\n✅ Dataset analysis completed!")
    print(f"📁 Results saved to: {results_path}")
    print(f"🌐 Open {results_path}/dataset_analysis_report.html to view the full report")


if __name__ == '__main__':
    main()