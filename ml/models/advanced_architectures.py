"""
Advanced model architectures for diabetic retinopathy detection
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import math
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class AttentionModule(nn.Module):
    """Spatial attention module for retinal images"""
    
    def __init__(self, in_channels: int, reduction_ratio: int = 16):
        super().__init__()
        self.in_channels = in_channels
        
        # Channel attention
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False),
            nn.Sigmoid()
        )
        
        # Spatial attention
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Channel attention
        ca = self.channel_attention(x)
        x = x * ca
        
        # Spatial attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        sa_input = torch.cat([avg_out, max_out], dim=1)
        sa = self.spatial_attention(sa_input)
        x = x * sa
        
        return x


class MultiScaleFeatureExtractor(nn.Module):
    """Multi-scale feature extraction for retinal analysis"""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        
        # Different scale branches
        self.branch1 = nn.Conv2d(in_channels, out_channels // 4, 1)
        self.branch3 = nn.Conv2d(in_channels, out_channels // 4, 3, padding=1)
        self.branch5 = nn.Conv2d(in_channels, out_channels // 4, 5, padding=2)
        self.branch_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels // 4, 1),
            nn.Upsample(scale_factor=1, mode='nearest')
        )
        
        self.conv_fusion = nn.Conv2d(out_channels, out_channels, 1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        size = x.size()[2:]
        
        branch1 = self.branch1(x)
        branch3 = self.branch3(x)
        branch5 = self.branch5(x)
        branch_pool = F.interpolate(
            self.branch_pool(x), 
            size=size, 
            mode='bilinear', 
            align_corners=False
        )
        
        out = torch.cat([branch1, branch3, branch5, branch_pool], dim=1)
        out = self.conv_fusion(out)
        out = self.bn(out)
        out = self.relu(out)
        
        return out


class DRAdvancedCNN(nn.Module):
    """Advanced CNN with attention and multi-scale features for DR detection"""
    
    def __init__(self, backbone: str = 'efficientnet_b0', num_classes: int = 5, 
                 pretrained: bool = True, use_attention: bool = True):
        super().__init__()
        
        # Backbone
        self.backbone = timm.create_model(
            backbone, 
            pretrained=pretrained, 
            features_only=True,
            out_indices=[2, 3, 4]  # Get intermediate features
        )
        
        # Get feature dimensions
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            features = self.backbone(dummy_input)
            self.feature_dims = [f.size(1) for f in features]
        
        # Multi-scale feature extractors
        self.ms_extractors = nn.ModuleList([
            MultiScaleFeatureExtractor(dim, dim) for dim in self.feature_dims
        ])
        
        # Attention modules
        self.use_attention = use_attention
        if use_attention:
            self.attention_modules = nn.ModuleList([
                AttentionModule(dim) for dim in self.feature_dims
            ])
        
        # Feature fusion
        total_features = sum(self.feature_dims)
        self.fusion_conv = nn.Conv2d(total_features, 512, 1)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        # Extract multi-scale features
        features = self.backbone(x)
        
        # Apply multi-scale extraction
        ms_features = []
        for i, (feat, ms_extractor) in enumerate(zip(features, self.ms_extractors)):
            ms_feat = ms_extractor(feat)
            
            # Apply attention if enabled
            if self.use_attention:
                ms_feat = self.attention_modules[i](ms_feat)
            
            # Upsample to common size (largest feature map)
            if i > 0:
                ms_feat = F.interpolate(
                    ms_feat, 
                    size=features[0].size()[2:], 
                    mode='bilinear', 
                    align_corners=False
                )
            
            ms_features.append(ms_feat)
        
        # Fuse features
        fused = torch.cat(ms_features, dim=1)
        fused = self.fusion_conv(fused)
        
        # Global pooling and classification
        pooled = self.global_pool(fused).flatten(1)
        output = self.classifier(pooled)
        
        return output


class EnsembleModel(nn.Module):
    """Ensemble of multiple models for improved performance"""
    
    def __init__(self, model_configs: List[Dict[str, Any]], num_classes: int = 5):
        super().__init__()
        
        self.models = nn.ModuleList()
        self.model_weights = nn.Parameter(torch.ones(len(model_configs)))
        
        for config in model_configs:
            if config['type'] == 'timm':
                model = timm.create_model(
                    config['name'],
                    pretrained=config.get('pretrained', True),
                    num_classes=num_classes
                )
            elif config['type'] == 'advanced_cnn':
                model = DRAdvancedCNN(
                    backbone=config['backbone'],
                    num_classes=num_classes,
                    pretrained=config.get('pretrained', True),
                    use_attention=config.get('use_attention', True)
                )
            else:
                raise ValueError(f"Unknown model type: {config['type']}")
            
            self.models.append(model)
        
        # Fusion layer
        self.fusion = nn.Linear(num_classes * len(model_configs), num_classes)
    
    def forward(self, x):
        outputs = []
        weights = F.softmax(self.model_weights, dim=0)
        
        for i, model in enumerate(self.models):
            out = model(x)
            # Apply learned weight
            out = out * weights[i]
            outputs.append(out)
        
        # Concatenate and fuse
        concatenated = torch.cat(outputs, dim=1)
        final_output = self.fusion(concatenated)
        
        return final_output


class VisionTransformerDR(nn.Module):
    """Vision Transformer adapted for diabetic retinopathy detection"""
    
    def __init__(self, model_name: str = 'vit_base_patch16_224', 
                 num_classes: int = 5, pretrained: bool = True):
        super().__init__()
        
        # Load pre-trained ViT
        self.vit = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0  # Remove head
        )
        
        # Get feature dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            features = self.vit(dummy_input)
            feature_dim = features.size(1)
        
        # Custom classification head for DR
        self.classifier = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Dropout(0.1),
            nn.Linear(feature_dim, feature_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim // 2, num_classes)
        )
        
        # Optional: Freeze early layers for fine-tuning
        self.freeze_early_layers()
    
    def freeze_early_layers(self, num_layers_to_freeze: int = 6):
        """Freeze early transformer layers for better fine-tuning"""
        for i, layer in enumerate(self.vit.blocks):
            if i < num_layers_to_freeze:
                for param in layer.parameters():
                    param.requires_grad = False
    
    def forward(self, x):
        features = self.vit(x)
        output = self.classifier(features)
        return output


class HybridCNNTransformer(nn.Module):
    """Hybrid CNN-Transformer model combining both architectures"""
    
    def __init__(self, cnn_backbone: str = 'efficientnet_b0', 
                 transformer_name: str = 'vit_small_patch16_224',
                 num_classes: int = 5, pretrained: bool = True):
        super().__init__()
        
        # CNN branch for local features
        self.cnn = DRAdvancedCNN(
            backbone=cnn_backbone,
            num_classes=128,  # Intermediate features
            pretrained=pretrained,
            use_attention=True
        )
        
        # Transformer branch for global features
        self.transformer = timm.create_model(
            transformer_name,
            pretrained=pretrained,
            num_classes=128  # Intermediate features
        )
        
        # Feature fusion and final classification
        self.fusion = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
        
        # Learnable fusion weights
        self.cnn_weight = nn.Parameter(torch.tensor(0.5))
        self.transformer_weight = nn.Parameter(torch.tensor(0.5))
    
    def forward(self, x):
        # Get features from both branches
        cnn_features = self.cnn(x)
        transformer_features = self.transformer(x)
        
        # Weighted fusion
        weights = F.softmax(torch.stack([self.cnn_weight, self.transformer_weight]), dim=0)
        weighted_cnn = cnn_features * weights[0]
        weighted_transformer = transformer_features * weights[1]
        
        # Concatenate and classify
        fused_features = torch.cat([weighted_cnn, weighted_transformer], dim=1)
        output = self.fusion(fused_features)
        
        return output


def create_advanced_model(model_type: str, num_classes: int = 5, **kwargs) -> nn.Module:
    """Factory function to create advanced models"""
    
    if model_type == 'advanced_cnn':
        return DRAdvancedCNN(
            backbone=kwargs.get('backbone', 'efficientnet_b0'),
            num_classes=num_classes,
            pretrained=kwargs.get('pretrained', True),
            use_attention=kwargs.get('use_attention', True)
        )
    
    elif model_type == 'vision_transformer':
        return VisionTransformerDR(
            model_name=kwargs.get('model_name', 'vit_base_patch16_224'),
            num_classes=num_classes,
            pretrained=kwargs.get('pretrained', True)
        )
    
    elif model_type == 'hybrid':
        return HybridCNNTransformer(
            cnn_backbone=kwargs.get('cnn_backbone', 'efficientnet_b0'),
            transformer_name=kwargs.get('transformer_name', 'vit_small_patch16_224'),
            num_classes=num_classes,
            pretrained=kwargs.get('pretrained', True)
        )
    
    elif model_type == 'ensemble':
        model_configs = kwargs.get('model_configs', [
            {'type': 'timm', 'name': 'efficientnet_b0'},
            {'type': 'timm', 'name': 'resnet50'},
            {'type': 'advanced_cnn', 'backbone': 'efficientnet_b1'}
        ])
        return EnsembleModel(model_configs, num_classes)
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# Pre-configured model recipes
MODEL_RECIPES = {
    'lightweight': {
        'type': 'advanced_cnn',
        'backbone': 'efficientnet_b0',
        'use_attention': True,
        'description': 'Lightweight model for resource-constrained environments'
    },
    
    'high_performance': {
        'type': 'ensemble',
        'model_configs': [
            {'type': 'advanced_cnn', 'backbone': 'efficientnet_b3', 'use_attention': True},
            {'type': 'timm', 'name': 'resnet101'},
            {'type': 'vision_transformer', 'model_name': 'vit_base_patch16_224'}
        ],
        'description': 'High-performance ensemble for maximum accuracy'
    },
    
    'transformer_based': {
        'type': 'vision_transformer',
        'model_name': 'vit_base_patch16_224',
        'description': 'Pure transformer model for global context understanding'
    },
    
    'hybrid_best': {
        'type': 'hybrid',
        'cnn_backbone': 'efficientnet_b2',
        'transformer_name': 'vit_small_patch16_224',
        'description': 'Best of both worlds - CNN local features + Transformer global context'
    }
}


def get_model_recipe(recipe_name: str, num_classes: int = 5) -> nn.Module:
    """Get a pre-configured model recipe"""
    if recipe_name not in MODEL_RECIPES:
        available = list(MODEL_RECIPES.keys())
        raise ValueError(f"Unknown recipe '{recipe_name}'. Available: {available}")
    
    recipe = MODEL_RECIPES[recipe_name].copy()
    model_type = recipe.pop('type')
    recipe.pop('description', None)
    
    logger.info(f"Creating model from recipe '{recipe_name}': {MODEL_RECIPES[recipe_name]['description']}")
    
    return create_advanced_model(model_type, num_classes, **recipe)


if __name__ == "__main__":
    # Test model creation
    print("Testing advanced model architectures...")
    
    # Test different model types
    models_to_test = [
        ('advanced_cnn', {}),
        ('vision_transformer', {}),
        ('hybrid', {}),
    ]
    
    for model_type, kwargs in models_to_test:
        print(f"\nTesting {model_type}...")
        try:
            model = create_advanced_model(model_type, num_classes=5, **kwargs)
            
            # Test forward pass
            dummy_input = torch.randn(2, 3, 224, 224)
            output = model(dummy_input)
            print(f"✅ {model_type}: Output shape {output.shape}")
            
            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"   Total params: {total_params:,}, Trainable: {trainable_params:,}")
            
        except Exception as e:
            print(f"❌ {model_type} failed: {e}")
    
    # Test model recipes
    print("\nTesting model recipes...")
    for recipe_name in MODEL_RECIPES:
        try:
            model = get_model_recipe(recipe_name)
            dummy_input = torch.randn(1, 3, 224, 224)
            output = model(dummy_input)
            print(f"✅ {recipe_name}: Output shape {output.shape}")
        except Exception as e:
            print(f"❌ {recipe_name} failed: {e}")
    
    print("\nAll tests completed!")