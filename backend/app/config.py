"""
Production-ready configuration for Diabetic Retinopathy API
"""
import os
import logging
from pathlib import Path
from typing import List, Optional
from functools import lru_cache
import torch


class Settings:
    """Application settings with environment variable support"""
    
    def __init__(self):
        # Base paths
        self.BASE_DIR = Path(__file__).parent.parent.absolute()
        self.PROJECT_ROOT = self.BASE_DIR.parent
        
        # Environment
        self.ENVIRONMENT = os.getenv('ENVIRONMENT', 'development')
        self.DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'
        
        # API Configuration
        self.API_HOST = os.getenv('API_HOST', '0.0.0.0')
        self.API_PORT = int(os.getenv('API_PORT', '8000'))
        self.API_WORKERS = int(os.getenv('API_WORKERS', '1'))
        self.API_TITLE = os.getenv('API_TITLE', 'Diabetic Retinopathy Detection API')
        self.API_VERSION = os.getenv('API_VERSION', '1.0.0')
        
        # CORS Configuration
        self.CORS_ORIGINS = self._parse_cors_origins()
        
        # Model Configuration
        self.MODEL_PATH = self._get_model_path()
        self.MODEL_DEVICE = self._get_device()
        self.IMAGE_SIZE = int(os.getenv('IMAGE_SIZE', '224'))
        self.MAX_IMAGE_SIZE_MB = int(os.getenv('MAX_IMAGE_SIZE_MB', '10'))
        self.SUPPORTED_IMAGE_FORMATS = ['image/jpeg', 'image/jpg', 'image/png']
        
        # Class Configuration
        self.CLASS_NAMES = [
            'No DR', 'Mild DR', 'Moderate DR', 'Severe DR', 'Proliferative DR'
        ]
        self.N_CLASSES = len(self.CLASS_NAMES)
        
        # Logging Configuration
        self.LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
        self.LOG_FORMAT = os.getenv(
            'LOG_FORMAT', 
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Security Configuration
        self.ALLOWED_HOSTS = self._parse_allowed_hosts()
        self.MAX_REQUEST_SIZE = int(os.getenv('MAX_REQUEST_SIZE_MB', '25')) * 1024 * 1024
        
        # Performance Configuration
        self.ENABLE_GRADCAM = os.getenv('ENABLE_GRADCAM', 'True').lower() == 'true'
        self.BATCH_SIZE = int(os.getenv('INFERENCE_BATCH_SIZE', '1'))
        
        # Cache Configuration
        self.ENABLE_MODEL_CACHE = os.getenv('ENABLE_MODEL_CACHE', 'True').lower() == 'true'
        
        # Health Check Configuration
        self.HEALTH_CHECK_TIMEOUT = int(os.getenv('HEALTH_CHECK_TIMEOUT', '30'))
        
        # Setup logging
        self._setup_logging()
        
        # Validate configuration
        self._validate_config()
    
    def _parse_cors_origins(self) -> List[str]:
        """Parse CORS origins from environment variable"""
        origins_str = os.getenv('CORS_ORIGINS', 'http://localhost:3000,http://127.0.0.1:3000,http://localhost:3001,http://127.0.0.1:3001')
        return [origin.strip() for origin in origins_str.split(',') if origin.strip()]
    
    def _parse_allowed_hosts(self) -> List[str]:
        """Parse allowed hosts from environment variable"""
        hosts_str = os.getenv('ALLOWED_HOSTS', '*')
        if hosts_str == '*':
            return ['*']
        return [host.strip() for host in hosts_str.split(',') if host.strip()]
    
    def _get_model_path(self) -> Path:
        """Get model path with fallback options"""
        model_path_env = os.getenv('MODEL_PATH')
        
        if model_path_env:
            model_path = Path(model_path_env)
            if model_path.is_absolute():
                return model_path
            else:
                return self.PROJECT_ROOT / model_path
        
        # Default search locations
        search_paths = [
            self.PROJECT_ROOT / 'ml' / 'models' / 'best_model.pth',
            self.PROJECT_ROOT / 'ml' / 'models' / 'best_model-1.pth',
            self.PROJECT_ROOT / 'best_model.pth',
            Path.cwd() / 'best_model.pth'
        ]
        
        for path in search_paths:
            if path.exists():
                logging.info(f"Found model at: {path}")
                return path
        
        # If no model found, use the primary default
        default_path = self.PROJECT_ROOT / 'ml' / 'models' / 'best_model.pth'
        logging.warning(f"Model not found in search paths. Using default: {default_path}")
        return default_path
    
    def _get_device(self) -> str:
        """Determine compute device with proper error handling"""
        try:
            use_cuda = os.getenv('USE_CUDA', 'auto').lower()
            
            if use_cuda == 'false':
                return 'cpu'
            elif use_cuda == 'true':
                if torch.cuda.is_available():
                    device = f'cuda:0'
                    logging.info(f"Using GPU device: {device}")
                    return device
                else:
                    logging.warning("CUDA requested but not available, falling back to CPU")
                    return 'cpu'
            else:  # auto
                if torch.cuda.is_available():
                    device = f'cuda:0'
                    logging.info(f"Auto-detected GPU device: {device}")
                    return device
                else:
                    logging.info("Auto-detected CPU device")
                    return 'cpu'
        except Exception as e:
            logging.error(f"Error determining device: {e}")
            return 'cpu'
    
    def _setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=getattr(logging, self.LOG_LEVEL.upper()),
            format=self.LOG_FORMAT,
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('api.log') if self.ENVIRONMENT == 'production' else logging.NullHandler()
            ]
        )
    
    def _validate_config(self):
        """Validate critical configuration settings"""
        errors = []
        
        # Check model path
        if not self.MODEL_PATH.exists():
            errors.append(f"Model file not found: {self.MODEL_PATH}")
        
        # Check image size
        if self.IMAGE_SIZE <= 0 or self.IMAGE_SIZE > 1024:
            errors.append(f"Invalid image size: {self.IMAGE_SIZE}")
        
        # Check max file size
        if self.MAX_IMAGE_SIZE_MB <= 0:
            errors.append(f"Invalid max image size: {self.MAX_IMAGE_SIZE_MB}MB")
        
        if errors:
            error_msg = "\n".join(f"  - {error}" for error in errors)
            logging.error(f"Configuration validation failed:\n{error_msg}")
            if self.ENVIRONMENT == 'production':
                raise ValueError(f"Invalid configuration: {error_msg}")
    
    def is_production(self) -> bool:
        """Check if running in production environment"""
        return self.ENVIRONMENT.lower() == 'production'
    
    def is_development(self) -> bool:
        """Check if running in development environment"""
        return self.ENVIRONMENT.lower() == 'development'


@lru_cache()
def get_settings() -> Settings:
    """Get cached application settings"""
    return Settings()


# Create global settings instance
settings = get_settings()

# Export commonly used settings for backwards compatibility
MODEL_PATH = str(settings.MODEL_PATH)
IMAGE_SIZE = settings.IMAGE_SIZE
CLASS_NAMES = settings.CLASS_NAMES
DEVICE = settings.MODEL_DEVICE
