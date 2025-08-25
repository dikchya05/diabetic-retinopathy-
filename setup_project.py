#!/usr/bin/env python3
"""
Automated setup script for Diabetic Retinopathy Detection Project
"""
import os
import sys
import subprocess
import platform
from pathlib import Path


def print_step(step, message):
    """Print a colored step message"""
    colors = {
        'info': '\033[94m',  # Blue
        'success': '\033[92m',  # Green  
        'warning': '\033[93m',  # Yellow
        'error': '\033[91m',  # Red
        'end': '\033[0m'  # End color
    }
    
    if step == 'info':
        print(f"{colors['info']}ℹ️  {message}{colors['end']}")
    elif step == 'success':
        print(f"{colors['success']}✅ {message}{colors['end']}")
    elif step == 'warning':
        print(f"{colors['warning']}⚠️  {message}{colors['end']}")
    elif step == 'error':
        print(f"{colors['error']}❌ {message}{colors['end']}")
    else:
        print(message)


def run_command(command, description, check=True):
    """Run a command and handle errors"""
    print_step('info', f"{description}...")
    print(f"Running: {command}")
    
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            capture_output=True, 
            text=True, 
            check=check
        )
        
        if result.returncode == 0:
            print_step('success', f"{description} completed")
            if result.stdout:
                print(result.stdout)
        else:
            print_step('error', f"{description} failed")
            if result.stderr:
                print(result.stderr)
            return False
            
    except subprocess.CalledProcessError as e:
        print_step('error', f"{description} failed: {e}")
        return False
    except Exception as e:
        print_step('error', f"Unexpected error: {e}")
        return False
    
    return True


def check_python_version():
    """Check if Python version is compatible"""
    print_step('info', "Checking Python version...")
    
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print_step('error', f"Python {version.major}.{version.minor} is not supported")
        print_step('error', "Please install Python 3.8 or higher")
        return False
    
    print_step('success', f"Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True


def check_system_requirements():
    """Check system requirements"""
    print_step('info', "Checking system requirements...")
    
    # Check operating system
    os_name = platform.system()
    print_step('info', f"Operating system: {os_name}")
    
    # Check if we're in the right directory
    if not Path('requirements.txt').exists():
        print_step('error', "requirements.txt not found. Make sure you're in the project directory.")
        return False
    
    print_step('success', "System requirements check passed")
    return True


def setup_virtual_environment():
    """Create and activate virtual environment"""
    print_step('info', "Setting up virtual environment...")
    
    venv_path = Path('venv')
    
    # Create virtual environment if it doesn't exist
    if not venv_path.exists():
        if not run_command('python -m venv venv', 'Creating virtual environment'):
            return False
    else:
        print_step('info', "Virtual environment already exists")
    
    # Activation command varies by OS
    if platform.system() == 'Windows':
        activate_cmd = 'venv\\Scripts\\activate'
        pip_cmd = 'venv\\Scripts\\python -m pip'
    else:
        activate_cmd = 'source venv/bin/activate'
        pip_cmd = 'venv/bin/python -m pip'
    
    print_step('success', f"Virtual environment ready. Use '{activate_cmd}' to activate.")
    return pip_cmd


def install_pytorch(pip_cmd):
    """Install PyTorch with appropriate version"""
    print_step('info', "Installing PyTorch...")
    
    # Check for CUDA availability
    try:
        result = subprocess.run(
            'nvidia-smi', 
            shell=True, 
            capture_output=True, 
            text=True
        )
        cuda_available = result.returncode == 0
    except:
        cuda_available = False
    
    if cuda_available:
        print_step('info', "NVIDIA GPU detected, installing PyTorch with CUDA support")
        pytorch_cmd = f'{pip_cmd} install torch torchvision --index-url https://download.pytorch.org/whl/cu118'
    else:
        print_step('info', "No GPU detected, installing CPU-only PyTorch")
        pytorch_cmd = f'{pip_cmd} install torch torchvision --index-url https://download.pytorch.org/whl/cpu'
    
    return run_command(pytorch_cmd, 'Installing PyTorch')


def install_dependencies(pip_cmd):
    """Install all project dependencies"""
    print_step('info', "Installing project dependencies...")
    
    # Upgrade pip first
    if not run_command(f'{pip_cmd} install --upgrade pip', 'Upgrading pip'):
        return False
    
    # Install main requirements
    if not run_command(f'{pip_cmd} install -r requirements.txt', 'Installing main dependencies'):
        return False
    
    # Install backend requirements
    if Path('backend/requirements.txt').exists():
        if not run_command(f'{pip_cmd} install -r backend/requirements.txt', 'Installing backend dependencies'):
            return False
    
    return True


def verify_installation(pip_cmd):
    """Verify that all dependencies are properly installed"""
    print_step('info', "Verifying installation...")
    
    # Test core imports
    test_script = '''
import torch
import torchvision
import cv2
import numpy as np
import pandas as pd
import matplotlib
import timm
import albumentations
from sklearn import metrics
import fastapi
import pytest
print("✅ All core dependencies verified!")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    try:
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    except:
        print("CUDA device info not available")
'''
    
    # Write test script to temporary file
    with open('temp_test.py', 'w') as f:
        f.write(test_script)
    
    # Run test
    if platform.system() == 'Windows':
        python_cmd = 'venv\\Scripts\\python'
    else:
        python_cmd = 'venv/bin/python'
    
    success = run_command(f'{python_cmd} temp_test.py', 'Verifying core dependencies')
    
    # Clean up
    try:
        os.remove('temp_test.py')
    except:
        pass
    
    return success


def test_project_modules(pip_cmd):
    """Test if project modules can be imported"""
    print_step('info', "Testing project modules...")
    
    test_script = '''
import sys
import os
sys.path.append('.')
try:
    from ml.utils import RetinopathyDataset, get_transforms
    from ml.evaluation import ModelEvaluator  
    from ml.models.advanced_architectures import create_advanced_model
    from backend.app.config import settings
    print("✅ All project modules imported successfully!")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)
'''
    
    with open('temp_module_test.py', 'w') as f:
        f.write(test_script)
    
    if platform.system() == 'Windows':
        python_cmd = 'venv\\Scripts\\python'
    else:
        python_cmd = 'venv/bin/python'
    
    success = run_command(f'{python_cmd} temp_module_test.py', 'Testing project modules')
    
    # Clean up
    try:
        os.remove('temp_module_test.py')
    except:
        pass
    
    return success


def setup_environment_file():
    """Setup environment configuration"""
    print_step('info', "Setting up environment configuration...")
    
    if not Path('.env').exists() and Path('.env.example').exists():
        if platform.system() == 'Windows':
            run_command('copy .env.example .env', 'Creating .env file', check=False)
        else:
            run_command('cp .env.example .env', 'Creating .env file', check=False)
        print_step('success', "Environment file created from template")
    else:
        print_step('info', "Environment file already exists or template not found")


def print_next_steps():
    """Print instructions for next steps"""
    print("\n" + "="*60)
    print_step('success', "🎉 Installation completed successfully!")
    print("="*60)
    
    print("\n📋 Next Steps:")
    
    if platform.system() == 'Windows':
        activate_cmd = 'venv\\Scripts\\activate'
    else:
        activate_cmd = 'source venv/bin/activate'
    
    steps = [
        f"1. Activate virtual environment: {activate_cmd}",
        "2. Download dataset from Kaggle APTOS 2019 Blindness Detection",
        "3. Configure .env file with your settings",
        "4. Train a model: python -m ml.train_advanced --help",
        "5. Start API server: python backend/start_server.py",
        "6. Run tests: pytest tests/ -v",
        "7. Check installation guide: cat INSTALLATION.md"
    ]
    
    for step in steps:
        print(f"   {step}")
    
    print(f"\n🐍 To activate your environment:")
    print(f"   {activate_cmd}")
    
    print(f"\n🧪 To test everything:")
    print(f"   pytest tests/ -v")
    
    print(f"\n🚀 To start the API:")
    print(f"   python backend/start_server.py")
    
    print(f"\n📖 For more details, see INSTALLATION.md")


def main():
    """Main setup function"""
    print("🏥 Diabetic Retinopathy Detection - Project Setup")
    print("=" * 50)
    
    # Check prerequisites
    if not check_python_version():
        return False
    
    if not check_system_requirements():
        return False
    
    # Setup virtual environment
    pip_cmd = setup_virtual_environment()
    if not pip_cmd:
        return False
    
    # Install dependencies
    if not install_pytorch(pip_cmd):
        print_step('warning', "PyTorch installation failed, trying alternative method...")
        if not run_command(f'{pip_cmd} install torch torchvision', 'Installing PyTorch (fallback)'):
            return False
    
    if not install_dependencies(pip_cmd):
        return False
    
    # Setup configuration
    setup_environment_file()
    
    # Verify installation
    if not verify_installation(pip_cmd):
        print_step('warning', "Verification failed, but installation may still work")
    
    if not test_project_modules(pip_cmd):
        print_step('warning', "Project modules test failed, but core dependencies are installed")
    
    # Show next steps
    print_next_steps()
    
    return True


if __name__ == '__main__':
    try:
        success = main()
        if not success:
            print_step('error', "Setup failed. Please check the error messages above.")
            print_step('info', "You can also try manual installation following INSTALLATION.md")
            sys.exit(1)
        else:
            print_step('success', "Setup completed successfully! 🎉")
    except KeyboardInterrupt:
        print_step('warning', "\nSetup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print_step('error', f"Unexpected error during setup: {e}")
        sys.exit(1)