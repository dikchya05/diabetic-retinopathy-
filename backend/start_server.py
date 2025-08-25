#!/usr/bin/env python3
"""
Production server startup script for Diabetic Retinopathy API
"""
import os
import sys
import uvicorn
import argparse
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backend.app.config import settings


def main():
    parser = argparse.ArgumentParser(description='Start Diabetic Retinopathy API Server')
    
    parser.add_argument(
        '--host',
        default=settings.API_HOST,
        help=f'Host to bind to (default: {settings.API_HOST})'
    )
    
    parser.add_argument(
        '--port',
        type=int,
        default=settings.API_PORT,
        help=f'Port to bind to (default: {settings.API_PORT})'
    )
    
    parser.add_argument(
        '--workers',
        type=int,
        default=settings.API_WORKERS,
        help=f'Number of worker processes (default: {settings.API_WORKERS})'
    )
    
    parser.add_argument(
        '--reload',
        action='store_true',
        default=settings.is_development(),
        help='Enable auto-reload (development mode)'
    )
    
    parser.add_argument(
        '--log-level',
        default=settings.LOG_LEVEL.lower(),
        choices=['debug', 'info', 'warning', 'error', 'critical'],
        help=f'Log level (default: {settings.LOG_LEVEL.lower()})'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("DIABETIC RETINOPATHY DETECTION API")
    print("=" * 60)
    print(f"Environment: {settings.ENVIRONMENT}")
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    print(f"Workers: {args.workers}")
    print(f"Model: {settings.MODEL_PATH}")
    print(f"Device: {settings.MODEL_DEVICE}")
    print(f"Log Level: {args.log_level.upper()}")
    print("-" * 60)
    
    # Validate critical settings before starting
    print("Validating configuration...")
    
    # Check if model exists
    model_path = Path(settings.MODEL_PATH)
    if model_path.exists():
        print(f"Model found: {model_path}")
    else:
        print(f"WARNING: Model not found: {model_path}")
        print("   The server will start but predictions will fail until a model is available.")
    
    print("-" * 60)
    print("Starting server...")
    
    # Configure uvicorn settings
    uvicorn_config = {
        "app": "backend.app.main:app",
        "host": args.host,
        "port": args.port,
        "log_level": args.log_level,
        "reload": args.reload,
        "reload_dirs": [str(project_root / "backend")] if args.reload else None,
        "workers": 1 if args.reload else args.workers,  # Can't use multiple workers with reload
    }
    
    # Remove None values
    uvicorn_config = {k: v for k, v in uvicorn_config.items() if v is not None}
    
    try:
        uvicorn.run(**uvicorn_config)
    except KeyboardInterrupt:
        print("\nServer stopped by user")
    except Exception as e:
        print(f"\nERROR: Server failed to start: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()