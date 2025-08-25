"""
Production-ready FastAPI application for Diabetic Retinopathy Detection
"""
import logging
import time
from typing import Dict, Any
from fastapi import FastAPI, UploadFile, File, HTTPException, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from PIL import Image
import io
import traceback

from .predict import predict, gradcam_base64, validate_model_availability
from .config import settings

# Setup logging
logger = logging.getLogger(__name__)

# Initialize FastAPI app with proper configuration
app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    description="AI-powered diabetic retinopathy detection and grading system",
    docs_url="/docs" if settings.is_development() else None,
    redoc_url="/redoc" if settings.is_development() else None,
    openapi_url="/openapi.json" if settings.is_development() else None
)

# Add trusted host middleware for security
if not settings.is_development():
    app.add_middleware(
        TrustedHostMiddleware, 
        allowed_hosts=settings.ALLOWED_HOSTS
    )

# Add CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "HEAD", "OPTIONS"],
    allow_headers=["*"],
)


# Request size middleware
@app.middleware("http")
async def limit_upload_size(request: Request, call_next):
    """Middleware to limit request size"""
    if request.method == "POST":
        content_length = request.headers.get('content-length')
        if content_length:
            content_length = int(content_length)
            if content_length > settings.MAX_REQUEST_SIZE:
                return JSONResponse(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    content={"detail": f"Request size {content_length} exceeds maximum {settings.MAX_REQUEST_SIZE}"}
                )
    
    response = await call_next(request)
    return response


# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Middleware to log requests"""
    start_time = time.time()
    
    response = await call_next(request)
    
    process_time = time.time() - start_time
    logger.info(
        f"{request.method} {request.url.path} - "
        f"Status: {response.status_code} - "
        f"Time: {process_time:.3f}s"
    )
    
    return response

@app.get("/", tags=["General"])
def read_root():
    """Root endpoint returning API information"""
    return {
        "message": "Diabetic Retinopathy Detection API",
        "version": settings.API_VERSION,
        "status": "operational"
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Comprehensive health check endpoint"""
    try:
        # Check model availability
        model_status = validate_model_availability()
        
        health_status = {
            "status": "healthy" if model_status["available"] else "degraded",
            "timestamp": time.time(),
            "version": settings.API_VERSION,
            "environment": settings.ENVIRONMENT,
            "model": {
                "available": model_status["available"],
                "path": str(settings.MODEL_PATH),
                "device": settings.MODEL_DEVICE
            },
            "configuration": {
                "image_size": settings.IMAGE_SIZE,
                "max_file_size_mb": settings.MAX_IMAGE_SIZE_MB,
                "gradcam_enabled": settings.ENABLE_GRADCAM
            }
        }
        
        if not model_status["available"]:
            health_status["error"] = model_status.get("error", "Model not available")
        
        status_code = 200 if model_status["available"] else 503
        return JSONResponse(content=health_status, status_code=status_code)
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            content={
                "status": "unhealthy",
                "timestamp": time.time(),
                "error": str(e)
            },
            status_code=503
        )


@app.get("/info", tags=["General"])
async def get_api_info():
    """Get API information and configuration"""
    return {
        "api": {
            "title": settings.API_TITLE,
            "version": settings.API_VERSION,
            "environment": settings.ENVIRONMENT
        },
        "model": {
            "classes": settings.CLASS_NAMES,
            "num_classes": settings.N_CLASSES,
            "input_size": settings.IMAGE_SIZE
        },
        "limits": {
            "max_file_size_mb": settings.MAX_IMAGE_SIZE_MB,
            "supported_formats": settings.SUPPORTED_IMAGE_FORMATS
        },
        "features": {
            "gradcam_visualization": settings.ENABLE_GRADCAM
        }
    }


@app.post("/predict", tags=["Prediction"])
async def predict_endpoint(file: UploadFile = File(...)):
    """
    Predict diabetic retinopathy severity from retinal image
    
    Parameters:
    - file: Image file (JPEG, PNG) of retinal fundus photograph
    
    Returns:
    - prediction: Predicted class index (0-4)
    - label: Predicted class name
    - confidence: Confidence score for the prediction
    - probabilities: Probability scores for all classes
    - gradcam_base64: GradCAM visualization (if enabled)
    """
    request_start_time = time.time()
    
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(
                status_code=400, 
                detail="No file provided"
            )
        
        # Check file size
        file_content = await file.read()
        file_size_mb = len(file_content) / (1024 * 1024)
        
        if file_size_mb > settings.MAX_IMAGE_SIZE_MB:
            raise HTTPException(
                status_code=413,
                detail=f"File size {file_size_mb:.2f}MB exceeds limit of {settings.MAX_IMAGE_SIZE_MB}MB"
            )
        
        # Validate content type
        if file.content_type not in settings.SUPPORTED_IMAGE_FORMATS:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file format. Supported formats: {', '.join(settings.SUPPORTED_IMAGE_FORMATS)}"
            )
        
        # Load and validate image
        try:
            image = Image.open(io.BytesIO(file_content))
            # Validate image
            image.verify()
            # Reload image for processing (verify() closes the image)
            image = Image.open(io.BytesIO(file_content)).convert("RGB")
            
            # Check image dimensions
            width, height = image.size
            if min(width, height) < 100:
                raise HTTPException(
                    status_code=400,
                    detail=f"Image too small ({width}x{height}). Minimum size is 100x100 pixels"
                )
                
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid or corrupted image file: {str(e)}"
            )
        
        # Log prediction request
        logger.info(f"Processing prediction request - File: {file.filename}, Size: {file_size_mb:.2f}MB")
        
        # Make prediction
        pred_idx, confidence, probs = predict(image)
        
        # Generate GradCAM if enabled
        cam_b64 = None
        if settings.ENABLE_GRADCAM:
            try:
                cam_b64 = gradcam_base64(image)
            except Exception as e:
                logger.warning(f"GradCAM generation failed: {e}")
                cam_b64 = None
        
        # Prepare response
        response_data = {
            "success": True,
            "prediction": int(pred_idx),
            "label": settings.CLASS_NAMES[int(pred_idx)],
            "confidence": float(confidence),
            "probabilities": [float(p) for p in probs],
            "processing_time_ms": int((time.time() - request_start_time) * 1000),
            "metadata": {
                "image_size": f"{width}x{height}",
                "file_size_mb": round(file_size_mb, 2),
                "model_device": settings.MODEL_DEVICE
            }
        }
        
        if cam_b64:
            response_data["gradcam_base64"] = cam_b64
        
        logger.info(f"Prediction completed - Result: {settings.CLASS_NAMES[int(pred_idx)]} ({confidence:.3f})")
        
        return JSONResponse(content=response_data)

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        if settings.is_development():
            traceback.print_exc()
        
        raise HTTPException(
            status_code=500, 
            detail="Internal server error during prediction" if settings.is_production() else str(e)
        )


# Error handlers
@app.exception_handler(413)
async def request_entity_too_large_handler(request: Request, exc):
    return JSONResponse(
        status_code=413,
        content={"detail": "Request entity too large"}
    )


@app.exception_handler(500)
async def internal_server_error_handler(request: Request, exc):
    logger.error(f"Internal server error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )
