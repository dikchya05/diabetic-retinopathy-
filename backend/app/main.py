from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Form, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from PIL import Image
import io
import traceback
import uuid
from datetime import datetime

from .predict import predict, gradcam_base64
from .config import CLASS_NAMES
from .medical_info import get_comprehensive_analysis
from .database import engine, Base, get_db
from .models import Patient, RetinalScan
from .routes import router
from .schemas import PredictionResponse, ScanHistoryItem
from typing import List, Optional

# Create database tables
Base.metadata.create_all(bind=engine)

app = FastAPI(title="DR Inference API")

# Add CORS Middleware
origins = [
    "http://localhost:3000",  # Next.js dev server
    "http://localhost:3001",  # Next.js alternate port
    "http://localhost:3002",  # Next.js alternate port
    "http://127.0.0.1:3000",  # Alternate localhost
    "http://127.0.0.1:3001",  # Alternate localhost
    "http://127.0.0.1:3002",  # Alternate localhost
    "https://yourfrontend.com"  # Production domain
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,        # Allowed origins
    allow_credentials=True,
    allow_methods=["*"],          # Allow all HTTP methods
    allow_headers=["*"],          # Allow all headers
)

# Include API routes
app.include_router(router)

@app.get("/")
def read_root():
    return {"message": "Diabetic Retinopathy Detection API", "version": "2.0"}

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictionResponse)
async def predict_endpoint(
    request: Request,
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    try:
        # Debug: Check the form data from request
        form_data = await request.form()
        print(f"=== FORM DATA DEBUG ===")
        print(f"All form keys: {list(form_data.keys())}")
        patient_id = form_data.get("patient_id")
        eye = form_data.get("eye", "Both")
        print(f"patient_id from form: '{patient_id}' (type: {type(patient_id)})")
        print(f"eye from form: '{eye}' (type: {type(eye)})")
        print(f"======================")
        
        # Convert patient_id from string to int if provided
        patient_id_int = None
        if patient_id:
            try:
                patient_id_int = int(patient_id)
                print(f"Processing scan for patient_id: {patient_id_int}")
            except ValueError:
                print(f"Invalid patient_id received: {patient_id}")
        else:
            print("Processing anonymous scan (no patient_id)")
            
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="Upload an image file.")
        img_bytes = await file.read()
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        pred_idx, confidence, probs = predict(image)

        try:
            cam_b64 = gradcam_base64(image)
        except Exception:
            cam_b64 = None
        
        # Get comprehensive medical analysis
        medical_analysis = get_comprehensive_analysis(int(pred_idx), confidence)
        
        # Generate scan ID
        scan_id = f"SCAN-{str(uuid.uuid4())[:8].upper()}"
        
        # Save to database if patient_id is provided
        if patient_id_int:
            # Extract follow-up weeks from the follow_up string
            follow_up_str = medical_analysis.get("follow_up", "")
            follow_up_weeks = 4  # Default
            if "month" in follow_up_str.lower():
                if "1 month" in follow_up_str.lower():
                    follow_up_weeks = 4
                elif "3 month" in follow_up_str.lower():
                    follow_up_weeks = 12
                elif "6 month" in follow_up_str.lower():
                    follow_up_weeks = 26
                elif "12 month" in follow_up_str.lower():
                    follow_up_weeks = 52
            elif "week" in follow_up_str.lower():
                try:
                    import re
                    weeks_match = re.search(r'(\d+)\s*week', follow_up_str.lower())
                    if weeks_match:
                        follow_up_weeks = int(weeks_match.group(1))
                except:
                    pass
            
            # Create scan record
            scan = RetinalScan(
                scan_id=scan_id,
                patient_id=patient_id_int,
                eye=eye,
                prediction_class=int(pred_idx),
                prediction_label=CLASS_NAMES[int(pred_idx)],
                confidence_score=confidence,
                probabilities=probs,
                severity=medical_analysis["severity"],
                risk_level=medical_analysis["risk_level"],
                urgency=medical_analysis["urgency"],
                follow_up_weeks=follow_up_weeks,
                current_state=medical_analysis["current_state"],
                recommendations=medical_analysis["recommendations"],
                risk_factors=medical_analysis["risk_factors"],
                prevention_tips=medical_analysis["prevention_tips"],
                statistics=medical_analysis["statistics"]
            )
            
            db.add(scan)
            db.commit()
            db.refresh(scan)
        
        # Prepare response
        response_data = {
            "scan_id": scan_id,
            "patient_id": patient_id_int,
            "prediction": int(pred_idx),
            "label": CLASS_NAMES[int(pred_idx)],
            "confidence": confidence,
            "probabilities": probs,
            "gradcam_base64": cam_b64,
            
            # Medical Information
            "severity": medical_analysis["severity"],
            "description": medical_analysis["description"],
            "risk_level": medical_analysis["risk_level"],
            
            # Current State Analysis
            "current_state": medical_analysis["current_state"],
            
            # Recommendations
            "recommendations": medical_analysis["recommendations"],
            "follow_up": medical_analysis["follow_up"],
            "urgency": medical_analysis["urgency"],
            
            # Risk Assessment
            "risk_factors": medical_analysis["risk_factors"],
            "prevention_tips": medical_analysis["prevention_tips"],
            
            # Statistics
            "statistics": medical_analysis["statistics"],
            
            # General Advice
            "general_advice": medical_analysis["general_advice"],
            
            # Confidence Note
            "confidence_note": medical_analysis["confidence_note"],
            
            # Resources
            "resources": medical_analysis["resources"]
        }

        return JSONResponse(response_data)

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

@app.get("/api/patients/{patient_id}/scans", response_model=List[ScanHistoryItem])
def get_patient_scans(patient_id: int, db: Session = Depends(get_db)):
    """Get all scan history for a specific patient"""
    scans = db.query(RetinalScan).filter(
        RetinalScan.patient_id == patient_id
    ).order_by(RetinalScan.scan_date.desc()).all()
    
    result = []
    for scan in scans:
        result.append(ScanHistoryItem(
            id=scan.id,
            scan_id=scan.scan_id,
            scan_date=scan.scan_date.isoformat() if hasattr(scan.scan_date, 'isoformat') else str(scan.scan_date),
            prediction_class=scan.prediction_class,
            prediction_label=scan.prediction_label,
            confidence_score=scan.confidence_score,
            severity=scan.severity,
            risk_level=scan.risk_level,
            urgency=scan.urgency,
            recommendations=scan.recommendations or [],
            current_state=scan.current_state or {},
            risk_factors=scan.risk_factors or {},
            statistics=scan.statistics or {}
        ))
    
    return result
