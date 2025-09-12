"""
API routes for patient management and scan history
"""
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime, timedelta
import uuid
import json

from .database import get_db
from .models import Patient, RetinalScan, ScanComparison
from .schemas import (
    PatientCreate, PatientUpdate, PatientResponse,
    RetinalScanCreate, RetinalScanResponse, ScanHistoryResponse,
    ScanComparisonCreate, ScanComparisonResponse,
    StatisticsResponse
)

router = APIRouter()

# Patient Routes
@router.post("/api/patients", response_model=PatientResponse)
async def create_patient(patient: PatientCreate, db: Session = Depends(get_db)):
    """Create a new patient record"""
    # Check if patient already exists
    existing = db.query(Patient).filter(
        (Patient.email == patient.email) | 
        (Patient.patient_id == patient.patient_id)
    ).first()
    
    if existing:
        raise HTTPException(status_code=400, detail="Patient already exists")
    
    # Generate patient ID if not provided
    if not patient.patient_id:
        patient.patient_id = f"PAT-{str(uuid.uuid4())[:8].upper()}"
    
    db_patient = Patient(**patient.dict())
    db.add(db_patient)
    db.commit()
    db.refresh(db_patient)
    
    return db_patient

@router.get("/api/patients", response_model=List[PatientResponse])
async def list_patients(
    skip: int = 0,
    limit: int = 100,
    search: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """List all patients with optional search"""
    query = db.query(Patient)
    
    if search:
        search_term = f"%{search}%"
        query = query.filter(
            (Patient.first_name.ilike(search_term)) |
            (Patient.last_name.ilike(search_term)) |
            (Patient.email.ilike(search_term)) |
            (Patient.patient_id.ilike(search_term))
        )
    
    patients = query.offset(skip).limit(limit).all()
    return patients

@router.get("/api/patients/{patient_id}", response_model=PatientResponse)
async def get_patient(patient_id: int, db: Session = Depends(get_db)):
    """Get a specific patient by ID"""
    patient = db.query(Patient).filter(Patient.id == patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    return patient

@router.put("/api/patients/{patient_id}", response_model=PatientResponse)
async def update_patient(
    patient_id: int,
    patient_update: PatientUpdate,
    db: Session = Depends(get_db)
):
    """Update patient information"""
    patient = db.query(Patient).filter(Patient.id == patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    update_data = patient_update.dict(exclude_unset=True)
    for field, value in update_data.items():
        setattr(patient, field, value)
    
    patient.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(patient)
    
    return patient

@router.delete("/api/patients/{patient_id}")
async def delete_patient(patient_id: int, db: Session = Depends(get_db)):
    """Delete a patient and all associated scans"""
    patient = db.query(Patient).filter(Patient.id == patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    db.delete(patient)
    db.commit()
    
    return {"message": "Patient deleted successfully"}

# Scan Routes
@router.get("/api/patients/{patient_id}/scans", response_model=List[RetinalScanResponse])
async def get_patient_scans(
    patient_id: int,
    eye: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Get all scans for a patient"""
    patient = db.query(Patient).filter(Patient.id == patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    query = db.query(RetinalScan).filter(RetinalScan.patient_id == patient_id)
    
    if eye:
        query = query.filter(RetinalScan.eye == eye)
    
    scans = query.order_by(RetinalScan.scan_date.desc()).all()
    return scans

@router.get("/api/patients/{patient_id}/history", response_model=ScanHistoryResponse)
async def get_patient_history(patient_id: int, db: Session = Depends(get_db)):
    """Get complete scan history for a patient"""
    patient = db.query(Patient).filter(Patient.id == patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    scans = db.query(RetinalScan).filter(
        RetinalScan.patient_id == patient_id
    ).order_by(RetinalScan.scan_date.desc()).all()
    
    # Determine progression trend
    progression_trend = None
    if len(scans) >= 2:
        latest = scans[0].prediction_class
        previous = scans[1].prediction_class
        
        if latest < previous:
            progression_trend = "Improving"
        elif latest > previous:
            progression_trend = "Worsening"
        else:
            progression_trend = "Stable"
    
    return ScanHistoryResponse(
        patient=patient,
        scans=scans,
        total_scans=len(scans),
        latest_scan=scans[0] if scans else None,
        progression_trend=progression_trend
    )

@router.post("/api/scans/{scan_id}/verify")
async def verify_scan(
    scan_id: int,
    clinician_diagnosis: str,
    clinician_notes: Optional[str] = None,
    verified_by: str = "Clinician",
    db: Session = Depends(get_db)
):
    """Mark a scan as clinically verified"""
    scan = db.query(RetinalScan).filter(RetinalScan.id == scan_id).first()
    if not scan:
        raise HTTPException(status_code=404, detail="Scan not found")
    
    scan.clinician_verified = True
    scan.clinician_diagnosis = clinician_diagnosis
    scan.clinician_notes = clinician_notes
    scan.verified_by = verified_by
    scan.verified_at = datetime.utcnow()
    
    db.commit()
    db.refresh(scan)
    
    return {"message": "Scan verified successfully", "scan_id": scan.scan_id}

# Comparison Routes
@router.post("/api/comparisons", response_model=ScanComparisonResponse)
async def create_comparison(
    comparison: ScanComparisonCreate,
    db: Session = Depends(get_db)
):
    """Create a comparison between two scans"""
    # Get both scans
    baseline = db.query(RetinalScan).filter(
        RetinalScan.id == comparison.baseline_scan_id
    ).first()
    current = db.query(RetinalScan).filter(
        RetinalScan.id == comparison.current_scan_id
    ).first()
    
    if not baseline or not current:
        raise HTTPException(status_code=404, detail="One or both scans not found")
    
    # Calculate progression
    severity_change = current.prediction_class - baseline.prediction_class
    confidence_change = current.confidence_score - baseline.confidence_score
    
    if severity_change < 0:
        progression_status = "Improved"
    elif severity_change > 0:
        progression_status = "Worsened"
    else:
        progression_status = "Stable"
    
    # Create comparison record
    db_comparison = ScanComparison(
        patient_id=comparison.patient_id,
        baseline_scan_id=comparison.baseline_scan_id,
        current_scan_id=comparison.current_scan_id,
        progression_status=progression_status,
        severity_change=severity_change,
        confidence_change=confidence_change
    )
    
    db.add(db_comparison)
    db.commit()
    db.refresh(db_comparison)
    
    return db_comparison

# Statistics Routes
@router.get("/api/statistics", response_model=StatisticsResponse)
async def get_statistics(db: Session = Depends(get_db)):
    """Get system-wide statistics"""
    total_patients = db.query(Patient).count()
    total_scans = db.query(RetinalScan).count()
    
    # Time-based statistics
    today = datetime.utcnow().date()
    week_ago = today - timedelta(days=7)
    month_ago = today - timedelta(days=30)
    
    scans_today = db.query(RetinalScan).filter(
        RetinalScan.scan_date >= today
    ).count()
    
    scans_this_week = db.query(RetinalScan).filter(
        RetinalScan.scan_date >= week_ago
    ).count()
    
    scans_this_month = db.query(RetinalScan).filter(
        RetinalScan.scan_date >= month_ago
    ).count()
    
    # Distribution statistics
    severity_distribution = {}
    urgency_distribution = {}
    risk_level_distribution = {}
    
    scans = db.query(RetinalScan).all()
    
    for scan in scans:
        # Severity distribution
        severity = scan.severity or "Unknown"
        severity_distribution[severity] = severity_distribution.get(severity, 0) + 1
        
        # Urgency distribution
        urgency = scan.urgency or "Unknown"
        urgency_distribution[urgency] = urgency_distribution.get(urgency, 0) + 1
        
        # Risk level distribution
        risk_level = scan.risk_level or "Unknown"
        risk_level_distribution[risk_level] = risk_level_distribution.get(risk_level, 0) + 1
    
    # Calculate averages
    avg_confidence = 0
    if scans:
        avg_confidence = sum(s.confidence_score for s in scans if s.confidence_score) / len(scans)
    
    # High risk patients
    high_risk_patients = db.query(Patient).join(RetinalScan).filter(
        RetinalScan.prediction_class >= 3
    ).distinct().count()
    
    # Patients needing follow-up
    two_weeks_ago = datetime.utcnow() - timedelta(weeks=2)
    patients_needing_followup = db.query(Patient).join(RetinalScan).filter(
        RetinalScan.scan_date <= two_weeks_ago,
        RetinalScan.follow_up_weeks <= 2
    ).distinct().count()
    
    return StatisticsResponse(
        total_patients=total_patients,
        total_scans=total_scans,
        scans_today=scans_today,
        scans_this_week=scans_this_week,
        scans_this_month=scans_this_month,
        severity_distribution=severity_distribution,
        urgency_distribution=urgency_distribution,
        risk_level_distribution=risk_level_distribution,
        average_confidence=avg_confidence,
        high_risk_patients=high_risk_patients,
        patients_needing_followup=patients_needing_followup
    )