"""
Pydantic schemas for request/response validation
"""
from pydantic import BaseModel, EmailStr, field_validator
from typing import Optional, List, Dict, Any
from datetime import datetime

# Patient Schemas
class PatientBase(BaseModel):
    patient_id: Optional[str] = None
    first_name: str
    last_name: str
    date_of_birth: Optional[datetime] = None
    email: EmailStr
    phone: Optional[str] = None
    
    # Medical Information
    diabetes_type: Optional[str] = None
    diabetes_duration_years: Optional[float] = None
    hba1c_latest: Optional[float] = None
    blood_pressure_systolic: Optional[int] = None
    blood_pressure_diastolic: Optional[int] = None
    
    # Risk Factors
    has_hypertension: bool = False
    has_kidney_disease: bool = False
    has_high_cholesterol: bool = False
    is_smoker: bool = False
    family_history_dr: bool = False
    
    # Medications
    medications: Optional[List[str]] = []
    insulin_therapy: bool = False

    @field_validator('phone', 'patient_id', 'diabetes_type', mode='before')
    @classmethod
    def empty_str_to_none(cls, v):
        """Convert empty strings to None for optional string fields"""
        if v == '' or v is None:
            return None
        return v

    @field_validator('date_of_birth', mode='before')
    @classmethod
    def empty_date_to_none(cls, v):
        """Convert empty strings to None for date fields"""
        if v == '' or v is None:
            return None
        return v

class PatientCreate(PatientBase):
    pass

class PatientUpdate(PatientBase):
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    email: Optional[EmailStr] = None

class PatientResponse(PatientBase):
    id: int
    created_at: datetime
    updated_at: datetime
    full_name: str
    age: Optional[int] = None
    
    class Config:
        from_attributes = True

# Retinal Scan Schemas
class RetinalScanBase(BaseModel):
    eye: Optional[str] = "Both"
    image_quality: Optional[float] = None
    
    # These will be filled by the prediction
    prediction_class: Optional[int] = None
    prediction_label: Optional[str] = None
    confidence_score: Optional[float] = None
    probabilities: Optional[List[float]] = None

class RetinalScanCreate(RetinalScanBase):
    patient_id: int

class RetinalScanResponse(RetinalScanBase):
    id: int
    scan_id: str
    patient_id: int
    scan_date: datetime
    
    # Medical Analysis
    severity: Optional[str] = None
    risk_level: Optional[str] = None
    urgency: Optional[str] = None
    follow_up_weeks: Optional[int] = None
    
    # Detailed Results
    current_state: Optional[Dict[str, Any]] = None
    recommendations: Optional[List[str]] = None
    risk_factors: Optional[Dict[str, Any]] = None
    prevention_tips: Optional[List[str]] = None
    statistics: Optional[Dict[str, Any]] = None
    
    # Clinical Validation
    clinician_verified: bool = False
    clinician_diagnosis: Optional[str] = None
    clinician_notes: Optional[str] = None
    
    # Computed properties
    is_urgent: bool
    days_since_scan: int
    
    class Config:
        from_attributes = True

# Scan History Response
class ScanHistoryResponse(BaseModel):
    patient: PatientResponse
    scans: List[RetinalScanResponse]
    total_scans: int
    latest_scan: Optional[RetinalScanResponse] = None
    progression_trend: Optional[str] = None  # Improving, Stable, Worsening

# Prediction Request/Response
class PredictionRequest(BaseModel):
    patient_id: Optional[int] = None
    eye: Optional[str] = "Both"

class PredictionResponse(BaseModel):
    scan_id: str
    patient_id: Optional[int] = None
    prediction: int
    label: str
    confidence: float
    probabilities: List[float]
    gradcam_base64: Optional[str] = None
    
    # Medical Information
    severity: str
    description: str
    risk_level: str
    current_state: Dict[str, Any]
    recommendations: List[str]
    follow_up: str
    urgency: str
    risk_factors: Dict[str, Any]
    prevention_tips: List[str]
    statistics: Dict[str, Any]
    general_advice: Dict[str, Any]
    confidence_note: str
    resources: Dict[str, Any]

# Comparison Schemas
class ScanComparisonCreate(BaseModel):
    patient_id: int
    baseline_scan_id: int
    current_scan_id: int

class ScanComparisonResponse(BaseModel):
    id: int
    patient_id: int
    baseline_scan_id: int
    current_scan_id: int
    progression_status: str
    severity_change: int
    confidence_change: float
    comparison_notes: Optional[str] = None
    recommendations: Optional[List[str]] = None
    created_at: datetime
    
    class Config:
        from_attributes = True

# Simple Scan Response for History
class ScanHistoryItem(BaseModel):
    id: int
    scan_id: str
    scan_date: str  # Will be converted to string
    prediction_class: int
    prediction_label: str
    confidence_score: float
    severity: Optional[str] = None
    risk_level: Optional[str] = None
    urgency: Optional[str] = None
    recommendations: Optional[List[Any]] = []
    current_state: Optional[Dict[str, Any]] = {}
    risk_factors: Optional[Dict[str, Any]] = {}
    statistics: Optional[Dict[str, Any]] = {}
    
    class Config:
        from_attributes = True

# Statistics Response
class StatisticsResponse(BaseModel):
    total_patients: int
    total_scans: int
    scans_today: int
    scans_this_week: int
    scans_this_month: int
    
    severity_distribution: Dict[str, int]
    urgency_distribution: Dict[str, int]
    risk_level_distribution: Dict[str, int]
    
    average_confidence: float
    high_risk_patients: int
    patients_needing_followup: int