"""
Database models for the Diabetic Retinopathy Detection System
"""
from sqlalchemy import Column, Integer, String, Float, DateTime, Text, ForeignKey, Boolean, JSON
from sqlalchemy.orm import relationship
from datetime import datetime
from .database import Base

class Patient(Base):
    __tablename__ = "patients"
    
    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(String, unique=True, index=True)  # NHI or custom ID
    first_name = Column(String, nullable=False)
    last_name = Column(String, nullable=False)
    date_of_birth = Column(DateTime)
    email = Column(String, unique=True, index=True)
    phone = Column(String)
    
    # Medical Information
    diabetes_type = Column(String)  # Type 1, Type 2, Gestational, etc.
    diabetes_duration_years = Column(Float)  # Years since diagnosis
    hba1c_latest = Column(Float)  # Latest HbA1c value
    blood_pressure_systolic = Column(Integer)
    blood_pressure_diastolic = Column(Integer)
    
    # Risk Factors
    has_hypertension = Column(Boolean, default=False)
    has_kidney_disease = Column(Boolean, default=False)
    has_high_cholesterol = Column(Boolean, default=False)
    is_smoker = Column(Boolean, default=False)
    family_history_dr = Column(Boolean, default=False)
    
    # Medications
    medications = Column(JSON)  # List of current medications
    insulin_therapy = Column(Boolean, default=False)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    scans = relationship("RetinalScan", back_populates="patient", cascade="all, delete-orphan")
    
    @property
    def full_name(self):
        return f"{self.first_name} {self.last_name}"
    
    @property
    def age(self):
        if self.date_of_birth:
            return (datetime.now() - self.date_of_birth).days // 365
        return None


class RetinalScan(Base):
    __tablename__ = "retinal_scans"
    
    id = Column(Integer, primary_key=True, index=True)
    scan_id = Column(String, unique=True, index=True)  # Unique scan identifier
    patient_id = Column(Integer, ForeignKey("patients.id"))
    
    # Scan Information
    scan_date = Column(DateTime, default=datetime.utcnow)
    eye = Column(String)  # Left, Right, or Both
    image_path = Column(String)  # Path to stored image
    image_quality = Column(Float)  # Quality score if available
    
    # AI Prediction Results
    prediction_class = Column(Integer)  # 0-4 (DR severity)
    prediction_label = Column(String)  # Human-readable label
    confidence_score = Column(Float)  # Model confidence
    probabilities = Column(JSON)  # All class probabilities
    gradcam_path = Column(String)  # Path to GradCAM visualization
    
    # Medical Analysis
    severity = Column(String)
    risk_level = Column(String)
    urgency = Column(String)
    follow_up_weeks = Column(Integer)  # Recommended follow-up in weeks
    
    # Detailed Results
    current_state = Column(JSON)
    recommendations = Column(JSON)
    risk_factors = Column(JSON)
    prevention_tips = Column(JSON)
    statistics = Column(JSON)
    
    # Clinical Validation
    clinician_verified = Column(Boolean, default=False)
    clinician_diagnosis = Column(String)
    clinician_notes = Column(Text)
    verified_by = Column(String)  # Clinician name/ID
    verified_at = Column(DateTime)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    patient = relationship("Patient", back_populates="scans")
    
    @property
    def is_urgent(self):
        return self.urgency and ("URGENT" in self.urgency.upper() or "EMERGENCY" in self.urgency.upper())
    
    @property
    def days_since_scan(self):
        return (datetime.utcnow() - self.scan_date).days


class ScanComparison(Base):
    __tablename__ = "scan_comparisons"
    
    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id"))
    
    # Scans being compared
    baseline_scan_id = Column(Integer, ForeignKey("retinal_scans.id"))
    current_scan_id = Column(Integer, ForeignKey("retinal_scans.id"))
    
    # Progression Analysis
    progression_status = Column(String)  # Improved, Stable, Worsened
    severity_change = Column(Integer)  # Change in DR class
    confidence_change = Column(Float)
    
    # Clinical Notes
    comparison_notes = Column(Text)
    recommendations = Column(JSON)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    comparison_date = Column(DateTime, default=datetime.utcnow)