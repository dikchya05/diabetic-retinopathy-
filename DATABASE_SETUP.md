# Database Setup Guide

## Overview
The Diabetic Retinopathy Detection System now includes PostgreSQL database integration for storing patient information and scan history.

## Database Features

### 1. Patient Management
- Store patient demographics and medical history
- Track diabetes-related information (HbA1c, blood pressure, medications)
- Monitor risk factors for diabetic retinopathy progression

### 2. Scan History
- Save all retinal scan results with timestamps
- Track progression over time
- Store AI predictions and confidence scores
- Clinical validation tracking

### 3. Data Relationships
- One-to-many relationship between patients and scans
- Scan comparison tracking for progression analysis
- Comprehensive medical history per patient

## Setup Instructions

### Option 1: PostgreSQL (Production)

1. **Install PostgreSQL**
   ```bash
   # Windows: Download from https://www.postgresql.org/download/windows/
   # Mac: brew install postgresql
   # Linux: sudo apt-get install postgresql
   ```

2. **Create Database**
   ```sql
   CREATE DATABASE diabetic_retinopathy;
   CREATE USER dr_user WITH PASSWORD 'your_password';
   GRANT ALL PRIVILEGES ON DATABASE diabetic_retinopathy TO dr_user;
   ```

3. **Configure Environment**
   Create `.env` file in backend directory:
   ```
   DATABASE_URL=postgresql://dr_user:your_password@localhost:5432/diabetic_retinopathy
   ```

### Option 2: SQLite (Development)

The system defaults to SQLite if no PostgreSQL is configured:
- Database file: `backend/diabetic_retinopathy.db`
- No additional setup required
- Automatically created on first run

## Database Schema

### Patients Table
- Personal information (name, email, phone, DOB)
- Medical history (diabetes type, duration, HbA1c)
- Risk factors (hypertension, kidney disease, smoking)
- Medications and treatments

### Retinal Scans Table
- Scan metadata (date, eye, quality)
- AI predictions (class, confidence, probabilities)
- Medical analysis (severity, urgency, recommendations)
- Clinical validation fields

### Scan Comparisons Table
- Track progression between scans
- Calculate severity changes
- Store comparison notes

## API Endpoints

### Patient Management
- `POST /api/patients` - Create new patient
- `GET /api/patients` - List all patients
- `GET /api/patients/{id}` - Get patient details
- `PUT /api/patients/{id}` - Update patient
- `DELETE /api/patients/{id}` - Delete patient

### Scan Management
- `POST /predict` - Analyze image (with optional patient_id)
- `GET /api/patients/{id}/scans` - Get patient's scan history
- `GET /api/patients/{id}/history` - Full history with progression
- `POST /api/scans/{id}/verify` - Clinical verification

### Statistics
- `GET /api/statistics` - System-wide statistics

## Usage Examples

### Creating a Patient
```javascript
const response = await fetch('http://localhost:8000/api/patients', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    first_name: 'John',
    last_name: 'Doe',
    email: 'john.doe@example.com',
    diabetes_type: 'Type 2',
    diabetes_duration_years: 5,
    hba1c_latest: 7.2
  })
});
```

### Analyzing with Patient ID
```javascript
const formData = new FormData();
formData.append('file', imageFile);
formData.append('patient_id', '123');
formData.append('eye', 'Right');

const response = await fetch('http://localhost:8000/predict', {
  method: 'POST',
  body: formData
});
```

## Data Privacy & Security

- Patient data is stored locally
- No data is sent to external services
- Implement appropriate access controls for production
- Regular backups recommended
- HIPAA compliance considerations for healthcare settings

## Troubleshooting

### Database Connection Issues
1. Check PostgreSQL is running: `pg_isready`
2. Verify credentials in `.env` file
3. Check database exists: `psql -U postgres -l`

### Migration Issues
If tables aren't created:
```python
from app.database import engine, Base
Base.metadata.create_all(bind=engine)
```

### Performance Optimization
- Add indexes for frequently queried fields
- Use connection pooling for production
- Regular VACUUM for PostgreSQL
- Consider partitioning for large datasets

## Future Enhancements

- [ ] Data encryption at rest
- [ ] Audit logging
- [ ] Automated backups
- [ ] Multi-tenancy support
- [ ] FHIR/HL7 integration
- [ ] Cloud database support (AWS RDS, Azure Database)