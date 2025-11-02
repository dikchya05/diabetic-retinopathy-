'use client';

import React, { useState } from 'react';
import styles from './PatientForm.module.css';

interface PatientData {
  first_name: string;
  last_name: string;
  email: string;
  phone: string;
  date_of_birth: string;
  diabetes_type: string;
  diabetes_duration_years: number;
  hba1c_latest: number;
  blood_pressure_systolic: number;
  blood_pressure_diastolic: number;
  has_hypertension: boolean;
  has_kidney_disease: boolean;
  has_high_cholesterol: boolean;
  is_smoker: boolean;
  family_history_dr: boolean;
  medications: string[];
  insulin_therapy: boolean;
}

interface PatientSubmitData {
  first_name: string;
  last_name: string;
  email: string;
  phone: string | null;
  date_of_birth: string | null;
  diabetes_type: string | null;
  diabetes_duration_years: number | null;
  hba1c_latest: number | null;
  blood_pressure_systolic: number | null;
  blood_pressure_diastolic: number | null;
  has_hypertension: boolean;
  has_kidney_disease: boolean;
  has_high_cholesterol: boolean;
  is_smoker: boolean;
  family_history_dr: boolean;
  medications: string[];
  insulin_therapy: boolean;
}

interface PatientFormProps {
  onSubmit: (patient: PatientSubmitData) => void;
  onSkip?: () => void;
  isLoading?: boolean;
}

export default function PatientForm({ onSubmit, onSkip, isLoading }: PatientFormProps) {
  const [formData, setFormData] = useState<PatientData>({
    first_name: '',
    last_name: '',
    email: '',
    phone: '',
    date_of_birth: '',
    diabetes_type: 'Type 2',
    diabetes_duration_years: 0,
    hba1c_latest: 0,
    blood_pressure_systolic: 120,
    blood_pressure_diastolic: 80,
    has_hypertension: false,
    has_kidney_disease: false,
    has_high_cholesterol: false,
    is_smoker: false,
    family_history_dr: false,
    medications: [],
    insulin_therapy: false,
  });

  const [medicationInput, setMedicationInput] = useState('');

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value, type } = e.target;

    if (type === 'checkbox') {
      const checked = (e.target as HTMLInputElement).checked;
      setFormData(prev => ({ ...prev, [name]: checked }));
    } else if (type === 'number') {
      setFormData(prev => ({ ...prev, [name]: parseFloat(value) || 0 }));
    } else {
      setFormData(prev => ({ ...prev, [name]: value }));
    }
  };

  const handleAddMedication = () => {
    if (medicationInput.trim()) {
      setFormData(prev => ({
        ...prev,
        medications: [...prev.medications, medicationInput.trim()]
      }));
      setMedicationInput('');
    }
  };

  const handleRemoveMedication = (index: number) => {
    setFormData(prev => ({
      ...prev,
      medications: prev.medications.filter((_, i) => i !== index)
    }));
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();

    // Clean up the data - convert empty strings to null for optional fields
    const cleanedData: PatientSubmitData = {
      ...formData,
      phone: formData.phone || null,
      date_of_birth: formData.date_of_birth || null,
      diabetes_type: formData.diabetes_type || null,
      diabetes_duration_years: formData.diabetes_duration_years || null,
      hba1c_latest: formData.hba1c_latest || null,
      blood_pressure_systolic: formData.blood_pressure_systolic || null,
      blood_pressure_diastolic: formData.blood_pressure_diastolic || null,
    };

    onSubmit(cleanedData);
  };

  return (
    <div className={styles.formContainer}>
      <form onSubmit={handleSubmit} className={styles.form}>
        <div className={styles.formHeader}>
          <h2>Patient Information</h2>
          <p>Provide patient details for better analysis and tracking</p>
        </div>

        <div className={styles.formSection}>
          <h3>Personal Information</h3>
          <div className={styles.formGrid}>
            <div className={styles.formGroup}>
              <label htmlFor="first_name">First Name *</label>
              <input
                type="text"
                id="first_name"
                name="first_name"
                value={formData.first_name}
                onChange={handleChange}
                required
                className={styles.input}
              />
            </div>

            <div className={styles.formGroup}>
              <label htmlFor="last_name">Last Name *</label>
              <input
                type="text"
                id="last_name"
                name="last_name"
                value={formData.last_name}
                onChange={handleChange}
                required
                className={styles.input}
              />
            </div>

            <div className={styles.formGroup}>
              <label htmlFor="email">Email *</label>
              <input
                type="email"
                id="email"
                name="email"
                value={formData.email}
                onChange={handleChange}
                required
                className={styles.input}
              />
            </div>

            <div className={styles.formGroup}>
              <label htmlFor="phone">Phone</label>
              <input
                type="tel"
                id="phone"
                name="phone"
                value={formData.phone}
                onChange={handleChange}
                className={styles.input}
              />
            </div>

            <div className={styles.formGroup}>
              <label htmlFor="date_of_birth">Date of Birth</label>
              <input
                type="date"
                id="date_of_birth"
                name="date_of_birth"
                value={formData.date_of_birth}
                onChange={handleChange}
                className={styles.input}
              />
            </div>
          </div>
        </div>

        <div className={styles.formSection}>
          <h3>Medical History</h3>
          <div className={styles.formGrid}>
            <div className={styles.formGroup}>
              <label htmlFor="diabetes_type">Diabetes Type</label>
              <select
                id="diabetes_type"
                name="diabetes_type"
                value={formData.diabetes_type}
                onChange={handleChange}
                className={styles.select}
              >
                <option value="Type 1">Type 1</option>
                <option value="Type 2">Type 2</option>
                <option value="Gestational">Gestational</option>
                <option value="Other">Other</option>
              </select>
            </div>

            <div className={styles.formGroup}>
              <label htmlFor="diabetes_duration_years">Years Since Diagnosis</label>
              <input
                type="number"
                id="diabetes_duration_years"
                name="diabetes_duration_years"
                value={formData.diabetes_duration_years}
                onChange={handleChange}
                min="0"
                step="0.5"
                className={styles.input}
              />
            </div>

            <div className={styles.formGroup}>
              <label htmlFor="hba1c_latest">Latest HbA1c (%)</label>
              <input
                type="number"
                id="hba1c_latest"
                name="hba1c_latest"
                value={formData.hba1c_latest}
                onChange={handleChange}
                min="0"
                max="20"
                step="0.1"
                className={styles.input}
              />
            </div>

            <div className={styles.formGroup}>
              <label htmlFor="blood_pressure_systolic">Blood Pressure (Systolic)</label>
              <input
                type="number"
                id="blood_pressure_systolic"
                name="blood_pressure_systolic"
                value={formData.blood_pressure_systolic}
                onChange={handleChange}
                min="60"
                max="250"
                className={styles.input}
              />
            </div>

            <div className={styles.formGroup}>
              <label htmlFor="blood_pressure_diastolic">Blood Pressure (Diastolic)</label>
              <input
                type="number"
                id="blood_pressure_diastolic"
                name="blood_pressure_diastolic"
                value={formData.blood_pressure_diastolic}
                onChange={handleChange}
                min="40"
                max="150"
                className={styles.input}
              />
            </div>
          </div>
        </div>

        <div className={styles.formSection}>
          <h3>Risk Factors</h3>
          <div className={styles.checkboxGrid}>
            <label className={styles.checkbox}>
              <input
                type="checkbox"
                name="has_hypertension"
                checked={formData.has_hypertension}
                onChange={handleChange}
              />
              <span>Hypertension</span>
            </label>

            <label className={styles.checkbox}>
              <input
                type="checkbox"
                name="has_kidney_disease"
                checked={formData.has_kidney_disease}
                onChange={handleChange}
              />
              <span>Kidney Disease</span>
            </label>

            <label className={styles.checkbox}>
              <input
                type="checkbox"
                name="has_high_cholesterol"
                checked={formData.has_high_cholesterol}
                onChange={handleChange}
              />
              <span>High Cholesterol</span>
            </label>

            <label className={styles.checkbox}>
              <input
                type="checkbox"
                name="is_smoker"
                checked={formData.is_smoker}
                onChange={handleChange}
              />
              <span>Smoker</span>
            </label>

            <label className={styles.checkbox}>
              <input
                type="checkbox"
                name="family_history_dr"
                checked={formData.family_history_dr}
                onChange={handleChange}
              />
              <span>Family History of DR</span>
            </label>

            <label className={styles.checkbox}>
              <input
                type="checkbox"
                name="insulin_therapy"
                checked={formData.insulin_therapy}
                onChange={handleChange}
              />
              <span>Insulin Therapy</span>
            </label>
          </div>
        </div>

        <div className={styles.formSection}>
          <h3>Current Medications</h3>
          <div className={styles.medicationInput}>
            <input
              type="text"
              value={medicationInput}
              onChange={(e) => setMedicationInput(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && (e.preventDefault(), handleAddMedication())}
              placeholder="Enter medication name"
              className={styles.input}
            />
            <button type="button" onClick={handleAddMedication} className={styles.addButton}>
              Add
            </button>
          </div>

          {formData.medications.length > 0 && (
            <div className={styles.medicationList}>
              {formData.medications.map((med, index) => (
                <div key={index} className={styles.medicationItem}>
                  <span>{med}</span>
                  <button
                    type="button"
                    onClick={() => handleRemoveMedication(index)}
                    className={styles.removeButton}
                  >
                    ×
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>

        <div className={styles.formActions}>
          {onSkip && (
            <button
              type="button"
              onClick={onSkip}
              className={styles.skipButton}
              disabled={isLoading}
            >
              Skip for Now
            </button>
          )}
          <button
            type="submit"
            className={styles.submitButton}
            disabled={isLoading || !formData.first_name || !formData.last_name || !formData.email}
          >
            {isLoading ? 'Saving...' : 'Save Patient'}
          </button>
        </div>
      </form>
    </div>
  );
}