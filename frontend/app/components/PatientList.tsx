'use client';

import React, { useState, useEffect } from 'react';
import styles from './PatientList.module.css';

interface Patient {
  id: number;
  patient_id: string;
  first_name: string;
  last_name: string;
  email: string;
  phone: string;
  diabetes_type: string;
  diabetes_duration_years: number;
  hba1c_latest: number;
  created_at: string;
}

interface PatientListProps {
  onSelectPatient: (patient: Patient) => void;
  selectedPatientId?: number;
}

export default function PatientList({ onSelectPatient, selectedPatientId }: PatientListProps) {
  const [patients, setPatients] = useState<Patient[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [searchTerm, setSearchTerm] = useState('');

  useEffect(() => {
    fetchPatients();
  }, []);

  const fetchPatients = async () => {
    try {
      const response = await fetch('http://localhost:8000/api/patients');
      if (!response.ok) throw new Error('Failed to fetch patients');
      const data = await response.json();
      setPatients(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load patients');
    } finally {
      setLoading(false);
    }
  };

  const filteredPatients = patients.filter(patient => {
    const searchLower = searchTerm.toLowerCase();
    return (
      patient.first_name.toLowerCase().includes(searchLower) ||
      patient.last_name.toLowerCase().includes(searchLower) ||
      patient.email.toLowerCase().includes(searchLower) ||
      patient.patient_id.toLowerCase().includes(searchLower)
    );
  });

  if (loading) {
    return (
      <div className={styles.container}>
        <div className={styles.loading}>Loading patients...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className={styles.container}>
        <div className={styles.error}>
          <p>Error: {error}</p>
          <button onClick={fetchPatients} className={styles.retryButton}>
            Retry
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className={styles.container}>
      <div className={styles.header}>
        <h2>Patient Records</h2>
        <div className={styles.searchBox}>
          <input
            type="text"
            placeholder="Search patients..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className={styles.searchInput}
          />
        </div>
      </div>

      {filteredPatients.length === 0 ? (
        <div className={styles.emptyState}>
          <p>No patients found</p>
          <p className={styles.emptyHint}>
            {searchTerm ? 'Try a different search term' : 'Add a new patient to get started'}
          </p>
        </div>
      ) : (
        <div className={styles.patientGrid}>
          {filteredPatients.map((patient) => (
            <div
              key={patient.id}
              className={`${styles.patientCard} ${
                selectedPatientId === patient.id ? styles.selected : ''
              }`}
              onClick={() => onSelectPatient(patient)}
            >
              <div className={styles.patientHeader}>
                <h3>{patient.first_name} {patient.last_name}</h3>
                <span className={styles.patientId}>{patient.patient_id}</span>
              </div>
              
              <div className={styles.patientInfo}>
                <div className={styles.infoRow}>
                  <span className={styles.label}>Email:</span>
                  <span className={styles.value}>{patient.email}</span>
                </div>
                <div className={styles.infoRow}>
                  <span className={styles.label}>Phone:</span>
                  <span className={styles.value}>{patient.phone || 'N/A'}</span>
                </div>
                <div className={styles.infoRow}>
                  <span className={styles.label}>Diabetes Type:</span>
                  <span className={styles.value}>{patient.diabetes_type || 'N/A'}</span>
                </div>
                <div className={styles.infoRow}>
                  <span className={styles.label}>Duration:</span>
                  <span className={styles.value}>
                    {patient.diabetes_duration_years ? `${patient.diabetes_duration_years} years` : 'N/A'}
                  </span>
                </div>
                <div className={styles.infoRow}>
                  <span className={styles.label}>HbA1c:</span>
                  <span className={styles.value}>
                    {patient.hba1c_latest ? `${patient.hba1c_latest}%` : 'N/A'}
                  </span>
                </div>
              </div>

              <div className={styles.cardActions}>
                <button 
                  className={styles.selectButton}
                  onClick={(e) => {
                    e.stopPropagation();
                    onSelectPatient(patient);
                  }}
                >
                  {selectedPatientId === patient.id ? 'Selected' : 'Select Patient'}
                </button>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}