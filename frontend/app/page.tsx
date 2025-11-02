"use client";
import { useState, useRef, useEffect } from "react";
import axios from "axios";
import { Bar } from "react-chartjs-2";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
} from "chart.js";
import styles from "./page.module.css";
import PatientList from "./components/PatientList";
import PatientForm from "./components/PatientForm";
import ScanHistory from "./components/ScanHistory";
import Toast from "./components/Toast";
import { useNotification } from "./hooks/useNotification";

ChartJS.register(CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend);

const UploadIcon = () => (
  <svg width="48" height="48" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
    <path d="M12 15V3M12 3L8 7M12 3L16 7" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
    <path d="M2 17L2 19C2 20.1046 2.89543 21 4 21L20 21C21.1046 21 22 20.1046 22 19L22 17" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
  </svg>
);

const LoadingSpinner = () => (
  <svg className="animate-spin" width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
    <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" style={{opacity: 0.25}}/>
    <path d="m12 2a10 10 0 0 1 10 10h-4a6 6 0 0 0-6-6z" fill="currentColor"/>
  </svg>
);

const EyeIcon = () => (
  <svg width="32" height="32" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
    <path d="M12 5C7.64 5 4 8.64 4 12s3.64 7 8 7 8-3.64 8-7-3.64-7-8-7z" stroke="currentColor" strokeWidth="2"/>
    <path d="M12 9a3 3 0 1 0 0 6 3 3 0 0 0 0-6z" stroke="currentColor" strokeWidth="2"/>
  </svg>
);

const CheckIcon = () => (
  <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
    <path d="M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41L9 16.17z"/>
  </svg>
);

const AlertIcon = () => (
  <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
    <path d="M12 2L2 20h20L12 2zm0 3.5L19.53 19H4.47L12 5.5zM11 10v4h2v-4h-2zm0 6v2h2v-2h-2z"/>
  </svg>
);

const CalendarIcon = () => (
  <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
    <path d="M19 3h-1V1h-2v2H8V1H6v2H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zm0 16H5V8h14v11z"/>
  </svg>
);

const HeartIcon = () => (
  <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
    <path d="M12 21.35l-1.45-1.32C5.4 15.36 2 12.28 2 8.5 2 5.42 4.42 3 7.5 3c1.74 0 3.41.81 4.5 2.09C13.09 3.81 14.76 3 16.5 3 19.58 3 22 5.42 22 8.5c0 3.78-3.4 6.86-8.55 11.54L12 21.35z"/>
  </svg>
);

const InfoIcon = () => (
  <svg width="32" height="32" viewBox="0 0 24 24" fill="currentColor">
    <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-6h2v6zm0-8h-2V7h2v2z"/>
  </svg>
);

const getDescriptionForSeverity = (severity: number) => {
  const descriptions = [
    'No signs of diabetic retinopathy detected. The retina appears healthy with no visible damage from diabetes.',
    'Early signs of diabetic retinopathy are present. Small areas of balloon-like swelling in blood vessels (microaneurysms) may be visible.',
    'Moderate diabetic retinopathy shows more extensive blood vessel blockages. Some blood vessels that nourish the retina are blocked.',
    'Severe diabetic retinopathy exhibits many more blocked blood vessels, depriving several areas of the retina of their blood supply.',
    'Proliferative diabetic retinopathy - the most advanced stage. New, abnormal blood vessels grow in the retina and may bleed or cause scarring.'
  ];
  return descriptions[severity] || 'Unknown severity level';
};

const getFollowUpForSeverity = (severity: number) => {
  const followUps = [
    'Annual eye exam recommended to monitor for any changes.',
    'Follow-up in 6-12 months. Monitor blood sugar levels closely.',
    'Follow-up in 3-6 months. Consider laser treatment if progression occurs.',
    'Follow-up in 2-4 months. Laser treatment may be necessary to prevent vision loss.',
    'Immediate treatment required. Anti-VEGF injections and/or laser surgery recommended.'
  ];
  return followUps[severity] || 'Consult with ophthalmologist for appropriate follow-up';
};

const PhoneIcon = () => (
  <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
    <path d="M6.62 10.79c1.44 2.83 3.76 5.14 6.59 6.59l2.2-2.2c.27-.27.67-.36 1.02-.24 1.12.37 2.33.57 3.57.57.55 0 1 .45 1 1V20c0 .55-.45 1-1 1-9.39 0-17-7.61-17-17 0-.55.45-1 1-1h3.5c.55 0 1 .45 1 1 0 1.25.2 2.45.57 3.57.11.35.03.74-.25 1.02l-2.2 2.2z"/>
  </svg>
);

export default function Home() {
  const [image, setImage] = useState<File | null>(null);
  const [result, setResult] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState<'scan' | 'patients' | 'new-patient' | 'history'>('scan');
  const [selectedPatient, setSelectedPatient] = useState<any>(null);
  const [showPatientForm, setShowPatientForm] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Use the notification hook
  const { notification, showSuccess, showError, showInfo, hideNotification } = useNotification();

  const labels = [
    "No DR",
    "Mild DR",
    "Moderate DR",
    "Severe DR",
    "Proliferative DR",
  ];

  const handleFileClick = () => {
    fileInputRef.current?.click();
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0] || null;
    setImage(file);
  }

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  };

  const handleDragEnter = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);

    const files = e.dataTransfer.files;
    if (files && files.length > 0) {
      const file = files[0];
      // Check if it's an image file
      if (file.type.startsWith('image/')) {
        setImage(file);
      } else {
        alert('Please drop an image file (JPG, PNG, JPEG)');
      }
    }
  };

  const handleSubmit = async () => {
    if (!image) return;
    const formData = new FormData();
    formData.append("file", image);
    
    // Add patient_id if a patient is selected
    if (selectedPatient?.id) {
      formData.append("patient_id", selectedPatient.id.toString());
    }

    setLoading(true);
    try {
      const res = await axios.post("http://localhost:8000/predict", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      setResult(res.data);

      // If scan was saved to a patient, show a success message
      if (selectedPatient?.id && res.data.patient_id) {
        showSuccess(`Scan saved for patient ${selectedPatient.first_name} ${selectedPatient.last_name}`);
      }
    } catch (err) {
      console.error(err);
      showError('Failed to analyze image. Please try again.');
    }
    setLoading(false);
  };

  const handlePatientSubmit = async (patientData: any) => {
    try {
      const response = await axios.post("http://localhost:8000/api/patients", patientData);
      setSelectedPatient(response.data);
      setShowPatientForm(false);
      setActiveTab('scan');
      showSuccess(`Patient ${response.data.first_name} ${response.data.last_name} created successfully!`);
    } catch (err) {
      console.error(err);
      showError('Failed to create patient. Please try again.');
    }
  };

  // Helper function to get urgency class
  const getUrgencyClass = (urgency: string) => {
    if (urgency?.includes("EMERGENCY")) return styles.urgencyEmergency;
    if (urgency?.includes("URGENT")) return styles.urgencyUrgent;
    if (urgency?.includes("High")) return styles.urgencyHigh;
    if (urgency?.includes("Moderate")) return styles.urgencyModerate;
    return styles.urgencyRoutine;
  };

  // Helper function to get risk class
  const getRiskClass = (riskLevel: string) => {
    if (riskLevel?.includes("Critical")) return styles.riskCritical;
    if (riskLevel?.includes("High")) return styles.riskHigh;
    if (riskLevel?.includes("Moderate")) return styles.riskModerate;
    return styles.riskLow;
  };

  return (
    <div className={styles.container}>
      {/* Header */}
      <header className={styles.header}>
        <div className={styles.headerContent}>
          <div className={styles.headerBrand}>
            <div className={styles.iconContainer}>
              <EyeIcon />
            </div>
            <div>
              <h1 className={styles.headerTitle}>DR Detector</h1>
              <p className={styles.headerSubtitle}>Diabetic Retinopathy Analysis</p>
            </div>
          </div>
        </div>
      </header>

      <main className={styles.main}>
        {/* Navigation Tabs */}
        <div className={styles.tabContainer}>
          <button
            className={`${styles.tab} ${activeTab === 'scan' ? styles.activeTab : ''}`}
            onClick={() => setActiveTab('scan')}
          >
            Scan Image
          </button>
          <button
            className={`${styles.tab} ${activeTab === 'patients' ? styles.activeTab : ''}`}
            onClick={() => setActiveTab('patients')}
          >
            Patient List
          </button>
          <button
            className={`${styles.tab} ${activeTab === 'new-patient' ? styles.activeTab : ''}`}
            onClick={() => setActiveTab('new-patient')}
          >
            New Patient
          </button>
          {selectedPatient && (
            <button
              className={`${styles.tab} ${activeTab === 'history' ? styles.activeTab : ''}`}
              onClick={() => setActiveTab('history')}
            >
              History ({selectedPatient.first_name})
            </button>
          )}
        </div>

        {/* Selected Patient Banner */}
        {selectedPatient && activeTab === 'scan' && (
          <div className={styles.patientBanner}>
            <div className={styles.patientBannerContent}>
              <span className={styles.patientBannerLabel}>Selected Patient:</span>
              <span className={styles.patientBannerName}>
                {selectedPatient.first_name} {selectedPatient.last_name}
              </span>
              <span className={styles.patientBannerId}>({selectedPatient.patient_id})</span>
            </div>
            <button
              className={styles.clearPatientButton}
              onClick={() => {
                setSelectedPatient(null);
                setImage(null);
                setResult(null);
                showInfo('Patient selection cleared');
              }}
            >
              Clear Selection
            </button>
          </div>
        )}

        {/* Tab Content */}
        {activeTab === 'scan' && (
          <>
            {/* Hero Section */}
            <div className={styles.hero}>
          <h1 className={styles.heroTitle}>
            AI-Powered Diabetic Retinopathy Detection
          </h1>
          <p className={styles.heroSubtitle}>
            Upload a retinal fundus image to get instant AI analysis with confidence scores and visual explanations
          </p>
        </div>

        {/* Upload Card */}
        <div className={styles.uploadCard}>
          <div className={styles.uploadContent}>
            <div className={styles.uploadLayout}>
              {/* Left side - Upload area */}
              <div className={styles.uploadAreaWrapper}>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/*"
                  onChange={handleFileChange}
                  className={styles.hiddenInput}
                />

                <div
                  onClick={handleFileClick}
                  onDragOver={handleDragOver}
                  onDragEnter={handleDragEnter}
                  onDragLeave={handleDragLeave}
                  onDrop={handleDrop}
                  className={styles.dropzone}
                  style={{
                    borderColor: isDragging ? '#3b82f6' : undefined,
                    backgroundColor: isDragging ? 'rgba(59, 130, 246, 0.05)' : undefined,
                  }}
                >
                  <div className={styles.dropzoneContent}>
                    <div className={styles.uploadIcon}>
                      <UploadIcon />
                    </div>
                    <div>
                      <p className={styles.uploadText}>
                        {isDragging
                          ? "Drop image here"
                          : image
                          ? "Change Image"
                          : "Drop your retinal image here"}
                      </p>
                      <p className={styles.uploadSubtext}>
                        {isDragging
                          ? "Release to upload"
                          : image
                          ? image.name
                          : "or click to browse files"}
                      </p>
                    </div>
                    <p className={styles.uploadHint}>
                      Supports: JPG, PNG, JPEG • Max size: 10MB
                    </p>
                  </div>
                </div>
              </div>
              
              {/* Right side - Image preview */}
              <div className={styles.previewAreaWrapper}>
                {image ? (
                  <div className={styles.imagePreview}>
                    <img
                      src={URL.createObjectURL(image)}
                      alt="Selected retinal image"
                      className={styles.previewImage}
                    />
                    <p className={styles.previewLabel}>
                      Selected Image
                    </p>
                  </div>
                ) : (
                  <div className={styles.emptyPreview}>
                    <svg width="64" height="64" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                      <rect x="3" y="3" width="18" height="18" rx="2" stroke="#d1d5db" strokeWidth="2"/>
                      <circle cx="8.5" cy="8.5" r="1.5" fill="#d1d5db"/>
                      <path d="M21 15L16 10L11 15" stroke="#d1d5db" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                      <path d="M13 13L8 18" stroke="#d1d5db" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                    </svg>
                    <p className={styles.emptyPreviewText}>No image selected</p>
                  </div>
                )}
              </div>
            </div>
            
            <button
              onClick={handleSubmit}
              disabled={!image || loading}
              className={styles.analyzeButton}
            >
              {loading ? (
                <>
                  <LoadingSpinner />
                  <span>Analyzing Image...</span>
                </>
              ) : (
                <span>Analyze Retinal Image</span>
              )}
            </button>
          </div>
        </div>

        {/* Results Section */}
        {result && (
          <div className={styles.resultsContainer}>
            {/* Main Prediction Card */}
            <div className={styles.resultsCard}>
              <div className={styles.resultsHeader}>
                <h2 className={styles.resultsTitle}>
                  Analysis Results
                </h2>
                <div className={styles.resultsSummary}>
                  <div className={styles.summaryItem}>
                    <p className={styles.summaryLabel}>Prediction</p>
                    <p className={styles.predictionValue}>
                      {labels[result.prediction]}
                    </p>
                  </div>
                  <div className={styles.divider}></div>
                  <div className={styles.summaryItem}>
                    <p className={styles.summaryLabel}>Confidence</p>
                    <p className={styles.confidenceValue}>
                      {(result.confidence * 100).toFixed(1)}%
                    </p>
                  </div>
                </div>
              </div>
              
              {/* Probability Chart */}
              <div className={styles.chartContainer}>
                <h3 className={styles.chartTitle}>
                  Probability Distribution
                </h3>
                <Bar
                  data={{
                    labels,
                    datasets: [
                      {
                        label: "Probability (%)",
                        data: result.probabilities.map((p: number) => p * 100),
                        backgroundColor: [
                          "rgba(34, 197, 94, 0.8)",   // No DR - Green
                          "rgba(59, 130, 246, 0.8)",   // Mild - Blue  
                          "rgba(245, 158, 11, 0.8)",   // Moderate - Yellow
                          "rgba(249, 115, 22, 0.8)",   // Severe - Orange
                          "rgba(239, 68, 68, 0.8)",    // Proliferative - Red
                        ],
                        borderRadius: 8,
                        borderSkipped: false,
                        maxBarThickness: 60,
                      },
                    ],
                  }}
                  options={{
                    responsive: true,
                    plugins: {
                      legend: { display: false },
                      tooltip: {
                        backgroundColor: "rgba(17, 24, 39, 0.9)",
                        titleColor: "#fff",
                        bodyColor: "#e5e7eb",
                        cornerRadius: 8,
                        padding: 12,
                        displayColors: false,
                      },
                    },
                    scales: {
                      y: {
                        beginAtZero: true,
                        max: 100,
                        ticks: {
                          color: "#6b7280",
                          font: { size: 12 },
                          callback: (val) => val + "%",
                        },
                        grid: {
                          color: "rgba(107, 114, 128, 0.2)",
                        },
                      },
                      x: {
                        ticks: {
                          color: "#374151",
                          font: { size: 12, weight: 500 },
                        },
                        grid: { display: false },
                      },
                    },
                    animation: { duration: 1200, easing: "easeOutQuart" },
                  }}
                />
              </div>
            </div>
            
            {/* GradCAM Visualization */}
            {result.gradcam_base64 && (
              <div className={styles.gradcamCard}>
                <div className={styles.gradcamHeader}>
                  <h3 className={styles.gradcamTitle}>
                    AI Attention Heatmap
                  </h3>
                  <p className={styles.gradcamDescription}>
                    Areas highlighted in red show where the AI focused its attention during analysis
                  </p>
                </div>
                <div className={styles.gradcamContainer}>
                  <img
                    src={`data:image/png;base64,${result.gradcam_base64}`}
                    alt="GradCAM Visualization"
                    className={styles.gradcamImage}
                  />
                </div>
              </div>
            )}
          </div>
        )}

        {/* Medical Information Section */}
        {result && (
          <div className={styles.medicalInfoContainer}>
            {/* Urgency Banner */}
            {result.urgency && (
              <div className={`${styles.urgencyBanner} ${getUrgencyClass(result.urgency)}`}>
                {result.urgency}
              </div>
            )}

            {/* Current State Analysis */}
            <div className={styles.medicalCard}>
              <div className={styles.medicalCardHeader}>
                <h3 className={styles.medicalCardTitle}>
                  <EyeIcon /> Current Retinal Analysis
                </h3>
              </div>
              <div className={styles.medicalCardContent}>
                <div className={`${styles.riskIndicator} ${getRiskClass(result.risk_level)}`}>
                  Risk Level: {result.risk_level}
                </div>
                <p style={{marginTop: '1rem', marginBottom: '1rem', color: '#4b5563'}}>
                  {result.description}
                </p>
                <div className={styles.infoGrid}>
                  <div className={styles.infoItem}>
                    <div className={styles.infoLabel}>Vision Impact</div>
                    <div className={styles.infoValue}>{result.current_state?.vision_impact}</div>
                  </div>
                  <div className={styles.infoItem}>
                    <div className={styles.infoLabel}>Retinal Changes</div>
                    <div className={styles.infoValue}>{result.current_state?.retinal_changes}</div>
                  </div>
                  <div className={styles.infoItem}>
                    <div className={styles.infoLabel}>Blood Vessel Status</div>
                    <div className={styles.infoValue}>{result.current_state?.blood_vessel_status}</div>
                  </div>
                  <div className={styles.infoItem}>
                    <div className={styles.infoLabel}>Follow-up Required</div>
                    <div className={styles.infoValue}>{result.follow_up}</div>
                  </div>
                </div>
              </div>
            </div>

            {/* Recommendations */}
            <div className={styles.medicalCard}>
              <div className={styles.medicalCardHeader}>
                <h3 className={styles.medicalCardTitle}>
                  <CheckIcon /> Medical Recommendations
                </h3>
              </div>
              <div className={styles.medicalCardContent}>
                <ul className={styles.recommendationsList}>
                  {result.recommendations?.map((rec: string, idx: number) => (
                    <li key={idx} className={styles.recommendationItem}>
                      <span className={styles.recommendationIcon}>▸</span>
                      {rec}
                    </li>
                  ))}
                </ul>
              </div>
            </div>

            {/* Risk Factors & Statistics */}
            <div className={styles.medicalCard}>
              <div className={styles.medicalCardHeader}>
                <h3 className={styles.medicalCardTitle}>
                  <AlertIcon /> Risk Assessment & Statistics
                </h3>
              </div>
              <div className={styles.medicalCardContent}>
                <div className={styles.statisticsGrid}>
                  <div className={styles.statCard}>
                    <div className={styles.statValue}>
                      {(() => {
                        const match = result.risk_factors?.progression_risk?.match(/(\d+(?:-\d+)?)/);
                        return match ? `${match[0]}%` : 'High';
                      })()}
                    </div>
                    <div className={styles.statLabel}>Progression Risk</div>
                  </div>
                  <div className={styles.statCard}>
                    <div className={styles.statValue}>
                      {(() => {
                        const riskText = result.statistics?.vision_loss_risk;
                        if (riskText?.toLowerCase().includes('very high')) return 'Very High';
                        if (riskText?.toLowerCase().includes('high')) return 'High';
                        if (riskText?.toLowerCase().includes('moderate')) return 'Moderate';
                        if (riskText?.toLowerCase().includes('low')) return 'Low';
                        if (riskText?.toLowerCase().includes('very low')) return 'Very Low';
                        return 'Assessed';
                      })()}
                    </div>
                    <div className={styles.statLabel}>Vision Loss Risk</div>
                  </div>
                  <div className={styles.statCard}>
                    <div className={styles.statValue} style={{fontSize: result.follow_up?.length > 15 ? '0.9rem' : '1.5rem'}}>
                      {result.follow_up || 'Consult Doctor'}
                    </div>
                    <div className={styles.statLabel}>Next Check-up</div>
                  </div>
                </div>
                
                {result.risk_factors?.key_factors && (
                  <div style={{marginTop: '1.5rem'}}>
                    <h4 style={{marginBottom: '0.5rem', color: '#374151'}}>Key Risk Factors</h4>
                    <div className={styles.preventionGrid}>
                      {result.risk_factors.key_factors.map((factor: string, idx: number) => (
                        <div key={idx} className={styles.preventionItem}>
                          <span>•</span> {factor}
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </div>

            {/* Prevention Tips */}
            <div className={styles.medicalCard}>
              <div className={styles.medicalCardHeader}>
                <h3 className={styles.medicalCardTitle}>
                  <HeartIcon /> Prevention & Lifestyle Tips
                </h3>
              </div>
              <div className={styles.medicalCardContent}>
                <div className={styles.preventionGrid}>
                  {result.prevention_tips?.map((tip: string, idx: number) => (
                    <div key={idx} className={styles.preventionItem}>
                      <span>✓</span> {tip}
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* Warning Signs */}
            {result.general_advice?.warning_signs && (
              <div className={styles.medicalCard}>
                <div className={styles.medicalCardHeader}>
                  <h3 className={styles.medicalCardTitle}>
                    <AlertIcon /> Emergency Warning Signs
                  </h3>
                </div>
                <div className={styles.medicalCardContent}>
                  <div className={styles.warningSignsList}>
                    {result.general_advice.warning_signs.map((sign: string, idx: number) => (
                      <div key={idx} className={styles.warningItem}>
                        <span>⚠</span> {sign}
                      </div>
                    ))}
                  </div>
                  <p style={{marginTop: '1rem', fontSize: '0.875rem', color: '#991b1b'}}>
                    If you experience any of these symptoms, seek immediate medical attention.
                  </p>
                </div>
              </div>
            )}

            {/* Resources */}
            {result.resources && (
              <div className={styles.medicalCard}>
                <div className={styles.medicalCardHeader}>
                  <h3 className={styles.medicalCardTitle}>
                    <CalendarIcon /> New Zealand Resources & Support
                  </h3>
                </div>
                <div className={styles.medicalCardContent}>
                  <div className={styles.resourcesGrid}>
                    <div className={styles.resourceCard}>
                      <div className={styles.resourceTitle}>🇳🇿 NZ Health Hotlines</div>
                      {result.resources.hotlines?.map((hotline: string, idx: number) => (
                        <div key={idx} className={styles.resourceLink}>{hotline}</div>
                      ))}
                    </div>
                    <div className={styles.resourceCard}>
                      <div className={styles.resourceTitle}>🌐 NZ Health Websites</div>
                      {result.resources.websites?.map((website: string, idx: number) => (
                        <a key={idx} href={`https://${website}`} target="_blank" rel="noopener noreferrer" className={styles.resourceLink}>
                          {website}
                        </a>
                      ))}
                    </div>
                  </div>
                  
                  {result.resources.nz_services && (
                    <div style={{marginTop: '1.5rem'}}>
                      <h4 style={{marginBottom: '1rem', color: '#166534', fontWeight: 600}}>
                        Available NZ Health Services
                      </h4>
                      <div className={styles.preventionGrid}>
                        {result.resources.nz_services.map((service: string, idx: number) => (
                          <div key={idx} className={styles.preventionItem} style={{background: 'linear-gradient(135deg, #dcfce7, #bbf7d0)', color: '#166534'}}>
                            <span>✓</span> {service}
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
        )}

            {/* Medical Disclaimer Section */}
            <div className={styles.disclaimerSection}>
          <div className={styles.disclaimerHeader}>
            <div className={styles.disclaimerIcon}>
              <InfoIcon />
            </div>
            <h2 className={styles.disclaimerTitle}>Important Medical Disclaimer</h2>
          </div>
          
          <div className={styles.disclaimerContent}>
            <p>
              <strong>This is an AI-generated medical assessment report</strong> created using advanced machine learning algorithms 
              trained on diabetic retinopathy images. While our system achieves high accuracy rates, it is designed to be a 
              <strong> screening tool only</strong> and should not replace professional medical consultation.
            </p>
            
            <ul className={styles.disclaimerList}>
              <li>AI predictions may not detect all cases of diabetic retinopathy</li>
              <li>Results should be confirmed by a qualified eye care professional</li>
              <li>This tool does not diagnose other eye conditions that may be present</li>
              <li>Accuracy depends on image quality and proper imaging techniques</li>
              <li>Treatment decisions should only be made by healthcare providers</li>
            </ul>
            
            <p>
              <strong>Your next steps:</strong> Please share these results with your GP, optometrist, or ophthalmologist. 
              In New Zealand, you can access free annual diabetic eye screening through your DHB. 
              If you experience any sudden vision changes, seek immediate medical attention by calling 111.
            </p>
          </div>
          
          <div className={styles.disclaimerActions}>
            <a href="tel:111" className={styles.disclaimerButton}>
              <PhoneIcon />
              <span>Call 111 - Emergency</span>
            </a>
            <a href="tel:0800611116" className={styles.disclaimerButtonSecondary}>
              <PhoneIcon />
              <span>Healthline: 0800 611 116</span>
            </a>
            <button 
              onClick={() => window.print()} 
              className={styles.disclaimerButtonSecondary}
            >
              <span>📄 Print Report for Doctor</span>
            </button>
          </div>
        </div>
          </>
        )}

        {/* Patients Tab */}
        {activeTab === 'patients' && (
          <div className={styles.tabContent}>
            <PatientList
              onSelectPatient={(patient) => {
                setSelectedPatient(patient);
                setActiveTab('scan');
                // Clear previous scan data
                setImage(null);
                setResult(null);
                showSuccess(`Selected patient: ${patient.first_name} ${patient.last_name}`);
              }}
              selectedPatientId={selectedPatient?.id}
            />
          </div>
        )}

        {/* New Patient Tab */}
        {activeTab === 'new-patient' && (
          <div className={styles.tabContent}>
            <PatientForm 
              onSubmit={handlePatientSubmit}
              onSkip={() => setActiveTab('scan')}
            />
          </div>
        )}

        {/* History Tab */}
        {activeTab === 'history' && selectedPatient && (
          <div className={styles.tabContent}>
            <ScanHistory 
              patient={selectedPatient}
              onViewScan={async (scan) => {
                // Convert scan data to the format expected by the results display
                const fullResult = {
                  scan_id: scan.scan_id,
                  patient_id: selectedPatient.id,
                  prediction: scan.prediction_class,
                  label: scan.prediction_label,
                  confidence: scan.confidence_score,
                  probabilities: scan.probabilities || [0, 0, 0, 0, 0],
                  gradcam_base64: null,
                  
                  // Medical Information
                  severity: scan.severity,
                  description: getDescriptionForSeverity(scan.prediction_class),
                  risk_level: scan.risk_level,
                  
                  // Current State
                  current_state: scan.current_state || {},
                  
                  // Recommendations and Follow-up
                  recommendations: scan.recommendations || [],
                  follow_up: getFollowUpForSeverity(scan.prediction_class),
                  urgency: scan.urgency,
                  
                  // Risk Assessment
                  risk_factors: scan.risk_factors || {},
                  prevention_tips: scan.prevention_tips || [],
                  
                  // Statistics
                  statistics: scan.statistics || {},
                  
                  // General Advice
                  general_advice: {
                    lifestyle: [
                      "Maintain healthy weight (BMI < 25)",
                      "Exercise regularly (30 min/day)",
                      "Follow diabetic diet strictly",
                      "Monitor blood sugar daily",
                      "Take all medications as prescribed"
                    ],
                    monitoring: [
                      "Regular HbA1c tests (every 3 months)",
                      "Blood pressure checks",
                      "Cholesterol monitoring",
                      "Kidney function tests",
                      "Regular foot examinations"
                    ],
                    warning_signs: [
                      "Sudden vision changes or loss",
                      "Flashes of light in vision",
                      "Dark curtain over vision",
                      "Sudden increase in floaters",
                      "Eye pain or pressure"
                    ]
                  },
                  
                  // Confidence Note
                  confidence_note: scan.confidence_score < 0.7 
                    ? "Note: Confidence is moderate. Consider retaking image or getting second opinion."
                    : "AI analysis confidence is high.",
                  
                  // Resources
                  resources: {
                    hotlines: [
                      "Healthline NZ: 0800 611 116 (24/7 free health advice)",
                      "Diabetes NZ: 0800 DIABETES (0800 342 238)",
                      "Blind Low Vision NZ: 0800 24 33 33",
                      "Emergency: 111 (for urgent eye emergencies)"
                    ],
                    websites: [
                      "diabetes.org.nz",
                      "health.govt.nz/your-health/conditions-and-treatments/diseases-and-illnesses/diabetes",
                      "blindlowvision.org.nz",
                      "eyehealthaotearoa.org.nz",
                      "maculardegenerationnz.org.nz"
                    ],
                    nz_services: [
                      "Free annual diabetes checks through your GP",
                      "Retinal screening through local DHB services",
                      "Green prescription for exercise support",
                      "Community diabetes education programmes"
                    ]
                  }
                };

                // Clear uploaded image and set historical result
                setImage(null);
                setResult(fullResult);
                setActiveTab('scan');
                showInfo('Viewing historical scan');
              }}
            />
          </div>
        )}
      </main>

      {/* Toast Notification */}
      {notification && (
        <Toast notification={notification} onClose={hideNotification} />
      )}
    </div>
  );
}
