'use client';

import React, { useState, useEffect } from 'react';
import { Bar, Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';
import styles from './ScanHistory.module.css';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend
);

interface Scan {
  id: number;
  scan_id: string;
  scan_date: string;
  prediction_class: number;
  prediction_label: string;
  confidence_score: number;
  severity: string;
  risk_level: string;
  urgency: string;
  recommendations: string[];
  current_state: any;
  risk_factors: any;
  statistics: any;
  probabilities?: number[];
  prevention_tips?: string[];
  eye?: string;
}

interface Patient {
  id: number;
  first_name: string;
  last_name: string;
  patient_id: string;
}

interface ScanHistoryProps {
  patient: Patient;
  onViewScan?: (scan: Scan) => void;
}

export default function ScanHistory({ patient, onViewScan }: ScanHistoryProps) {
  const [scans, setScans] = useState<Scan[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedScan, setSelectedScan] = useState<Scan | null>(null);
  const [viewMode, setViewMode] = useState<'timeline' | 'grid'>('timeline');

  useEffect(() => {
    if (patient?.id) {
      fetchScanHistory();
    }
  }, [patient]);

  const fetchScanHistory = async () => {
    try {
      setLoading(true);
      const response = await fetch(`http://localhost:8000/api/patients/${patient.id}/scans`);
      if (!response.ok) throw new Error('Failed to fetch scan history');
      const data = await response.json();
      setScans(data);
      if (data.length > 0) {
        setSelectedScan(data[0]);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load scan history');
    } finally {
      setLoading(false);
    }
  };

  const getSeverityColor = (severity: number) => {
    const colors = ['#10b981', '#3b82f6', '#f59e0b', '#f97316', '#ef4444'];
    return colors[severity] || '#6b7280';
  };

  const getSeverityLabel = (severity: number) => {
    const labels = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative'];
    return labels[severity] || 'Unknown';
  };

  const getUrgencyClass = (urgency: string) => {
    if (urgency?.includes('EMERGENCY')) return styles.urgencyEmergency;
    if (urgency?.includes('URGENT')) return styles.urgencyUrgent;
    if (urgency?.includes('High')) return styles.urgencyHigh;
    if (urgency?.includes('Moderate')) return styles.urgencyModerate;
    return styles.urgencyRoutine;
  };

  const getRiskClass = (riskLevel: string) => {
    if (riskLevel?.includes('Critical')) return styles.riskCritical;
    if (riskLevel?.includes('High')) return styles.riskHigh;
    if (riskLevel?.includes('Moderate')) return styles.riskModerate;
    return styles.riskLow;
  };

  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    return date.toLocaleDateString('en-NZ', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  // Prepare progression chart data
  const progressionData = {
    labels: scans.map(scan => new Date(scan.scan_date).toLocaleDateString('en-NZ', { month: 'short', day: 'numeric' })).reverse(),
    datasets: [
      {
        label: 'DR Severity Level',
        data: scans.map(scan => scan.prediction_class).reverse(),
        borderColor: '#3b82f6',
        backgroundColor: 'rgba(59, 130, 246, 0.1)',
        tension: 0.1,
        fill: true,
        pointRadius: 6,
        pointHoverRadius: 8,
        pointBackgroundColor: scans.map(scan => getSeverityColor(scan.prediction_class)).reverse(),
      },
    ],
  };

  const progressionOptions = {
    responsive: true,
    plugins: {
      legend: {
        display: false,
      },
      title: {
        display: true,
        text: 'DR Progression Over Time',
      },
      tooltip: {
        callbacks: {
          label: (context: any) => {
            const scan = scans[scans.length - 1 - context.dataIndex];
            return [
              `Severity: ${getSeverityLabel(scan.prediction_class)}`,
              `Confidence: ${(scan.confidence_score * 100).toFixed(1)}%`,
              `Risk: ${scan.risk_level}`,
            ];
          },
        },
      },
    },
    scales: {
      y: {
        min: 0,
        max: 4,
        ticks: {
          stepSize: 1,
          callback: (value: any) => getSeverityLabel(value),
        },
      },
    },
  };

  if (loading) {
    return <div className={styles.loading}>Loading scan history...</div>;
  }

  if (error) {
    return <div className={styles.error}>Error: {error}</div>;
  }

  if (scans.length === 0) {
    return (
      <div className={styles.emptyState}>
        <h3>No Scan History</h3>
        <p>No retinal scans found for {patient.first_name} {patient.last_name}</p>
      </div>
    );
  }

  return (
    <div className={styles.container}>
      <div className={styles.header}>
        <div>
          <h2>Scan History</h2>
          <p className={styles.subtitle}>
            {patient.first_name} {patient.last_name} - {scans.length} scan{scans.length !== 1 ? 's' : ''} on record
          </p>
        </div>
        <div className={styles.viewToggle}>
          <button
            className={viewMode === 'timeline' ? styles.activeView : ''}
            onClick={() => setViewMode('timeline')}
          >
            Timeline
          </button>
          <button
            className={viewMode === 'grid' ? styles.activeView : ''}
            onClick={() => setViewMode('grid')}
          >
            Grid
          </button>
        </div>
      </div>

      {/* Progression Chart */}
      {scans.length > 1 && (
        <div className={styles.progressionChart}>
          <Line data={progressionData} options={progressionOptions} />
        </div>
      )}

      {/* Timeline View */}
      {viewMode === 'timeline' && (
        <div className={styles.timeline}>
          {scans.map((scan, index) => (
            <div key={scan.id} className={styles.timelineItem}>
              <div className={styles.timelineMarker}>
                <div 
                  className={styles.timelineDot}
                  style={{ backgroundColor: getSeverityColor(scan.prediction_class) }}
                />
                {index < scans.length - 1 && <div className={styles.timelineLine} />}
              </div>
              
              <div 
                className={`${styles.timelineContent} ${selectedScan?.id === scan.id ? styles.selected : ''}`}
                onClick={() => setSelectedScan(scan)}
              >
                <div className={styles.timelineHeader}>
                  <div>
                    <h3 className={styles.scanDate}>{formatDate(scan.scan_date)}</h3>
                    <span className={styles.scanId}>Scan ID: {scan.scan_id}</span>
                  </div>
                  <div className={styles.severityBadge} style={{ backgroundColor: getSeverityColor(scan.prediction_class) }}>
                    {getSeverityLabel(scan.prediction_class)}
                  </div>
                </div>
                
                <div className={styles.scanSummary}>
                  <div className={styles.summaryItem}>
                    <span className={styles.label}>Confidence:</span>
                    <span className={styles.value}>{(scan.confidence_score * 100).toFixed(1)}%</span>
                  </div>
                  <div className={styles.summaryItem}>
                    <span className={styles.label}>Risk Level:</span>
                    <span className={`${styles.value} ${getRiskClass(scan.risk_level)}`}>{scan.risk_level}</span>
                  </div>
                  <div className={styles.summaryItem}>
                    <span className={styles.label}>Urgency:</span>
                    <span className={`${styles.value} ${getUrgencyClass(scan.urgency)}`}>{scan.urgency}</span>
                  </div>
                </div>

                {selectedScan?.id === scan.id && (
                  <div className={styles.expandedContent}>
                    <div className={styles.section}>
                      <h4>Current State</h4>
                      <div className={styles.stateGrid}>
                        <div>
                          <span className={styles.stateLabel}>Vision Impact:</span>
                          <p>{scan.current_state?.vision_impact}</p>
                        </div>
                        <div>
                          <span className={styles.stateLabel}>Retinal Changes:</span>
                          <p>{scan.current_state?.retinal_changes}</p>
                        </div>
                        <div>
                          <span className={styles.stateLabel}>Blood Vessel Status:</span>
                          <p>{scan.current_state?.blood_vessel_status}</p>
                        </div>
                      </div>
                    </div>

                    <div className={styles.section}>
                      <h4>Recommendations</h4>
                      <ul className={styles.recommendationList}>
                        {scan.recommendations?.map((rec, idx) => (
                          <li key={idx}>{rec}</li>
                        ))}
                      </ul>
                    </div>

                    {scan.risk_factors && (
                      <div className={styles.section}>
                        <h4>Risk Factors</h4>
                        <p className={styles.progressionRisk}>
                          {scan.risk_factors.progression_risk}
                        </p>
                        <ul className={styles.riskList}>
                          {scan.risk_factors.key_factors?.map((factor: string, idx: number) => (
                            <li key={idx}>{factor}</li>
                          ))}
                        </ul>
                      </div>
                    )}

                    <button 
                      className={styles.viewFullReport}
                      onClick={() => onViewScan && onViewScan(scan)}
                    >
                      View Full Report
                    </button>
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Grid View */}
      {viewMode === 'grid' && (
        <div className={styles.scanGrid}>
          {scans.map((scan) => (
            <div 
              key={scan.id} 
              className={`${styles.scanCard} ${selectedScan?.id === scan.id ? styles.selected : ''}`}
              onClick={() => setSelectedScan(scan)}
            >
              <div className={styles.cardHeader}>
                <h3>{formatDate(scan.scan_date)}</h3>
                <div className={styles.severityBadge} style={{ backgroundColor: getSeverityColor(scan.prediction_class) }}>
                  {getSeverityLabel(scan.prediction_class)}
                </div>
              </div>
              
              <div className={styles.cardBody}>
                <div className={styles.metric}>
                  <span className={styles.metricLabel}>Confidence</span>
                  <span className={styles.metricValue}>{(scan.confidence_score * 100).toFixed(1)}%</span>
                </div>
                <div className={styles.metric}>
                  <span className={styles.metricLabel}>Risk Level</span>
                  <span className={`${styles.metricValue} ${getRiskClass(scan.risk_level)}`}>
                    {scan.risk_level}
                  </span>
                </div>
                <div className={styles.metric}>
                  <span className={styles.metricLabel}>Urgency</span>
                  <span className={`${styles.metricValue} ${getUrgencyClass(scan.urgency)}`}>
                    {scan.urgency}
                  </span>
                </div>
              </div>

              <button 
                className={styles.cardButton}
                onClick={(e) => {
                  e.stopPropagation();
                  onViewScan && onViewScan(scan);
                }}
              >
                View Details
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}