"use client";
import { useState, useRef } from "react";
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

ChartJS.register(CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend);

export default function Home() {
  const [image, setImage] = useState<File | null>(null);
  const [result, setResult] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

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
    setImage(e.target.files?.[0] || null);
  };

  const handleSubmit = async () => {
    if (!image) return;
    const formData = new FormData();
    formData.append("file", image);

    setLoading(true);
    try {
      const res = await axios.post("http://localhost:8000/predict", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      setResult(res.data);
    } catch (err) {
      console.error(err);
    }
    setLoading(false);
  };

  return (
    <>
      <main className="container">
        <h1>Diabetic Retinopathy Detector</h1>

        <div className="upload-section">
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            onChange={handleFileChange}
            className="hidden-file-input"
          />

          <button onClick={handleFileClick} className="file-picker-btn">
            {image ? `Selected: ${image.name}` : "Choose Image"}
          </button>

          <button
            onClick={handleSubmit}
            disabled={!image || loading}
            className="btn-primary"
          >
            {loading ? "Predicting..." : "Upload & Predict"}
          </button>
        </div>

        {result && (
          <section className="result-section">
            <h2>
              Prediction:{" "}
              <span className="prediction-label">{labels[result.prediction]}</span>
            </h2>
            <p className="confidence">
              Confidence: <strong>{(result.confidence * 100).toFixed(2)}%</strong>
            </p>

            <div className="chart-wrapper">
              <Bar
                data={{
                  labels,
                  datasets: [
                    {
                      label: "Probability (%)",
                      data: result.probabilities.map((p: number) => p * 100),
                      backgroundColor: "rgba(14, 165, 233, 0.8)",
                      borderRadius: 8,
                      borderSkipped: false,
                      maxBarThickness: 40,
                    },
                  ],
                }}
                options={{
                  responsive: true,
                  plugins: {
                    legend: { display: false },
                    tooltip: {
                      enabled: true,
                      backgroundColor: "rgba(14, 165, 233, 0.85)",
                      titleColor: "#fff",
                      bodyColor: "#e0f2fe",
                      cornerRadius: 6,
                      padding: 8,
                      displayColors: false,
                      caretSize: 6,
                    },
                  },
                  scales: {
                    y: {
                      beginAtZero: true,
                      max: 100,
                      ticks: {
                        color: "#0ea5e9",
                        font: { weight: 600, size: 14 },
                        stepSize: 20,
                        callback: (val) => val + "%",
                      },
                      grid: {
                        color: "rgba(14, 165, 233, 0.15)",
                      },
                      border: {
                        display: true,
                        color: "rgba(14, 165, 233, 0.3)",
                        width: 1,
                      },
                    },
                    x: {
                      ticks: {
                        color: "#0369a1",
                        font: { weight: 600, size: 14 },
                      },
                      grid: { display: false },
                    },
                  },
                  animation: { duration: 900, easing: "easeOutQuart" },
                }}
              />
            </div>

            {result.gradcam_base64 && (
              <div className="gradcam-section">
                <h3>Grad-CAM Heatmap</h3>
                <img
                  src={`data:image/png;base64,${result.gradcam_base64}`}
                  alt="GradCAM"
                  className="gradcam-img"
                />
              </div>
            )}
          </section>
        )}
      </main>

      <style jsx>{`
        :root {
          --color-blue: #0ea5e9;
          --color-blue-dark: #0369a1;
          --color-cyan: #0bc5ea;
          --color-bg-light: #f0f9ff;
          --color-bg-dark: #1e293b;
          --color-text-light: #334155;
          --color-text-dark: #cbd5e1;
          --color-white: #fff;
        }

        .container {
          min-height: 100vh;
          padding: 3rem 1.5rem;
          background: var(--color-bg-light);
          color: var(--color-text-light);
          display: flex;
          flex-direction: column;
          align-items: center;
          justify-content: center;
          transition: background-color 0.5s ease, color 0.5s ease;
          font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
        }

        @media (prefers-color-scheme: dark) {
          .container {
            background: var(--color-bg-dark);
            color: var(--color-text-dark);
          }
        }

        h1 {
          font-size: 2.5rem;
          font-weight: 800;
          margin-bottom: 2.5rem;
          color: var(--color-blue-dark);
          text-shadow: 0 2px 4px rgba(14, 165, 233, 0.4);
        }

        .upload-section {
          background: var(--color-white);
          padding: 2rem;
          border-radius: 1.25rem;
          box-shadow: 0 10px 20px rgba(14, 165, 233, 0.15);
          width: 100%;
          max-width: 400px;
          display: flex;
          flex-direction: column;
          align-items: center;
          gap: 1.5rem;
          transition: background-color 0.5s ease;
        }

        @media (prefers-color-scheme: dark) {
          .upload-section {
            background: #273549;
            box-shadow: 0 10px 20px rgba(14, 165, 233, 0.5);
          }
        }

        /* Hide the default file input */
        .hidden-file-input {
          display: none;
        }

        .file-picker-btn {
          width: 100%;
          padding: 0.75rem 0;
          font-size: 1.125rem;
          font-weight: 700;
          border-radius: 9999px;
          border: 2px solid var(--color-cyan);
          background: linear-gradient(90deg, #38bdf8, #06b6d4);
          color: var(--color-white);
          cursor: pointer;
          box-shadow: 0 5px 15px rgba(6, 182, 212, 0.6);
          transition: background 0.3s ease, transform 0.15s ease;
          text-align: center;
          user-select: none;
        }
        .file-picker-btn:hover {
          background: linear-gradient(90deg, #0ea5e9, #0891b2);
          transform: scale(1.05);
          box-shadow: 0 7px 20px rgba(6, 182, 212, 0.8);
        }

        .btn-primary {
          width: 100%;
          padding: 0.75rem 0;
          font-size: 1.125rem;
          font-weight: 700;
          border-radius: 9999px;
          border: none;
          background: linear-gradient(90deg, #3b82f6, #06b6d4);
          color: var(--color-white);
          box-shadow: 0 5px 15px rgba(14, 165, 233, 0.6);
          cursor: pointer;
          transition: background 0.3s ease, transform 0.15s ease;
        }
        .btn-primary:hover:not(:disabled) {
          background: linear-gradient(90deg, #2563eb, #0891b2);
          transform: scale(1.05);
        }
        .btn-primary:disabled {
          background: #94a3b8;
          cursor: not-allowed;
          box-shadow: none;
          transform: none;
        }

        .result-section {
          margin-top: 3rem;
          width: 100%;
          max-width: 900px;
          background: var(--color-white);
          border-radius: 1.5rem;
          padding: 2.5rem 2rem;
          box-shadow: 0 15px 30px rgba(14, 165, 233, 0.2);
          transition: background-color 0.5s ease;
        }

        @media (prefers-color-scheme: dark) {
          .result-section {
            background: #1e293b;
            box-shadow: 0 15px 30px rgba(14, 165, 233, 0.6);
          }
        }

        .result-section h2 {
          font-size: 2rem;
          font-weight: 700;
          color: var(--color-blue-dark);
          margin-bottom: 0.75rem;
        }
        .prediction-label {
          color: #4f46e5; /* Indigo-600 */
        }

        .confidence {
          font-size: 1.125rem;
          font-weight: 500;
          margin-bottom: 2rem;
          color: var(--color-text-light);
        }
        @media (prefers-color-scheme: dark) {
          .confidence {
            color: var(--color-text-dark);
          }
        }

        .chart-wrapper {
          background: linear-gradient(90deg, #e0f2fe, #bae6fd);
          border-radius: 1rem;
          padding: 1.5rem;
          box-shadow: inset 0 0 10px #7dd3fcaa;
        }

        .gradcam-section {
          margin-top: 2.5rem;
          text-align: center;
        }
        .gradcam-section h3 {
          font-size: 1.5rem;
          font-weight: 700;
          color: var(--color-blue-dark);
          margin-bottom: 1rem;
        }
        .gradcam-img {
          max-width: 100%;
          border-radius: 1rem;
          box-shadow: 0 10px 25px rgba(14, 165, 233, 0.35);
        }
      `}</style>
    </>
  );
}
