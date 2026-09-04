# SkyGuard AI

## SIH26073 — AI/ML-Based Intelligent Anomaly Detection for Automatic Weather Stations (AWS)

SkyGuard AI is an AI/ML-based real-time anomaly detection and sensor health monitoring system for Automatic Weather Stations (AWS).

The system analyzes:

- Temperature (°C)
- Atmospheric Pressure (hPa)
- Relative Humidity (%)

to identify abnormal observations, sensor faults, temporal anomalies, communication/data-quality issues, and possible sensor degradation.

## Objectives

- Detect anomalies in real time
- Identify sensor faults and abnormal observations
- Detect spikes, frozen values, communication errors, and drift
- Learn temporal and multivariate patterns
- Distinguish genuine meteorological variations from sensor/data anomalies
- Provide anomaly confidence and severity
- Provide explainable AI-based reasoning
- Monitor sensor health
- Support possible corrected/imputed values
- Support scalable monitoring of multiple AWS stations

## Proposed Architecture

AWS Data
→ Data Quality Layer
→ Feature Engineering
→ AI Anomaly Detection
→ Ensemble Decision Engine
→ Root Cause Analysis
→ Explainable AI
→ Sensor Health
→ Correction/Imputation
→ Dashboard

## AI/ML Approach

The project will investigate a hybrid anomaly detection approach consisting of:

1. Rule-based data quality checks
2. Isolation Forest for multivariate anomaly detection
3. LSTM Autoencoder for temporal anomaly detection
4. Multivariate consistency analysis
5. Ensemble anomaly scoring
6. Root-cause classification
7. Explainable AI
8. Sensor health and degradation monitoring

## Real-Time Demonstration

The prototype will use historical AWS observations and a local streaming simulator to reproduce real-time sensor observations.

Synthetic anomalies will be injected to demonstrate:

- Sudden spikes
- Sudden drops
- Frozen sensor values
- Gradual sensor drift
- Missing observations
- Communication failures
- Multivariate inconsistencies

The prototype is designed to operate without dependence on external weather APIs.

## Technology Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- PyTorch / TensorFlow
- SHAP
- Flask / FastAPI
- SQLite
- HTML
- CSS
- JavaScript
- Plotly / Chart.js
- Git / GitHub

## Project Structure

```text
SkyGuard-AI/
│
├── backend/
├── dashboard/
├── data/
│   ├── raw/
│   ├── processed/
│   └── synthetic/
├── docs/
├── models/
├── notebooks/
├── src/
│   ├── anomaly/
│   ├── correction/
│   ├── data/
│   ├── explainability/
│   ├── features/
│   ├── health/
│   ├── models/
│   └── streaming/
├── tests/
│
├── .gitignore
├── README.md
└── requirements.txt
