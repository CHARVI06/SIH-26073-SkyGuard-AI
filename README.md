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

##  Data Engineering and Core ML

Member 's reusable pipeline is in `src/data/`, `src/features/`, and `src/models/`.
It reads the supplied immutable `Dataset/imd_maitri.csv` file (which has no
header row), converts its `-999` missing-value marker, interpolates only short
internal gaps, and produces model-ready temperature, pressure, and humidity
features. The mapping is verified against the supplied NetCDF subset:
`tempr` is temperature, `rh` is atmospheric pressure, and `ap` is relative
humidity. Wind speed/direction are present in the source but are not baseline
model features.

From the repository root, install dependencies and run:

```powershell
python -m pip install -r requirements.txt
python -m src.run_member1
pytest -q
```

The CLI writes regenerable outputs to `data/processed/` and `data/synthetic/`:
a feature dataset, labelled synthetic spike/drop/freeze/drift/missing scenarios,
`member1_report.json` containing source quality facts, feature names, and
Isolation Forest evaluation metrics, plus a lightweight baseline artifact at
`models/member1_isolation_forest.joblib`. These derived artifacts are ignored
by Git.
See `docs/dataset.md` for exact dataset constraints and `docs/member1.md` for
the feature and model contract provided to later members.

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
