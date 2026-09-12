# SkyGuard AI

## SIH 2026: Intelligent Anomaly Detection for Automatic Weather Stations

SkyGuard AI is an AI/ML prototype for detecting unusual behavior in Automatic Weather Station (AWS) observations. It improves the reliability of weather-station data used for monitoring and disaster preparedness by identifying readings that may indicate sensor faults, data-quality problems, or unusual environmental conditions.

The current model pipeline analyzes:

- Temperature in °C
- Atmospheric pressure in hPa
- Relative humidity in %

The prototype focuses on anomaly detection and decision support. It does not claim to predict disasters.

## Problem

AWS data can contain sudden spikes, drops, frozen readings, sensor drift, missing observations, and communication gaps. A fixed threshold can miss patterns that develop over time or appear only when several sensor measurements are considered together.

SkyGuard AI uses data validation, temporal features, multivariate anomaly detection, and controlled fault scenarios to identify such behavior early.

## Solution overview

```text
Historical AWS observations
        ↓
Data validation and preprocessing
        ↓
Temporal and rolling feature engineering
        ↓
Isolation Forest baseline and temporal modelling
        ↓
Anomaly score and anomaly flag
        ↓
Synthetic-fault evaluation and reporting
```

## Dataset

The project uses the supplied IMD Maitri historical dataset:

- Files: `Dataset/imd_maitri.csv` and `Dataset/imd_maitri.nc`
- CSV coverage: 1985-01-01 to 2016-12-19
- CSV observations: 155,170
- Missing-data marker: `-999`

The CSV has no header row. Its mapping was verified against the NetCDF subset:

| CSV field | Verified project field |
| --- | --- |
| `tempr` | `temperature_c` |
| `rh` | `pressure_hpa` |
| `ap` | `humidity_pct` |
| `ws`, `wd` | Wind variables retained in the source, not baseline model inputs |

The preprocessing layer converts `-999` to missing values, interpolates only short internal gaps, and does not bridge longer outages. See [docs/dataset.md](docs/dataset.md) for complete source constraints.

## ML approach

### Feature engineering

The pipeline builds a reusable feature set from the three baseline measurements:

- Cyclical hour-of-day and day-of-year features
- Outage-aware segments
- One-step deltas
- Trailing 6-observation and 24-observation means and standard deviations

Rolling features only use earlier observations and reset after long gaps. This prevents data leakage across outages and avoids using the reading being scored in its own history.

### Isolation Forest baseline

`IsolationForestBaseline` applies median imputation, standard scaling, and Scikit-learn Isolation Forest. It produces an `anomaly_score` where a larger value represents a more unusual observation. The model uses the earliest 70% of the time series for training and evaluates the later chronological holdout.

### Temporal modelling

The repository includes an LSTM Autoencoder for sequence-based temporal anomaly analysis. It uses 24-observation sequences to identify behavior that may not be apparent from a single observation.

### Controlled evaluation scenarios

The project injects deterministic, labelled scenarios into a copy of the later holdout data:

- Sudden spikes
- Sudden drops
- Frozen sensor values
- Gradual drift
- Missing values

These labels are used only for evaluation. They are never included in the model-training data.

## Running the project

### Requirements

- Python 3.10 or newer recommended
- The supplied `Dataset/imd_maitri.csv` available in the project dataset location

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Build processed features, generate synthetic evaluation scenarios, train the Isolation Forest baseline, and write the report:

```powershell
python -m src.run_member1
```

Run tests:

```powershell
pytest -q
```

Train the LSTM Autoencoder after generating the processed feature file:

```powershell
python -m src.models.train_lstm
```

## Outputs

The reproducible pipeline creates these derived files:

```text
data/processed/imd_maitri_member1_features.csv
data/processed/member1_report.json
data/synthetic/imd_maitri_synthetic_anomalies.csv
models/member1_isolation_forest.joblib
models/lstm_autoencoder.pt
models/lstm_scaler.joblib
```

Generated datasets and model artifacts are reproducible and excluded from Git.

## Technology stack

| Area | Technologies |
| --- | --- |
| Data processing | Python, Pandas, NumPy |
| Baseline anomaly detection | Scikit-learn, Isolation Forest, Joblib |
| Temporal modelling | PyTorch, LSTM Autoencoder |
| Analysis and evaluation | Pandas, Matplotlib, Seaborn, Pytest |
| Supporting tools | Flask, Plotly, SHAP, Jupyter |

## Team contributions

| Member | Responsibility |
| --- | --- |
| Member 1 | Data ingestion, validation, preprocessing, feature engineering, and Isolation Forest baseline |
| Member 2 | Temporal anomaly analysis, LSTM Autoencoder, and anomaly-evaluation experiments |
| Member 3 | Multi-sensor consistency analysis, synthetic anomaly scenarios, and fault-type testing |
| Member 4 | Backend integration, REST API design, database handling, and streaming workflow |
| Member 5 | Dashboard development, frontend design, data visualization, and alert presentation |
| Member 6 | Explainability integration, testing, documentation, feasibility analysis, and SIH presentation preparation |

## Repository structure

```text
SkyGuard-AI/
├── Dataset/                     # Supplied raw IMD files, not modified
├── data/
│   ├── processed/               # Reproducible feature outputs
│   └── synthetic/               # Reproducible injected scenarios
├── docs/
│   ├── dataset.md               # Dataset mapping and constraints
│   └── member1.md               # Feature and baseline contract
├── models/                      # Reproducible trained artifacts
├── notebooks/                   # EDA and model notebooks
├── src/
│   ├── anomaly/                 # Temporal and multivariate evaluation
│   ├── data/                    # Loading, validation, preprocessing, injection
│   ├── features/                # Temporal, rolling, and feature contracts
│   ├── models/                  # Isolation Forest and LSTM Autoencoder
│   └── run_member1.py           # Main reproducible baseline pipeline
├── tests/
├── README.md
└── requirements.txt
```

## Limitations and next steps

The present implementation works with historical AWS observations and controlled replay scenarios. A production deployment would add authenticated ingestion from physical AWS devices, station identifiers, alert routing, field-validated fault labels, model monitoring, and periodic retraining.

## References

- World Meteorological Organization, *Guide to Instruments and Methods of Observation* (WMO-No. 8)
- Liu, Ting, and Zhou, *Isolation Forest*, ICDM 2008
- Lundberg and Lee, *A Unified Approach to Interpreting Model Predictions*, NeurIPS 2017
