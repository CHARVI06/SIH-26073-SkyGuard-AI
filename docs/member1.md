# Member 1 Feature and Baseline Contract

`python -m src.run_member1` creates the reproducible Member 1 handoff.

## Model-ready variables

Base variables are `temperature_c`, `pressure_hpa`, and `humidity_pct`. The
pipeline adds cyclic hour/day-of-year values to distinguish regular daily and
seasonal variation from unusual readings, an outage-aware `segment_id`, and
trailing 6- and 24-observation means, standard deviations, and one-step deltas.
These describe local level, variability, and abrupt change. Rolling references
use only earlier readings and reset after gaps longer than three hours, so they
do not bridge outages or leak the currently scored value.

`src.features.feature_contract.MODEL_FEATURE_COLUMNS` is the canonical ordered
input contract for Member 2 and any integration layer. It contains the three
base measurements, four cyclical calendar values, and each base variable's
one-step delta plus 6- and 24-observation trailing mean and standard deviation.
The generated `data/processed/member1_report.json` repeats this exact list.

## Baseline contract

`IsolationForestBaseline` fits a median imputer, standard scaler, and Isolation
Forest. Scikit-learn's `score_samples()` is larger for normal observations; the
baseline negates it, so its public `anomaly_score` is larger for more unusual
observations. `is_anomaly` is true when that score is at or above the training
quantile threshold. Training always uses the earliest 70% of the time series;
synthetic evaluation is injected only into the later chronological holdout. It
intentionally does not assign severity, confidence,
root cause, health score, or corrections; those are Member 2 responsibilities.

## Synthetic evaluation

`inject_synthetic_anomalies()` makes deterministic labelled spike, drop, freeze,
drift, and missing-value demonstrations in a copy of clean data. It supplies
`is_synthetic_anomaly`, `anomaly_type`, and `affected_feature` solely as
evaluation labels. The CLI reports precision, recall, F1, average precision,
and ROC-AUC where both classes exist. The fitted lightweight model is saved to
`models/member1_isolation_forest.joblib` (ignored by Git and reproducible).
