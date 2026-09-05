"""Focused tests for Member 1's portable data and classical ML pipeline."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.load_data import load_imd_netcdf, project_root
from src.data.preprocess import preprocess_observations
from src.data.synthetic import inject_synthetic_anomalies
from src.features.temporal_features import add_temporal_features
from src.features.rolling_features import add_rolling_features
from src.features.feature_contract import MODEL_FEATURE_COLUMNS, select_model_features
from src.models.isolation_forest import IsolationForestBaseline


def raw_frame(rows: int = 300) -> pd.DataFrame:
    time = pd.date_range("2020-01-01", periods=rows, freq="h")
    x = np.linspace(0, 8 * np.pi, rows)
    return pd.DataFrame({"timestamp": time, "tempr": 15 + np.sin(x), "rh": 1000 + np.cos(x), "ws": 5, "wd": 100, "ap": 60 + np.sin(x)})


def test_preprocess_replaces_sentinel_and_tracks_imputation():
    raw = raw_frame(20)
    raw.loc[5, "tempr"] = -999
    raw.loc[6, "ap"] = -999
    clean = preprocess_observations(raw)
    assert clean["temperature_c"].notna().all()
    assert clean["was_imputed"].any()
    assert {"pressure_hpa", "humidity_pct"}.issubset(clean.columns)


def test_features_reset_after_gap():
    clean = preprocess_observations(raw_frame())
    clean.loc[100:, "timestamp"] += pd.Timedelta(hours=8)
    temporal = add_temporal_features(clean)
    result = add_rolling_features(temporal, ("temperature_c",))
    assert result["segment_id"].nunique() == 2
    assert "temperature_c_rolling_mean_6" in result


def test_rolling_mean_uses_only_prior_values():
    frame = pd.DataFrame({
        "timestamp": pd.date_range("2020-01-01", periods=7, freq="h"),
        "temperature_c": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 100.0],
    })
    featured = add_rolling_features(add_temporal_features(frame), ("temperature_c",))
    assert featured.loc[6, "temperature_c_rolling_mean_6"] == 3.5


def test_synthetic_injection_is_deterministic_and_does_not_mutate_input():
    clean = preprocess_observations(raw_frame())
    before = clean.copy(deep=True)
    first = inject_synthetic_anomalies(clean, random_state=11, per_type=5)
    second = inject_synthetic_anomalies(clean, random_state=11, per_type=5)
    pd.testing.assert_frame_equal(first, second)
    pd.testing.assert_frame_equal(clean, before)


def test_synthetic_labels_and_baseline_evaluation():
    clean = preprocess_observations(raw_frame())
    featured = add_rolling_features(add_temporal_features(clean), ("temperature_c", "pressure_hpa", "humidity_pct")).dropna()
    synthetic = inject_synthetic_anomalies(featured, per_type=5)
    feature_columns = [column for column in featured if column not in {"timestamp", "segment_id", "was_imputed"}]
    model = IsolationForestBaseline(contamination=0.1, n_estimators=50).fit(featured, feature_columns)
    metrics = model.evaluate(synthetic)
    assert synthetic["is_synthetic_anomaly"].any()
    assert 0 <= metrics["f1"] <= 1


def test_model_feature_contract_is_complete_and_stable():
    clean = preprocess_observations(raw_frame())
    featured = add_rolling_features(add_temporal_features(clean), ("temperature_c", "pressure_hpa", "humidity_pct"))
    assert select_model_features(featured.columns) == list(MODEL_FEATURE_COLUMNS)


def test_netcdf_reader_decodes_verified_subset_when_raw_file_is_available():
    path = project_root() / "Dataset" / "imd_maitri.nc"
    if not path.is_file():
        pytest.skip("Supplied NetCDF source is intentionally not tracked in Git.")
    frame = load_imd_netcdf(path)
    assert len(frame) == 16_514
    assert frame.loc[0, "timestamp"] == pd.Timestamp("2015-01-01 00:00:00")
    assert frame.loc[len(frame) - 1, "timestamp"] == pd.Timestamp("2016-12-19 12:00:00")
    assert frame.loc[0, ["tempr", "rh", "ap"]].tolist() == pytest.approx([-1.6, 972.2, 53.0])


def test_higher_negated_score_is_classified_as_anomaly():
    """Guard against reversing IsolationForest score_samples semantics."""
    rng = np.random.default_rng(7)
    train = pd.DataFrame({"feature": rng.normal(0, 0.2, size=300)})
    candidates = pd.DataFrame({"feature": [0.0, 0.1, 8.0]})
    model = IsolationForestBaseline(contamination=0.02, n_estimators=100, random_state=7).fit(train, ["feature"])
    result = model.predict(candidates)
    assert result.loc[2, "anomaly_score"] > result.loc[:1, "anomaly_score"].max()
    assert result.loc[2, "anomaly_score"] >= model.threshold_
    assert bool(result.loc[2, "is_anomaly"])


def test_saved_baseline_preserves_scores(tmp_path):
    train = pd.DataFrame({"feature": np.linspace(-1, 1, 100)})
    model = IsolationForestBaseline(n_estimators=25, random_state=3).fit(train, ["feature"])
    artifact = tmp_path / "baseline.joblib"
    model.save(artifact)
    restored = IsolationForestBaseline.load(artifact)
    assert restored.predict(train)["anomaly_score"].equals(model.predict(train)["anomaly_score"])
