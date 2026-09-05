"""CLI to generate Member 1 processed/synthetic data and evaluate the baseline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import pandas as pd

from src.data.preprocess import BASE_VARIABLES, preprocess_csv
from src.data.synthetic import inject_synthetic_anomalies
from src.data.validate import quality_report
from src.data.load_data import load_imd_csv
from src.features.temporal_features import add_temporal_features
from src.features.rolling_features import add_rolling_features
from src.features.feature_contract import select_model_features
from src.models.isolation_forest import IsolationForestBaseline


def feature_frame(frame):
    featured = add_temporal_features(frame)
    featured = add_rolling_features(featured, BASE_VARIABLES)
    return featured


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and evaluate the Member 1 Isolation Forest baseline.")
    parser.add_argument("--input", default=None, help="Optional path to imd_maitri.csv")
    parser.add_argument("--output-dir", default="data", help="Directory for derived outputs")
    parser.add_argument("--model-output", default="models/member1_isolation_forest.joblib", help="Path for the lightweight baseline artifact")
    args = parser.parse_args()
    output = Path(args.output_dir)
    raw = load_imd_csv(args.input)
    clean = preprocess_csv(args.input)
    featured = feature_frame(clean).dropna().reset_index(drop=True)
    train_size = int(len(featured) * 0.7)
    if train_size < 200 or len(featured) - train_size < 200:
        raise ValueError("The cleaned feature dataset is too small for a chronological train/evaluation split.")
    train_frame = featured.iloc[:train_size].copy()
    split_timestamp = featured.iloc[train_size]["timestamp"]
    evaluation_base = clean.loc[clean["timestamp"] >= split_timestamp].copy()
    # Synthetic anomalies are deliberately injected only after the temporal
    # split, so neither their values nor their labels can influence fitting.
    synthetic_base = inject_synthetic_anomalies(evaluation_base)
    # Rebuild temporal features after injection, with enough prior clean history
    # for the 24-observation trailing window. This mirrors live inference and
    # prevents stale deltas/rolling statistics from inflating evaluation.
    history = clean.loc[clean["timestamp"] < split_timestamp].tail(48)
    synthetic = feature_frame(pd.concat([history, synthetic_base], ignore_index=True))
    synthetic = synthetic.loc[synthetic["timestamp"] >= split_timestamp].copy()
    feature_columns = select_model_features(featured.columns)
    model = IsolationForestBaseline().fit(train_frame, feature_columns)
    metrics = model.evaluate(synthetic)
    (output / "processed").mkdir(parents=True, exist_ok=True)
    (output / "synthetic").mkdir(parents=True, exist_ok=True)
    featured.to_csv(output / "processed" / "imd_maitri_member1_features.csv", index=False)
    synthetic.to_csv(output / "synthetic" / "imd_maitri_synthetic_anomalies.csv", index=False)
    model.save(args.model_output)
    report = {
        "raw_quality": quality_report(raw),
        "feature_columns": feature_columns,
        "split": {"method": "chronological", "train_rows": len(train_frame), "evaluation_rows": len(synthetic)},
        "baseline_metrics": metrics,
    }
    (output / "processed" / "member1_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
