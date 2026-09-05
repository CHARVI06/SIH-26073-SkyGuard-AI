"""Isolation Forest baseline with explicit feature contracts and evaluation."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class IsolationForestBaseline:
    contamination: float = 0.02
    random_state: int = 42
    n_estimators: int = 200

    def fit(self, frame: pd.DataFrame, feature_columns: list[str]) -> "IsolationForestBaseline":
        if not feature_columns:
            raise ValueError("At least one feature is required.")
        if set(feature_columns).difference(frame.columns):
            raise ValueError("Requested model feature is absent from training data.")
        self.feature_columns = list(feature_columns)
        self.pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", IsolationForest(contamination=self.contamination, n_estimators=self.n_estimators, random_state=self.random_state)),
        ])
        self.pipeline.fit(frame.loc[:, self.feature_columns])
        # Higher score means more anomalous; threshold calibrated on normal training data.
        train_scores = -self.pipeline.named_steps["model"].score_samples(
            self.pipeline[:-1].transform(frame.loc[:, self.feature_columns])
        )
        self.threshold_ = float(np.quantile(train_scores, 1 - self.contamination))
        return self

    def _validate_prediction_frame(self, frame: pd.DataFrame) -> None:
        if not hasattr(self, "pipeline"):
            raise RuntimeError("Fit or load the baseline before calling predict.")
        absent = set(self.feature_columns).difference(frame.columns)
        if absent:
            raise ValueError(f"Prediction frame is missing model features: {sorted(absent)}")

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        self._validate_prediction_frame(frame)
        transformed = self.pipeline[:-1].transform(frame.loc[:, self.feature_columns])
        scores = -self.pipeline.named_steps["model"].score_samples(transformed)
        result = frame.copy()
        result["anomaly_score"] = scores
        result["is_anomaly"] = scores >= self.threshold_
        return result

    def save(self, path: str | Path) -> None:
        """Persist the compact fitted baseline and its feature contract."""
        if not hasattr(self, "pipeline"):
            raise RuntimeError("Fit the baseline before saving it.")
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, destination)

    @classmethod
    def load(cls, path: str | Path) -> "IsolationForestBaseline":
        """Load a baseline produced by :meth:`save` with basic type checking."""
        model = joblib.load(path)
        if not isinstance(model, cls) or not hasattr(model, "threshold_"):
            raise ValueError("The artifact is not a fitted IsolationForestBaseline.")
        return model

    def evaluate(self, frame: pd.DataFrame, label_column: str = "is_synthetic_anomaly") -> dict[str, float]:
        if label_column not in frame:
            raise ValueError(f"Evaluation label is missing: {label_column}")
        predicted = self.predict(frame)
        truth = frame[label_column].astype(bool)
        precision, recall, f1, _ = precision_recall_fscore_support(truth, predicted["is_anomaly"], average="binary", zero_division=0)
        metrics = {"precision": float(precision), "recall": float(recall), "f1": float(f1), "average_precision": float(average_precision_score(truth, predicted["anomaly_score"]))}
        if truth.nunique() == 2:
            metrics["roc_auc"] = float(roc_auc_score(truth, predicted["anomaly_score"]))
        return metrics
