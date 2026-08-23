"""Candidate model training.

Two candidates compete for promotion:

- a calibrated, standardized **logistic regression** — the inherently
  interpretable champion the paper argues for via Caruana et al. (2015);
- a **histogram gradient boosting** challenger for the accuracy baseline.

An operating threshold is chosen on a held-out calibration split (never on
the evaluation set) to maximize balanced accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from feg_mlops.config import ModelSettings
from feg_mlops.data.schema import FEATURE_COLUMNS, LABEL_COLUMN

RANDOM_STATE = 20250823


@dataclass(frozen=True)
class TrainedCandidate:
    name: str
    estimator: Any  # fitted sklearn estimator with predict_proba
    threshold: float  # approval threshold: score < threshold => approve
    calibration_split_n: int

    def risk_scores(self, features: pd.DataFrame) -> np.ndarray:
        order = [c for c in FEATURE_COLUMNS if c in features.columns]
        scores = self.estimator.predict_proba(features[order])[:, 1]
        return np.asarray(scores, dtype=float)


def _balanced_accuracy_threshold(y: np.ndarray, scores: np.ndarray) -> float:
    """Score threshold maximizing balanced accuracy of the implied decision."""
    grid = np.unique(np.quantile(scores, np.linspace(0.01, 0.99, 197)))
    best_thr, best_ba = float(grid[0]), -1.0
    for thr in grid:
        predicted_default = scores >= thr
        ba = balanced_accuracy_score(y, predicted_default.astype(int))
        if ba > best_ba:
            best_ba, best_thr = ba, float(thr)
    return best_thr


def _logistic_pipeline(settings: ModelSettings) -> Pipeline:
    params = settings.models.logistic_regression
    clf: LogisticRegression | CalibratedClassifierCV = LogisticRegression(
        C=params.C, max_iter=params.max_iter, random_state=RANDOM_STATE
    )
    if params.calibrate:
        clf = CalibratedClassifierCV(clf, method="sigmoid", cv=3)
    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])


def _boosting_pipeline(settings: ModelSettings) -> Pipeline:
    params = settings.models.gradient_boosting
    clf: HistGradientBoostingClassifier | CalibratedClassifierCV = HistGradientBoostingClassifier(
        learning_rate=params.learning_rate,
        max_iter=params.max_iter,
        max_depth=params.max_depth,
        early_stopping=params.early_stopping,
        random_state=RANDOM_STATE,
    )
    return Pipeline([("clf", CalibratedClassifierCV(clf, method="sigmoid", cv=3))])


def train_candidates(
    train_features: pd.DataFrame,
    train_labels: pd.Series,
    settings: ModelSettings,
    sample_weights: np.ndarray | None = None,
) -> dict[str, TrainedCandidate]:
    """Train all configured candidates and pick their operating thresholds.

    The threshold for each candidate is selected on an internal calibration
    split (20%) using the *observed* training labels — the same information
    constraint a production system faces.
    """
    x = train_features[list(FEATURE_COLUMNS)]
    y = train_labels.to_numpy()
    if sample_weights is not None:
        sample_weights = np.asarray(sample_weights, dtype=float)

    fit_idx, cal_idx = train_test_split(
        np.arange(len(x)), test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    factories: dict[str, tuple[Pipeline, bool]] = {
        "logistic_regression": (_logistic_pipeline(settings), True),
        "gradient_boosting": (_boosting_pipeline(settings), False),
    }

    candidates: dict[str, TrainedCandidate] = {}
    for name, (pipeline, supports_weights) in factories.items():
        if sample_weights is not None and supports_weights:
            pipeline.fit(x.iloc[fit_idx], y[fit_idx], clf__sample_weight=sample_weights[fit_idx])
        else:
            pipeline.fit(x.iloc[fit_idx], y[fit_idx])
        cal_scores = pipeline.predict_proba(x.iloc[cal_idx])[:, 1]
        threshold = _balanced_accuracy_threshold(y[cal_idx], cal_scores)
        candidates[name] = TrainedCandidate(
            name=name,
            estimator=pipeline,
            threshold=threshold,
            calibration_split_n=len(cal_idx),
        )
    return candidates


def frame_for_training(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split a KYC dataframe into (features, observed labels)."""
    return df[list(FEATURE_COLUMNS)], df[LABEL_COLUMN]
