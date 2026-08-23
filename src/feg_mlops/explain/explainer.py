"""Feature attribution via SHAP with a dependency-free fallback.

The paper prescribes post-hoc, model-agnostic XAI — explicitly naming SHAP —
generated as versioned artifacts inside the pipeline (Sections 3.4/4.2).
``ShapExplainer`` is the primary engine; ``MarginalAblationExplainer`` is a
numpy-only fallback with the same interface so the pipeline never hard-fails
on an environment where SHAP is unavailable.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from feg_mlops.data.schema import FEATURE_COLUMNS

SHAP_DETERMINISM_SEED = 20250823

# A callable mapping a feature frame to predicted default-risk scores.
RiskScoreFn = Callable[[pd.DataFrame], np.ndarray]


class ShapExplainer:
    """Permutation-SHAP attributions of predicted default risk."""

    name = "shap"

    def __init__(self, score_fn: RiskScoreFn, background: pd.DataFrame) -> None:
        import shap  # local import: heavy and compiled

        self._features = list(background.columns)
        X = np.asarray(background[self._features], dtype=float)
        self._masker = shap.maskers.Independent(X, max_samples=min(len(X), 256))

        def model_fn(arr: np.ndarray) -> np.ndarray:
            frame = pd.DataFrame(arr, columns=self._features)
            return np.asarray(score_fn(frame), dtype=float)

        self._explainer = shap.Explainer(model_fn, self._masker)

    def explain_batch(self, frame: pd.DataFrame) -> np.ndarray:
        X = np.asarray(frame[self._features], dtype=float)
        np.random.seed(SHAP_DETERMINISM_SEED)
        explanation = self._explainer(X, max_evals=200, silent=True)
        return np.asarray(explanation.values, dtype=float)

    def explain_row(self, row: pd.Series) -> dict[str, float]:
        attributions = self.explain_batch(row.to_frame().T)[0]
        return dict(zip(self._features, map(float, attributions), strict=True))


class MarginalAblationExplainer:
    """Fallback: per-feature marginal effect against background expectations.

    attribution_i(x) = risk(x) - E_background[risk(x with feature i replaced)]
    Same sign convention as SHAP (positive = increases predicted risk).
    """

    name = "marginal_ablation"

    def __init__(self, score_fn: RiskScoreFn, background: pd.DataFrame) -> None:
        self._features = list(background.columns)
        self._score_fn = score_fn
        self._background_values = {
            col: np.asarray(background[col], dtype=float) for col in self._features
        }
        self._background_sample = background.iloc[:64].copy()

    def explain_batch(self, frame: pd.DataFrame) -> np.ndarray:
        rows = frame[self._features].to_dict("records")
        out = np.zeros((len(rows), len(self._features)))
        for i, row in enumerate(rows):
            base_row = pd.DataFrame([row], columns=self._features)
            base = float(self._score_fn(base_row)[0])
            for j, col in enumerate(self._features):
                samples = self._background_sample.copy()
                samples[col] = self._background_values[col][: len(samples)]
                perturbed = base_row.loc[base_row.index.repeat(len(samples))].reset_index(drop=True)
                perturbed[col] = samples[col].to_numpy()
                expected = float(np.mean(self._score_fn(perturbed)))
                out[i, j] = base - expected
        return out

    def explain_row(self, row: pd.Series) -> dict[str, float]:
        attributions = self.explain_batch(row.to_frame().T)[0]
        return dict(zip(self._features, map(float, attributions), strict=True))


def build_explainer(
    score_fn: RiskScoreFn, background: pd.DataFrame
) -> ShapExplainer | MarginalAblationExplainer:
    """Prefer SHAP; fall back to the numpy implementation when unavailable."""
    try:
        return ShapExplainer(score_fn, background)
    except ImportError:  # pragma: no cover - exercised only without shap
        return MarginalAblationExplainer(score_fn, background)


def global_importance(attributions: np.ndarray) -> dict[str, float]:
    """Mean absolute attribution per feature, descending."""
    means = np.abs(np.asarray(attributions)).mean(axis=0)
    ranked = sorted(
        zip(FEATURE_COLUMNS, map(float, means), strict=len(FEATURE_COLUMNS) == len(means)),
        key=lambda kv: kv[1],
        reverse=True,
    )
    return dict(ranked)


def global_importance_markdown(importance: dict[str, float], top: int = 10) -> str:
    lines = ["| Feature | Mean |SHAP| |", "|---|---|"]
    for name, value in list(importance.items())[:top]:
        lines.append(f"| `{name}` | {value:.4f} |")
    return "\n".join(lines) + "\n"
