"""Data drift detection: PSI, Jensen-Shannon divergence and KS.

Implements the paper's "Deployment & Monitoring" stage controls — drift
detectors on input distributions (Section 4.3, citing Polyzotis et al. 2018)
— with two-sided alert thresholds from ``monitoring.yaml``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from feg_mlops.config import DriftThresholds

EPSILON = 1e-6


@dataclass(frozen=True)
class FeatureDrift:
    feature: str
    method: str  # "psi" | "js" | "ks"
    statistic: float
    level: str  # "ok" | "warning" | "critical"

    def to_dict(self) -> dict[str, object]:
        return {
            "feature": self.feature,
            "method": self.method,
            "statistic": round(self.statistic, 5),
            "level": self.level,
        }


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index on quantile bins of the reference window."""
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        # Degenerate (near-constant) reference: compare point masses.
        p_ref = np.array([max(1.0 - expected.mean(), EPSILON), min(expected.mean(), 1.0 - EPSILON)])
        p_cur = np.array([max(1.0 - actual.mean(), EPSILON), min(actual.mean(), 1.0 - EPSILON)])
        return float(np.sum((p_cur - p_ref) * np.log(p_cur / p_ref)))
    edges[0], edges[-1] = -np.inf, np.inf

    def proportions(values: np.ndarray) -> np.ndarray:
        counts = np.histogram(values, bins=edges)[0].astype(float)
        return counts / max(len(values), 1)

    p_ref = proportions(expected)
    p_cur = proportions(actual)
    p_ref = np.clip(p_ref, EPSILON, None)
    p_cur = np.clip(p_cur, EPSILON, None)
    return float(np.sum((p_cur - p_ref) * np.log(p_cur / p_ref)))


def js_divergence_categorical(expected: pd.Series, actual: pd.Series) -> float:
    """Jensen-Shannon divergence (natural log) between category mixes."""
    categories = sorted(set(expected.dropna()) | set(actual.dropna()))
    if not categories:
        return 0.0
    p = np.array([(expected == c).mean() for c in categories])
    q = np.array([(actual == c).mean() for c in categories])
    p = np.clip(p, EPSILON, None) / np.clip(p, EPSILON, None).sum()
    q = np.clip(q, EPSILON, None) / np.clip(q, EPSILON, None).sum()
    m = 0.5 * (p + q)
    kl_pm = float(np.sum(p * np.log(p / m)))
    kl_qm = float(np.sum(q * np.log(q / m)))
    return 0.5 * (kl_pm + kl_qm)


def ks_statistic(expected: np.ndarray, actual: np.ndarray) -> float:
    return float(
        ks_2samp(np.asarray(expected, dtype=float), np.asarray(actual, dtype=float)).statistic
    )


def _level(value: float, warning: float, critical: float) -> str:
    if value >= critical:
        return "critical"
    if value >= warning:
        return "warning"
    return "ok"


def _is_categorical(series: pd.Series) -> bool:
    return not pd.api.types.is_numeric_dtype(series) or series.nunique() <= 2


def compute_feature_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    features: list[str] | tuple[str, ...],
    thresholds: DriftThresholds,
) -> list[FeatureDrift]:
    """Drift per feature: PSI for continuous, JS for categorical/binary."""
    results: list[FeatureDrift] = []
    for feature in features:
        if feature not in reference.columns or feature not in current.columns:
            continue
        ref_col, cur_col = reference[feature], current[feature]
        if _is_categorical(ref_col):
            stat = js_divergence_categorical(ref_col, cur_col)
            method = "js"
            level = _level(
                stat, thresholds.js_divergence.warning, thresholds.js_divergence.critical
            )
        else:
            stat = psi(ref_col.to_numpy(), cur_col.to_numpy())
            method = "psi"
            level = _level(stat, thresholds.psi.warning, thresholds.psi.critical)
        results.append(FeatureDrift(feature=feature, method=method, statistic=stat, level=level))
    return results


def drift_report_markdown(drifts: list[FeatureDrift]) -> str:
    lines = ["| Feature | Method | Statistic | Level |", "|---|---|---|---|"]
    for d in sorted(drifts, key=lambda d: d.statistic, reverse=True):
        lines.append(f"| `{d.feature}` | {d.method} | {d.statistic:.4f} | {d.level} |")
    return "\n".join(lines) + "\n"
