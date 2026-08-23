"""Drift statistics and alert evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from feg_mlops.config import load_configs
from feg_mlops.monitoring.alerts import evaluate_alerts, render_alerts_markdown
from feg_mlops.monitoring.drift import (
    compute_feature_drift,
    js_divergence_categorical,
    psi,
)
from feg_mlops.monitoring.fairness_drift import fairness_drift


def test_psi_identical_is_zero():
    x = np.random.default_rng(0).normal(size=500)
    assert psi(x, x) == pytest.approx(0.0, abs=1e-9)


def test_psi_shifted_distribution_is_large():
    rng = np.random.default_rng(1)
    ref = rng.normal(0, 1, 2000)
    cur = rng.normal(1.0, 1, 2000)
    assert psi(ref, cur) > 0.25


def test_psi_hand_computed_two_bins():
    ref = np.array([0.0] * 50 + [1.0] * 50, dtype=float)
    cur = np.array([0.0] * 90 + [1.0] * 10, dtype=float)
    # Quantile binning collapses to a single split at 0.5 -> Bernoulli(0.5)
    # vs Bernoulli(0.1): PSI = sum((q-p)ln(q/p)).
    p, q = 0.5, 0.1
    expected = (q - p) * np.log(q / p) + ((1 - q) - (1 - p)) * np.log((1 - q) / (1 - p))
    assert psi(ref, cur) == pytest.approx(expected, rel=1e-6)


def test_js_identical_is_zero():
    s = pd.Series(["x", "y", "x", "y"])
    assert js_divergence_categorical(s, s) == pytest.approx(0.0, abs=1e-9)


def test_js_mixed_categories():
    a = pd.Series(["x"] * 90 + ["y"] * 10)
    b = pd.Series(["x"] * 50 + ["y"] * 50)
    assert js_divergence_categorical(a, b) > 0.10


def test_compute_feature_drift_levels(config_dir, tmp_path):
    settings = load_configs(config_dir)
    rng = np.random.default_rng(3)
    ref = pd.DataFrame({"f1": rng.normal(0, 1, 1000), "cat": rng.choice(["a", "b"], 1000)})
    cur_ok = pd.DataFrame({"f1": rng.normal(0, 1, 1000), "cat": rng.choice(["a", "b"], 1000)})
    cur_bad = pd.DataFrame({"f1": rng.normal(1.2, 1, 1000), "cat": rng.choice(["a", "a"], 1000)})

    ok = {
        d.feature: d
        for d in compute_feature_drift(
            ref, cur_ok, ["f1", "cat"], settings.monitoring_settings.drift
        )
    }
    bad = {
        d.feature: d
        for d in compute_feature_drift(
            ref, cur_bad, ["f1", "cat"], settings.monitoring_settings.drift
        )
    }

    assert ok["f1"].level == "ok" and ok["f1"].method == "psi"
    assert ok["cat"].method == "js"
    assert bad["f1"].level == "critical"
    assert bad["cat"].level == "critical"


def _fairness(di: float) -> object:
    # Minimal duck-typed report for drift comparison.
    class R:
        disparate_impact_ratio = di
        statistical_parity_difference = 0.0
        equal_opportunity_difference = 0.0

    return R()


def test_fairness_drift_levels(config_dir):
    settings = load_configs(config_dir)
    thresholds = settings.monitoring_settings.fairness_drift

    ok = fairness_drift(_fairness(0.90), _fairness(0.88), thresholds)
    warn = fairness_drift(_fairness(0.90), _fairness(0.83), thresholds)
    crit = fairness_drift(_fairness(0.90), _fairness(0.75), thresholds)
    assert ok.level == "ok"
    assert warn.level == "warning"
    assert crit.level == "critical"
    assert crit.di_drop == pytest.approx(0.15, abs=1e-6)


def test_alerts_evaluate_and_render(config_dir):
    settings = load_configs(config_dir)
    from feg_mlops.monitoring.drift import FeatureDrift
    from feg_mlops.monitoring.fairness_drift import FairnessDriftResult

    drifts = [
        FeatureDrift("a", "psi", 0.30, "critical"),
        FeatureDrift("b", "psi", 0.12, "warning"),
        FeatureDrift("c", "psi", 0.01, "ok"),
    ]
    fdrift = FairnessDriftResult(0.75, 0.90, 0.15, -0.10, -0.05, "critical")
    alerts = evaluate_alerts(drifts, fdrift, settings)
    components = {(a.severity, a.component) for a in alerts}
    assert ("critical", "data-drift") in components
    assert ("critical", "fairness-drift") in components
    text = render_alerts_markdown(alerts)
    assert "CRITICAL" in text
    assert "No alerts" in render_alerts_markdown([])
