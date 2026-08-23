"""Fairness gates and the policy engine."""

from __future__ import annotations

import numpy as np
import pandas as pd

from feg_mlops.config import load_configs
from feg_mlops.fairness import fairness_report
from feg_mlops.governance.policy import PolicyEngine


def _report(selection_priv: float, selection_unpriv: float):
    n = 100
    groups = pd.Series(["35_plus"] * n + ["under_35"] * n)
    approved = np.array(
        [True] * int(selection_priv * n)
        + [False] * (n - int(selection_priv * n))
        + [True] * int(selection_unpriv * n)
        + [False] * (n - int(selection_unpriv * n))
    )
    labels = np.zeros(2 * n, dtype=int)
    return fairness_report(groups, approved, labels, "toy", "age_band", "35_plus", "under_35")


def test_gates_pass_when_parity_holds(config_dir):
    settings = load_configs(config_dir)
    engine = PolicyEngine(settings.model_settings, settings.fairness_settings)
    gates = engine.fairness_gates(_report(0.70, 0.72))
    assert gates.passed, [f.name for f in gates.failures]


def test_gates_fail_on_disparate_impact(config_dir):
    settings = load_configs(config_dir)
    engine = PolicyEngine(settings.model_settings, settings.fairness_settings)
    gates = engine.fairness_gates(_report(0.70, 0.40))
    assert not gates.passed
    failed = {f.name for f in gates.failures}
    assert "disparate_impact_min" in failed
    assert "statistical_parity_abs_max" in failed


def test_promotion_decision_blocks_and_reports_reasons(config_dir):
    settings = load_configs(config_dir)
    engine = PolicyEngine(settings.model_settings, settings.fairness_settings)
    decision = engine.promotion_decision(_report(0.70, 0.40), auc=0.85)
    assert not decision.allowed
    assert any("disparate_impact_min" in r for r in decision.reasons)

    decision = engine.promotion_decision(_report(0.70, 0.40), auc=0.60)
    assert any("performance floor" in r for r in decision.reasons)

    assert engine.promotion_decision(_report(0.70, 0.72), auc=0.85).allowed
