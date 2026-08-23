"""Mitigation tests: reweighing math and group-threshold optimization."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from feg_mlops.fairness import subgroup_metrics
from feg_mlops.fairness.mitigation import (
    apply_group_thresholds,
    optimize_group_thresholds,
    reweighing_weights,
)


def test_reweighing_balanced_joint_is_identity():
    groups = pd.Series(["A", "A", "B", "B"])
    labels = pd.Series([0, 1, 0, 1])
    weights = reweighing_weights(groups, labels)
    assert np.allclose(weights, 1.0)


def test_reweighing_decorrelates_group_and_label():
    # Biased joint: B is over-labeled bad.
    groups = pd.Series(["A"] * 6 + ["B"] * 6)
    labels = pd.Series([0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1])
    weights = reweighing_weights(groups, labels)
    df = pd.DataFrame({"g": groups, "y": labels, "w": weights})
    # Weighted joint P(A=a, Y=y) must equal P(A=a) * P(Y=y).
    total = df["w"].sum()
    for g in ("A", "B"):
        for y in (0, 1):
            cell = df[(df.g == g) & (df.y == y)]["w"].sum() / total
            marginal = (df.g == g).mean() * (df.y == y).mean()
            assert cell == pytest.approx(marginal, rel=1e-9)
    assert (weights > 0).all()


def test_group_thresholds_equalize_tpr_and_respect_floor():
    rng = np.random.default_rng(7)
    n = 400
    groups = pd.Series(np.where(rng.random(n) < 0.5, "A", "B"))
    # Group B's scores are shifted up (the discrimination the post-processor
    # must correct).
    scores = rng.random(n) + 0.25 * (groups == "B").to_numpy()
    labels = pd.Series((rng.random(n) < 0.4).astype(int))
    base_threshold = 0.5

    thresholds = optimize_group_thresholds(groups, labels, scores, base_threshold, min_recall=0.7)
    approved = apply_group_thresholds(groups, scores, thresholds)

    tprs = {}
    for g in ("A", "B"):
        sg = subgroup_metrics(groups, approved, labels.to_numpy(), g)
        tprs[g] = sg.tpr
    assert tprs["A"] == pytest.approx(tprs["B"], abs=0.03)
    assert min(tprs.values()) >= 0.68  # floor honored (quantile tolerance)


def test_group_thresholds_all_good_group_matches_quantile():
    groups = pd.Series(["A", "A", "A", "B"])
    labels = pd.Series([0, 0, 0, 0])
    scores = np.array([0.1, 0.2, 0.3, 0.4])
    thresholds = optimize_group_thresholds(groups, labels, scores, 0.25, min_recall=0.5)
    # Both groups are all-good; common TPR = A's TPR at base = 2/3.
    expected_a = float(np.quantile([0.1, 0.2, 0.3], 2 / 3))
    assert thresholds["A"] == pytest.approx(expected_a)
    assert thresholds["B"] == pytest.approx(0.4)  # B's only good score
