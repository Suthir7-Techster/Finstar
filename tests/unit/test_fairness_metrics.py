"""Fairness metrics on hand-computed toy examples."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from feg_mlops.fairness import fairness_report, subgroup_metrics


def test_disparate_impact_and_parity():
    # Privileged: 8/10 approved. Unprivileged: 4/10 approved.
    groups = pd.Series(["A"] * 10 + ["B"] * 10)
    approved = np.array([True] * 8 + [False] * 2 + [True] * 4 + [False] * 6)
    labels = np.zeros(20, dtype=int)  # everyone good: selection rate == TPR
    report = fairness_report(groups, approved, labels, "toy", "age_band", "A", "B")
    assert report.disparate_impact_ratio == pytest.approx(0.4 / 0.8)
    assert report.statistical_parity_difference == pytest.approx(0.4 - 0.8)
    assert report.equal_opportunity_difference == pytest.approx(0.4 - 0.8)


def test_equal_opportunity_and_average_odds():
    # Group A: 5 good (4 approved), 5 bad (1 approved) -> TPR .8, FPR .2
    # Group B: 5 good (2 approved), 5 bad (2 approved) -> TPR .4, FPR .4
    groups = pd.Series(["A"] * 10 + ["B"] * 10)
    approved = np.array(
        [
            True,
            True,
            True,
            True,
            False,
            True,
            False,
            False,
            False,
            False,
            True,
            True,
            False,
            False,
            False,
            True,
            True,
            False,
            False,
            False,
        ]
    )
    labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1] * 2)
    report = fairness_report(groups, approved, labels, "toy", "age_band", "A", "B")
    assert report.equal_opportunity_difference == pytest.approx(0.4 - 0.8)
    assert report.average_odds_difference == pytest.approx(0.5 * ((0.4 - 0.8) + (0.4 - 0.2)))


def test_subgroup_metrics_fields():
    groups = pd.Series(["A", "A", "A"])
    approved = np.array([True, False, True])
    labels = np.array([0, 0, 1])
    sg = subgroup_metrics(groups, approved, labels, "A")
    assert sg.n == 3 and sg.n_good == 2 and sg.n_bad == 1
    assert sg.selection_rate == pytest.approx(2 / 3)
    assert sg.tpr == pytest.approx(0.5)  # 1 of 2 good approved
    assert sg.fpr == pytest.approx(1.0)  # the bad applicant was approved
    assert sg.accuracy == pytest.approx(1 / 3)  # only the approved-good row is correct


def test_missing_group_raises():
    groups = pd.Series(["A"] * 4)
    with pytest.raises(ValueError):
        fairness_report(
            groups,
            np.ones(4, dtype=bool),
            np.zeros(4, dtype=int),
            "toy",
            "age_band",
            "A",
            "B",
        )


def test_zero_privileged_selection_rate_di_is_zero():
    groups = pd.Series(["A", "B"])
    approved = np.array([False, True])
    labels = np.zeros(2, dtype=int)
    report = fairness_report(groups, approved, labels, "toy", "age_band", "A", "B")
    assert report.disparate_impact_ratio == 0.0
