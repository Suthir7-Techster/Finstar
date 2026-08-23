"""Counterfactual engine and reason codes with a synthetic linear scorer."""

from __future__ import annotations

import pandas as pd
import pytest

from feg_mlops.data.schema import FEATURE_SPECS, MUTABLE_FEATURES
from feg_mlops.explain.counterfactual import find_counterfactual
from feg_mlops.explain.reason_codes import reason_codes


def _frame() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    rng = __import__("numpy").random.default_rng(11)
    background = pd.DataFrame({s.name: rng.uniform(s.lo, s.hi, 200) for s in FEATURE_SPECS})
    row = background.iloc[0].copy()
    row["credit_utilization"] = 0.9  # high utilization drives risk up
    row["pct_on_time_payments"] = 0.7
    return background, row, background


def _linear_scorer(frame: pd.DataFrame) -> list[float]:
    import numpy as np

    return np.clip(
        0.5
        + 0.6 * (frame["credit_utilization"] - 0.4)
        - 0.6 * (frame["pct_on_time_payments"] - 0.8)
        + 0.2 * (frame["debt_to_income"] - 0.3),
        0.0,
        1.0,
    ).to_numpy()


def test_counterfactual_finds_target_and_respects_constraints():
    background, row, _ = _frame()
    attributions = {
        "credit_utilization": 0.30,
        "pct_on_time_payments": -0.25,
        "debt_to_income": 0.10,
    }
    result = find_counterfactual(
        row,
        _linear_scorer,
        target_score=0.45,
        background=background,
        attributions=attributions,
    )
    assert result.found
    assert result.new_score is not None and result.new_score <= 0.45
    assert set(result.changes) <= set(MUTABLE_FEATURES)
    for feat, change in result.changes.items():
        spec = next(s for s in FEATURE_SPECS if s.name == feat)
        assert spec.lo <= change["to"] <= spec.hi
        if spec.improve_direction < 0:
            assert change["to"] < change["from"]
        else:
            assert change["to"] > change["from"]


def test_counterfactual_reports_best_effort_when_unreachable():
    background, row, _ = _frame()

    def harsh_scorer(frame: pd.DataFrame) -> list[float]:
        return [0.99] * len(frame)  # nothing ever helps

    result = find_counterfactual(
        row,
        harsh_scorer,
        target_score=0.5,
        background=background,
        attributions={"credit_utilization": 0.3},
    )
    assert not result.found
    assert "Closest achievable" in result.render() or "No single" in result.render()


def test_reason_codes_rank_risk_factors():
    row = pd.Series({"credit_utilization": 0.9, "income_annual": 20000.0})
    attributions = {
        "credit_utilization": 0.31,  # pushes risk up
        "pct_on_time_payments": -0.40,  # pushes risk down: excluded
        "num_delinquencies_24m": 0.15,
        "income_annual": 0.08,
    }
    codes = reason_codes(row, attributions, max_codes=2)
    assert [c.code for c in codes] == ["credit_utilization", "num_delinquencies_24m"]
    assert "utilization" in codes[0].text.lower()
    assert codes[0].contribution == pytest.approx(0.31)
