"""Actionable counterfactual explanations ("what would change the decision").

Counterfactuals answer the customer-facing question the paper highlights:
*why was I declined, and what would have led to a better outcome?* Only
features the applicant can plausibly act on (per the schema's mutability
flags) are ever proposed, every candidate value stays within schema bounds,
and moves must go in the feature's "improving" direction.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd

from feg_mlops.data.schema import FEATURE_SPECS_BY_NAME, MUTABLE_FEATURES
from feg_mlops.explain.explainer import RiskScoreFn

MAX_CHANGES = 3
SEARCH_WIDTH = 8  # top-N actions considered for combinations


@dataclass(frozen=True)
class CounterfactualResult:
    found: bool
    original_score: float
    new_score: float | None
    target_score: float
    changes: dict[str, dict[str, float]]  # feature -> {from, to}

    def render(self) -> str:
        header = (
            f"Original predicted risk {self.original_score:.3f}; target ≤ {self.target_score:.3f}. "
        )
        if not self.changes:
            return header + "No single actionable change improves the outcome."
        parts = [
            f"{feat}: {chg['from']:.3f} → {chg['to']:.3f}" for feat, chg in self.changes.items()
        ]
        if self.found:
            return (
                header
                + f"Reaching the target (risk → {self.new_score:.3f}) is achievable by: "
                + "; ".join(parts)
            )
        return (
            header
            + f"Closest achievable improvement is risk → {self.new_score:.3f} via: "
            + "; ".join(parts)
        )


def _candidate_actions(
    row: pd.Series,
    background: pd.DataFrame,
    attributions: dict[str, float],
) -> list[tuple[str, float]]:
    """Actionable (feature, target value) moves ordered by expected impact."""
    actions: list[tuple[str, float, float]] = []
    for feat in MUTABLE_FEATURES:
        spec = FEATURE_SPECS_BY_NAME[feat]
        current = float(row[feat])
        q = background[feat].quantile
        if spec.improve_direction < 0:
            targets = [float(q(0.25)), float(q(0.10))]
        else:
            targets = [float(q(0.75)), float(q(0.90))]
        for value in targets:
            value = float(np.clip(value, spec.lo, spec.hi))
            if spec.integer:
                value = float(round(value))
            if abs(value - current) < 1e-9:
                continue
            # Actions must move the feature in its improving direction.
            if spec.improve_direction < 0 and value >= current:
                continue
            if spec.improve_direction > 0 and value <= current:
                continue
            impact = abs(attributions.get(feat, 0.0))
            actions.append((feat, value, impact))
    actions.sort(key=lambda a: a[2], reverse=True)
    return [(f, v) for f, v, _ in actions]


def find_counterfactual(
    row: pd.Series,
    score_fn: RiskScoreFn,
    target_score: float,
    background: pd.DataFrame,
    attributions: dict[str, float],
) -> CounterfactualResult:
    """Search ≤ MAX_CHANGES actions that push risk to the target.

    Combinations of increasing size over the highest-impact actions are
    evaluated; if the target is unreachable within the budget, the closest
    achievable improvement is returned (``found=False``) so the applicant
    still receives concrete guidance.
    """
    row = row.copy()
    original = float(score_fn(row.to_frame().T)[0])
    actions = _candidate_actions(row, background, attributions)[:SEARCH_WIDTH]
    if not actions:
        return CounterfactualResult(
            found=False,
            original_score=original,
            new_score=None,
            target_score=target_score,
            changes={},
        )

    def evaluate(combo: tuple[tuple[str, float], ...]) -> float:
        candidate = row.copy()
        for feat, value in combo:
            candidate[feat] = value
        return float(score_fn(candidate.to_frame().T)[0])

    best_score: float | None = None
    best_changes: dict[str, dict[str, float]] = {}
    for size in range(1, MAX_CHANGES + 1):
        for combo in combinations(actions, size):
            features = [f for f, _ in combo]
            if len(set(features)) != len(features):
                continue  # skip duplicate-feature combos
            score = evaluate(combo)
            changes = {feat: {"from": float(row[feat]), "to": value} for feat, value in combo}
            if score <= target_score:
                return CounterfactualResult(
                    found=True,
                    original_score=original,
                    new_score=score,
                    target_score=target_score,
                    changes=changes,
                )
            if best_score is None or score < best_score:
                best_score, best_changes = score, changes

    return CounterfactualResult(
        found=False,
        original_score=original,
        new_score=best_score,
        target_score=target_score,
        changes=best_changes,
    )
