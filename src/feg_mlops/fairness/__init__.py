"""Fairness layer: metrics, mitigation and policy gates.

Convention used throughout the package: the *favorable* outcome is approval.
``approved`` is a boolean decision per applicant, the label column is
``defaulted`` (1 = bad outcome), so:

- selection rate  = P(approved | group)
- true positive rate (TPR) = P(approved | group, truly good)  — access to
  credit for creditworthy applicants (equal opportunity)
- false positive rate (FPR) = P(approved | group, truly bad)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SubgroupMetrics:
    group: str
    n: int
    n_good: int
    n_bad: int
    selection_rate: float
    tpr: float  # P(approved | good)
    fpr: float  # P(approved | bad)
    accuracy: float  # accuracy of the default prediction implied by the decision


@dataclass(frozen=True)
class FairnessReport:
    """All fairness evidence for one model + threshold on one dataset."""

    dataset: str
    protected_attribute: str
    privileged_group: str
    unprivileged_group: str
    disparate_impact_ratio: float
    statistical_parity_difference: float
    equal_opportunity_difference: float
    average_odds_difference: float
    subgroups: list[SubgroupMetrics] = field(default_factory=list)

    def subgroup(self, group: str) -> SubgroupMetrics:
        for sg in self.subgroups:
            if sg.group == group:
                return sg
        raise KeyError(group)

    def to_dict(self) -> dict[str, object]:
        d = asdict(self)
        return d

    def render_markdown(self) -> str:
        lines = [
            f"## Fairness report — {self.dataset}",
            "",
            f"Protected attribute: `{self.protected_attribute}` "
            f"(privileged: `{self.privileged_group}`, unprivileged: `{self.unprivileged_group}`)",
            "",
            "| Metric | Value |",
            "|---|---|",
            f"| Disparate impact ratio | {self.disparate_impact_ratio:.3f} |",
            f"| Statistical parity difference | {self.statistical_parity_difference:+.3f} |",
            f"| Equal opportunity difference | {self.equal_opportunity_difference:+.3f} |",
            f"| Average odds difference | {self.average_odds_difference:+.3f} |",
            "",
            "| Group | n | Good | Bad | Selection rate | TPR | FPR | Accuracy |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for sg in self.subgroups:
            lines.append(
                f"| {sg.group} | {sg.n} | {sg.n_good} | {sg.n_bad} | "
                f"{sg.selection_rate:.3f} | {sg.tpr:.3f} | {sg.fpr:.3f} | {sg.accuracy:.3f} |"
            )
        return "\n".join(lines) + "\n"


def _safe_rate(numerator: int, denominator: int) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def subgroup_metrics(
    groups: pd.Series | np.ndarray,
    approved: np.ndarray | pd.Series,
    labels: np.ndarray | pd.Series,
    group: str,
) -> SubgroupMetrics:
    """Per-group selection/outcome metrics for one group value."""
    mask = np.asarray(groups) == group
    ap = np.asarray(approved, dtype=bool)[mask]
    y = np.asarray(labels)[mask]
    good = y == 0
    bad = ~good
    pred_default = ~ap  # decision implies predicted default when not approved
    return SubgroupMetrics(
        group=group,
        n=int(mask.sum()),
        n_good=int(good.sum()),
        n_bad=int(bad.sum()),
        selection_rate=_safe_rate(int(ap.sum()), int(mask.sum())),
        tpr=_safe_rate(int(ap[good].sum()), int(good.sum())),
        fpr=_safe_rate(int(ap[bad].sum()), int(bad.sum())),
        accuracy=_safe_rate(int((pred_default == (y == 1)).sum()), len(y)),
    )


def fairness_report(
    groups: pd.Series | np.ndarray,
    approved: np.ndarray | pd.Series,
    labels: np.ndarray | pd.Series,
    dataset: str,
    protected_attribute: str,
    privileged_group: str,
    unprivileged_group: str,
) -> FairnessReport:
    """Compute the full fairness evidence table.

    Raises ValueError if either group is absent or empty.
    """
    subgroups = [
        subgroup_metrics(groups, approved, labels, g)
        for g in (privileged_group, unprivileged_group)
    ]
    priv, unpriv = subgroups
    if priv.n == 0 or unpriv.n == 0:
        raise ValueError("both protected groups must be present and non-empty")

    di = unpriv.selection_rate / priv.selection_rate if priv.selection_rate > 0 else 0.0
    return FairnessReport(
        dataset=dataset,
        protected_attribute=protected_attribute,
        privileged_group=privileged_group,
        unprivileged_group=unprivileged_group,
        disparate_impact_ratio=float(di),
        statistical_parity_difference=float(unpriv.selection_rate - priv.selection_rate),
        equal_opportunity_difference=float(unpriv.tpr - priv.tpr),
        average_odds_difference=float(0.5 * ((unpriv.tpr - priv.tpr) + (unpriv.fpr - priv.fpr))),
        subgroups=subgroups,
    )
