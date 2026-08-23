"""Model evaluation: aggregate performance plus subgroup-stratified reporting.

The paper is explicit that aggregate scores hide discrimination (Section 3.2);
every evaluation therefore produces per-group performance alongside AUC-style
metrics, evaluated against *ground-truth* outcomes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, roc_auc_score

from feg_mlops.fairness import SubgroupMetrics, subgroup_metrics


@dataclass(frozen=True)
class PerformanceReport:
    dataset: str
    n: int
    auc: float
    brier: float
    ks: float
    balanced_accuracy: float  # at the operating threshold
    approval_rate: float
    subgroups: list[SubgroupMetrics]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def render_markdown(self) -> str:
        lines = [
            f"## Performance report — {self.dataset}",
            "",
            f"AUC **{self.auc:.3f}** · Brier {self.brier:.3f} · KS {self.ks:.3f} · "
            f"balanced accuracy {self.balanced_accuracy:.3f} · "
            f"approval rate {self.approval_rate:.3f}",
            "",
            "| Group | n | Accuracy | TPR (good approved) | FPR (bad approved) |",
            "|---|---|---|---|---|",
        ]
        for sg in self.subgroups:
            lines.append(
                f"| {sg.group} | {sg.n} | {sg.accuracy:.3f} | {sg.tpr:.3f} | {sg.fpr:.3f} |"
            )
        return "\n".join(lines) + "\n"


def _ks_statistic(labels: np.ndarray, scores: np.ndarray) -> float:
    """Kolmogorov-Smirnov separation between good and bad score distributions."""
    good = scores[labels == 0]
    bad = scores[labels == 1]
    if len(good) == 0 or len(bad) == 0:
        return 0.0
    all_scores = np.sort(np.concatenate([good, bad]))
    cdf_good = np.searchsorted(np.sort(good), all_scores, side="right") / len(good)
    cdf_bad = np.searchsorted(np.sort(bad), all_scores, side="right") / len(bad)
    return float(np.max(np.abs(cdf_good - cdf_bad)))


def performance_report(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    groups: pd.Series | np.ndarray | None,
    dataset: str,
) -> PerformanceReport:
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=float)
    approved = scores < threshold
    predicted_default = (~approved).astype(int)

    subgroups: list[SubgroupMetrics] = []
    if groups is not None:
        for g in pd.unique(np.asarray(groups)):
            subgroups.append(subgroup_metrics(groups, approved, labels, str(g)))

    return PerformanceReport(
        dataset=dataset,
        n=len(labels),
        auc=float(roc_auc_score(labels, scores)) if len(np.unique(labels)) == 2 else 0.0,
        brier=float(brier_score_loss(labels, scores)),
        ks=_ks_statistic(labels, scores),
        balanced_accuracy=float(balanced_accuracy_score(labels, predicted_default)),
        approval_rate=float(approved.mean()),
        subgroups=subgroups,
    )
