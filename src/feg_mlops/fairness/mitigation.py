"""Fairness mitigation: reweighing (pre-process) and group-threshold
optimization (post-process).

The paper (Section 4.2) prescribes re-weighting and constraint-based
mitigation across MLOps stages; these two techniques are the standard,
auditable building blocks (Kamiran & Calders 2012; Hardt et al. 2016).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def reweighing_weights(groups: pd.Series, labels: pd.Series) -> np.ndarray:
    """Kamiran & Calders (2012) reweighing.

    Returns per-row weights W(A=a, Y=y) = P(A=a) P(Y=y) / P(A=a, Y=y) that make
    the joint (group, label) distribution statistically independent, removing
    label bias against the unprivileged group before training.
    """
    groups = groups.reset_index(drop=True)
    labels = labels.reset_index(drop=True)
    joint = pd.DataFrame({"g": groups, "y": labels}).value_counts(normalize=True)
    p_group = groups.value_counts(normalize=True)
    p_label = labels.value_counts(normalize=True)

    weights = np.ones(len(groups), dtype=float)
    for i, (g, y) in enumerate(zip(groups, labels, strict=True)):
        p_joint = float(joint.get((g, y), 0.0))
        if p_joint > 0:
            weights[i] = float(p_group.get(g, 0.0)) * float(p_label.get(y, 0.0)) / p_joint
    return weights


def optimize_group_thresholds(
    groups: pd.Series,
    labels: pd.Series,
    scores: np.ndarray,
    base_threshold: float,
    min_recall: float = 0.0,
) -> dict[str, float]:
    """Post-processing: per-group thresholds that equalize opportunity.

    Targets a common true-positive rate (approval among truly good
    applicants) for every group: the privileged group's TPR at
    ``base_threshold``, floored at ``min_recall`` so the policy's
    subgroup-recall gate is met by construction. Each group's threshold is
    the exact score quantile achieving that TPR among its good applicants.
    Returns ``{group: threshold}``.
    """
    groups_arr = np.asarray(groups)
    labels_arr = np.asarray(labels)
    scores = np.asarray(scores, dtype=float)
    group_values = [str(g) for g in pd.unique(groups_arr)]

    thresholds: dict[str, float] = {}
    ref_tpr = 0.0
    for g in group_values:
        mask = groups_arr == g
        good_scores = scores[mask & (labels_arr == 0)]
        if len(good_scores) == 0:
            thresholds[g] = float(base_threshold)
            continue
        ref_tpr = max(ref_tpr, float((good_scores < base_threshold).mean()))

    target_tpr = max(ref_tpr, min_recall)
    for g in group_values:
        mask = groups_arr == g
        good_scores = scores[mask & (labels_arr == 0)]
        if len(good_scores) == 0:
            thresholds[g] = float(base_threshold)
            continue
        # P(good score < q_tau) = tau, so the tau-quantile approves exactly
        # tau of the group's good applicants.
        thresholds[g] = float(np.quantile(good_scores, target_tpr))
    return thresholds


def apply_group_thresholds(
    groups: pd.Series, scores: np.ndarray, thresholds: dict[str, float]
) -> np.ndarray:
    """Boolean approval decisions using per-group thresholds."""
    groups_arr = np.asarray(groups)
    scores = np.asarray(scores, dtype=float)
    approved = np.zeros(len(scores), dtype=bool)
    for g, thr in thresholds.items():
        approved[groups_arr == g] = scores[groups_arr == g] < thr
    return approved
