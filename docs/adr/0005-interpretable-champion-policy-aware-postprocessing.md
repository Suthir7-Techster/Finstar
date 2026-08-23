# ADR 0005 — Interpretable champion model and policy-aware threshold post-processing

- Status: Accepted
- Date: 2026-08-23

## Context

The paper argues (via Caruana et al., 2015) for inherently interpretable models in
high-stakes decisioning, while also noting post-hoc XAI and constraint-based mitigation.
Group-threshold post-processing usually equalizes opportunity *downward* (anchor the
privileged group, move the unprivileged threshold), which can violate subgroup-recall
floors after reweighing shifts score distributions.

## Decision

1. Two candidates compete: calibrated, standardized **logistic regression** (interpretable
   champion) and **histogram gradient boosting** (accuracy challenger). Selection policy =
   fairness gates first, then AUC.
2. The group-threshold optimizer is **policy-aware**: it targets the privileged group's
   TPR floored at `min_subgroup_recall`, then sets each group's threshold to the exact
   score quantile achieving that TPR — equal opportunity *and* the recall gate hold by
   construction.

## Consequences

- Mitigation cannot trade away the recall floor it was asked to respect; the property is
  unit-tested (`test_group_thresholds_equalize_tpr_and_respect_floor`).
- Equalizing at the higher TPR can raise approval volume (and FPR); the four-fifths gate
  and per-group FPR reporting keep that visible to reviewers.
- The champion stays fully interpretable: coefficients + SHAP reason codes tell one story.
