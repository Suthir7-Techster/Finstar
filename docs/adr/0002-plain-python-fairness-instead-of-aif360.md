# ADR 0002 — Self-implemented fairness metrics and mitigation instead of AIF360

- Status: Accepted
- Date: 2026-08-23

## Context

The paper cites AI Fairness 360 (Bellamy et al., 2019) as the reference toolkit. AIF360
pulls a heavy dependency graph (older pandas/tensorboard constraints), installs poorly on
newest Python versions, and obscures the metric semantics behind its two-class dataset
abstraction.

## Decision

Implement the fairness layer directly in numpy/pandas (~150 lines): selection-rate-based
disparate impact and statistical parity, equal-opportunity and average-odds differences on
the *favorable* (approval) outcome, per-group performance tables; Kamiran–Calders
reweighing; policy-aware group-threshold post-processing. Metric math is pinned by
hand-computed unit tests.

## Consequences

- Zero extra dependencies; runs on Python 3.11–3.14; semantics are inspectable and
  testable (a governance requirement in its own right).
- We forgo AIF360's algorithm zoo (optimization-based pre-processing, reject-option
  classification). The mitigation config in `fairness.yaml` is the extension point.
- Cross-tooling note: definitions follow the standard AIF360/Hardt et al. conventions, so
  results are comparable with external audits.
