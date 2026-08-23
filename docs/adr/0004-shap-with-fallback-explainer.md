# ADR 0004 — SHAP as the primary explainer behind an adapter, with a numpy fallback

- Status: Accepted
- Date: 2026-08-23

## Context

The paper names SHAP explicitly. SHAP is the de-facto standard but is a compiled, fast-moving
dependency that can lag new Python versions; a reference implementation should not
hard-fail where SHAP cannot install.

## Decision

All consumers depend on the `RiskScoreFn` + explainer interface
(`feg_mlops.explain.explainer`). `build_explainer()` returns `ShapExplainer`
(permutation-SHAP over a 256-row background, seeded for determinism, progress bars
silenced) and falls back to `MarginalAblationExplainer` — a dependency-free marginal-effect
attribution with the same sign convention — when SHAP is unavailable.

## Consequences

- Pipelines and the serving layer produce explanations in every environment; the engine in
  use is recorded in audit events (`explainability.artifacts_generated.engine`).
- The fallback is not Shapley-exact; it is documented as such and only engages without SHAP.
- Runtime explanations at serving time reuse the background sample embedded in the
  registered model payload, keeping training/serving attribution semantics consistent.
