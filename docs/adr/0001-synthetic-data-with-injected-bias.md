# ADR 0001 — Synthetic KYC data with documented bias injection

- Status: Accepted
- Date: 2026-08-23

## Context

The source paper defines no dataset. Public credit datasets (German Credit, Lending Club)
are real but (a) require network downloads in CI, (b) have unknown, uncontrolled bias
structure, and (c) cannot demonstrate that our tooling detects *this* bias, because ground
truth about the bias is absent.

## Decision

Generate synthetic KYC applicants from a seeded numpy process whose **ground-truth** default
risk depends only on legitimate financial features, then inject the documented historical
bias into **training labels** (good→bad flips against the under-35 band) plus a proxy channel
(credit-history deficit). All parameters live in `configs/data.yaml`.

## Consequences

- Bias detection/mitigation demos are deterministic and reproducible; the unmitigated model
  fails the four-fifths gate by construction, and the mitigation effect is measurable
  against ground-truth labels.
- No licensing or privacy exposure; CI is fully offline.
- Absolute metric values are illustrative, not market benchmarks — stated in every model card.
- Production adoption replaces only `SyntheticKycGenerator`; everything downstream consumes
  the `DataFrame` contract (`applicant_id`, `age`, protected attribute, features, labels,
  ground truth).
