# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-08-23

Initial release: executable reference implementation of the FEG (Fairness,
Explainability, Governance) framework from Oduro-Gyan et al. (2025) for KYC/FinTech
decisioning.

### Added

- **Data layer** — deterministic synthetic KYC generator with documented historical label
  bias and proxy correlation; vectorized schema validation; quality gates (nulls,
  duplicates, class balance, protected-group representativeness).
- **Fairness layer** — disparate impact, statistical parity, equal opportunity, average
  odds and per-subgroup metrics; Kamiran–Calders reweighing; policy-aware group-threshold
  post-processing; hard gates driven by `configs/fairness.yaml`.
- **Modeling layer** — calibrated logistic-regression champion vs histogram-gradient-
  boosting challenger; subgroup-stratified evaluation; file-based model registry with
  content hashes, lineage, approval workflow and integrity verification.
- **Explainability layer** — SHAP attributions (permutation) with a numpy fallback
  adapter; versioned global-importance artifacts; adverse-action reason codes; actionable
  counterfactual search honoring feature mutability and bounds.
- **Governance layer** — append-only SHA-256 hash-chained audit trail with tamper
  detection; automatic model cards; policy engine used by pipeline and CI.
- **Monitoring layer** — PSI / Jensen–Shannon drift, fairness drift vs registration
  baseline, warning/critical alerts, audited monitoring runs.
- **Pipeline CLI** — `feg train | monitor | audit-verify | approve` with fail-closed
  semantics (gate failure blocks registration and exits non-zero).
- **Decision API** — FastAPI service: banded decisions (approve/refer/decline), reason
  codes, counterfactuals, human review queue with audited resolution, model registry
  endpoints, health/ready probes.
- **Engineering** — pytest suite (58 tests, 92% coverage, 85% gate), ruff, mypy strict,
  pre-commit, GitHub Actions CI (lint/type/test/audit/build/docker + dedicated governance
  job), release workflow (tag → sdist/wheel + GHCR image + GitHub Release), multi-stage
  non-root Dockerfile and docker-compose demo, ADRs, architecture and API documentation.
