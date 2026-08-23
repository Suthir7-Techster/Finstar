# ADR 0003 — File-based model registry and audit trail (no external services)

- Status: Accepted (for the reference implementation)
- Date: 2026-08-23

## Context

Governance needs a model registry with lineage and a tamper-evident audit trail. MLflow,
cloud registries and ledger databases all provide these — at the cost of external
services, credentials and CI complexity.

## Decision

Implement both as plain local artifacts:

- **Registry**: `artifacts/registry/<name>/vN/` with `model.joblib`, `manifest.json`
  (version, status candidate→approved, model SHA-256, data hash, git SHA, metrics,
  fairness evidence, artifact hashes, approvals) plus generated reports and the model card.
- **Audit trail**: append-only JSONL where each record's hash covers the previous record's
  hash (SHA-256); `feg audit-verify` recomputes the chain.

## Consequences

- The whole story (`train → monitor → serve → audit`) runs anywhere Python runs, including
  CI and the Docker demo, with zero infrastructure.
- The audit trail is single-process (thread lock only); multi-worker deployments must front
  it with a database that preserves the same hash-chain semantics.
- The registry lacks cross-team access control; promotion to MLflow/cloud is a swap of the
  `ModelRegistry` class — the manifest schema is designed to map 1:1.
