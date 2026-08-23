# Security Policy

## Supported versions

Pre-1.0: security fixes apply to the latest `main` and the most recent tagged release.

## Reporting a vulnerability

Please report vulnerabilities privately via GitHub security advisories
("Report a vulnerability" under the Security tab) rather than public issues. Include
reproduction steps and affected components. Expect an acknowledgement within 7 days.

## Security model summary

- **Audit integrity**: the audit trail is an append-only JSONL with a SHA-256 hash chain;
  `feg audit-verify` detects retroactive edits or deletions.
- **Model integrity**: registry manifests pin each model file's SHA-256;
  `ModelRegistry.verify_integrity()` re-derives it on demand.
- **Privacy**: production decision events log a feature-vector content hash and reason-code
  names only — never raw applicant data.
- **Container**: images run as a non-root user with a healthcheck; the base image is a
  pinned `python:*-slim` digest.
- **Supply chain**: `pip-audit` runs in CI on every push and PR.

## Known limitations (reference implementation)

- The file audit trail is single-process (thread lock); multi-worker deployments must back
  it with a database preserving the same hash-chain semantics.
- The decision API ships without authentication; in production it must sit behind an
  authenticating gateway with RBAC mapping to the approval/review roles.
