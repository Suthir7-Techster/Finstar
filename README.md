# feg-mlops

**An executable reference implementation of the Fairness–Explainability–Governance (FEG)
framework for responsible KYC/FinTech decisioning.**

Built from the paper:
> Oduro-Gyan, J., Eleweke, I., Ajuwon, S., Bello, A., & Arotayo, A.-L. (2025).
> *Embedding Responsible AI into MLOps Pipelines: Ensuring Fairness, Explainability, and
> Governance in KYC and FinTech Decisioning.* Journal of Scientific Research and Reports,
> 31(12), 563–581. [DOI: 10.9734/jsrr/2025/v31i123797](https://doi.org/10.9734/jsrr/2025/v31i123797)

The paper's thesis: responsible-AI principles exist as documents and point-in-time audits,
but are **not wired into MLOps pipelines** — and no working implementation exists to
validate the idea. This repository is that implementation: a KYC credit-risk model
surrounded by fairness gates, SHAP explainability, a tamper-evident audit trail, drift
monitoring, and a decision API with human-in-the-loop review — where **a model that fails
fairness policy is never registered**, in training, in CI, and at serving time.

## What a `feg train` run proves

From a seeded synthetic KYC dataset with documented historical label bias (see
[ADR 0001](docs/adr/0001-synthetic-data-with-injected-bias.md)):

| Evidence | Unmitigated candidate | Registered model |
|---|---|---|
| Disparate impact ratio (policy ≥ 0.80) | **0.572 — gate FAILS** | **0.878 — passes** |
| AUC (ground-truth outcomes) | 0.830 | 0.830 |
| Mitigation | — | reweighing (Kamiran & Calders 2012) |

The pipeline refuses to register the unmitigated model, applies mitigation automatically,
re-evaluates, and only then registers it with lineage (data hash, git SHA, fairness
evidence, model card). Monitoring then detects drift: a severely shifted production window
raises **critical data-drift and fairness-drift alerts**, a mild window raises none, and
`feg audit-verify` confirms the hash-chained audit trail is intact. Every one of these
claims is pinned by the test suite.

## Architecture

```mermaid
flowchart LR
    subgraph training["feg train"]
        GEN[Synthetic KYC data\n(seed + documented bias)] --> QC[Quality gates]
        QC --> TR[Calibrated LogReg + HistGBM]
        TR --> FG{Fairness gates}
        FG -- fail --> MIT[Auto-mitigation\nreweighing + thresholds] --> FG
        FG -- pass --> XAI[SHAP artifacts] --> REG[(Model registry\n+ model card + lineage)]
    end
    subgraph serving["feg serve"]
        API[POST /v1/decisions] --> BAND{Band}
        BAND -- refer --> REV[Human review queue]
        BAND -- decline --> RC[Reason codes + counterfactual]
    end
    subgraph monitoring["feg monitor"]
        DRIFT[PSI / JS / KS drift] --> ALERT[Alerts]
        FD[Fairness drift vs registry baseline] --> ALERT
    end
    REG --> API
    REG --> DRIFT
    training --> AUD[(Hash-chained audit trail)]
    serving --> AUD
    monitoring --> AUD
```

Full component description and production-hardening notes:
[docs/architecture.md](docs/architecture.md).

## Quickstart

```bash
python -m pip install -e ".[dev]"
make demo          # train under fairness gates -> monitor -> verify audit chain
make serve         # decision API on :8000 (Swagger UI at /docs)
```

Try a decision:

```bash
curl -s http://localhost:8000/v1/decisions -H 'Content-Type: application/json' -d '{
  "features": {
    "income_annual": 48000, "debt_to_income": 0.42, "credit_history_months": 130,
    "num_open_accounts": 6, "num_delinquencies_24m": 1, "employment_tenure_months": 50,
    "months_since_last_delinquency": 8, "kyc_docs_verified": 1, "device_trust_score": 0.6,
    "ip_country_risk": 0.3, "velocity_apps_90d": 2, "avg_balance_6m": 2500,
    "balance_volatility": 0.45, "pct_on_time_payments": 0.78, "credit_utilization": 0.8,
    "has_previous_default": 0
  }
}'
```

Declines come back with plain-language reason codes and an actionable counterfactual
(`risk 0.94 → 0.46 by reducing credit utilization to 0.30 and on-time payments to 0.96`).
Full API reference: [docs/api.md](docs/api.md).

### Docker

```bash
docker compose up      # trains under gates, then serves the API on :8000
```

## The governance model

| Control | Mechanism | Fails closed when |
|---|---|---|
| Data quality | Schema, ranges, nulls, class/group balance gates | Data violates policy (`PipelineError`) |
| Fairness | DI ≥ 0.80, \|SPD\| ≤ 0.10, \|EOD\| ≤ 0.10, subgroup recall/count floors | Model can't pass even after mitigation → not registered, exit 1 |
| Explainability | SHAP global artifacts versioned per model; per-decision reason codes + counterfactuals | Serving declines without reasons is unrepresentable |
| Traceability | SHA-256 hash-chained audit trail; registry lineage (data hash, git SHA, approvals) | `feg audit-verify` exits non-zero on tampering |
| Human oversight | `refer` band → review queue; `candidate → approved` sign-off before serving preference | Unreviewed borderline decisions never auto-finalize |
| Monitoring | PSI / Jensen–Shannon drift + fairness drift vs registration baseline | Alert levels wired to policy thresholds |

All thresholds are **policy documents** (`configs/*.yaml`) validated by pydantic — changing
policy never touches code. The same gates run in CI
([governance job](.github/workflows/ci.yml)), so "a discriminatory model fails the build"
is literal.

## Problems and solutions

The derived problem→solution analysis, each mapped to paper section, implementation module
and verifying test: **[docs/problems-and-solutions.md](docs/problems-and-solutions.md)**.

## Project structure

```
configs/                  policy-as-code (fairness gates, bands, drift thresholds)
src/feg_mlops/
  config/                 typed YAML loading + validation
  data/                   synthetic KYC generator, schema, quality gates
  fairness/               group metrics, reweighing, threshold post-processing, gates
  modeling/               candidate training, subgroup evaluation, model registry
  explain/                SHAP adapter + fallback, counterfactuals, reason codes
  governance/             hash-chained audit trail, model cards, policy engine
  monitoring/             PSI/JS/KS drift, fairness drift, alerts
  pipeline/               CLI: train / monitor / audit-verify / approve
  serving/                FastAPI decision service + review queue
tests/                    unit + integration (coverage-gated at 85%)
docs/                     architecture, API, problems-and-solutions, ADRs
```

## Development

```bash
make setup        # install + pre-commit hooks
make lint         # ruff format + lint
make typecheck    # mypy --strict
make coverage     # pytest with coverage gate (>=85%)
```

- Python ≥ 3.11 (CI matrix: 3.11, 3.14)
- [CHANGELOG](CHANGELOG.md) · [Contributing](CONTRIBUTING.md) · [Security](SECURITY.md)
- Decisions record: [docs/adr/](docs/adr/)

## License

[MIT](LICENSE)
