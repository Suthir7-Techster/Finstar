# Architecture

feg-mlops operationalizes the FEG (Fairness, Explainability, Governance) framework from
Oduro-Gyan et al. (2025) as an executable MLOps pipeline around a KYC credit-risk model.
This document explains the component layout, data flow and artifact lifecycle, and what to
harden for production.

## Component map

```mermaid
flowchart LR
    subgraph policy["Policy as code (configs/*.yaml)"]
        PC[fairness gates / decision bands / drift thresholds]
    end

    subgraph training["feg train — FEG training pipeline"]
        GEN[Synthetic KYC generator\n(seed, injected bias)] --> QC[Quality gates]
        QC --> TR[Candidate training\nLogReg + HistGBM]
        TR --> FAIR[Fairness gates\nDI / SPD / EOD / subgroup floors]
        FAIR -- fail --> MIT[Automatic mitigation\nreweighing + thresholds]
        MIT --> FAIR2[Re-evaluate]
        FAIR2 --> XAI[SHAP artifacts]
        FAIR -- pass --> XAI
        XAI --> REG[(Model registry\n+ model card + lineage)]
    end

    subgraph serving["feg serve — decision API"]
        API[POST /v1/decisions] --> BAND{Band}
        BAND -- refer --> REV[Human review queue]
        BAND -- decline --> CF[Reason codes + counterfactual]
    end

    subgraph monitoring["feg monitor"]
        WIN[Drift windows] --> DRIFT[PSI / JS / KS]
        WIN --> FDRIFT[Fairness drift vs registry baseline]
        DRIFT --> ALERT[Alerts]
        FDRIFT --> ALERT
    end

    REG --> API
    REG --> monitoring
    training --> AUD[(Hash-chained audit trail)]
    serving --> AUD
    monitoring --> AUD
    PC -.->|gates| FAIR
    PC -.->|bands| BAND
    PC -.->|thresholds| ALERT
```

## Module layout

| Module | Responsibility |
|---|---|
| `feg_mlops.config` | Typed loading/validation of the four policy YAML documents |
| `feg_mlops.data` | Deterministic synthetic KYC generation, schema, quality gates |
| `feg_mlops.fairness` | Group metrics, reweighing, group-threshold post-processing, gates |
| `feg_mlops.modeling` | Candidate training, subgroup evaluation, model registry |
| `feg_mlops.explain` | SHAP adapter (+numpy fallback), counterfactuals, reason codes |
| `feg_mlops.governance` | Hash-chained audit trail, model cards, policy engine |
| `feg_mlops.monitoring` | PSI/JS/KS drift, fairness drift, alert evaluation |
| `feg_mlops.pipeline` | CLI orchestration (`feg train / monitor / audit-verify / approve`) |
| `feg_mlops.serving` | FastAPI decision engine, review queue, model endpoints |

## Data flow and lifecycle

1. **Generate** — seeded synthetic KYC applicants; training labels carry documented
   historical bias; the eval split carries ground truth, so mitigation effects are measured
   against true outcomes, not the biased process.
2. **Gate** — schema/range/null/balance/representativeness checks fail the run before any
   model sees bad data.
3. **Train** — calibrated logistic regression (interpretable champion) and histogram
   gradient boosting (challenger); operating threshold selected on an internal calibration
   split using observed labels only.
4. **Fairness gates** — disparate impact ≥ 0.80 (four-fifths), |SPD| and |EOD| ≤ 0.10,
   per-group recall and count floors. Failure triggers mitigation, then re-evaluation.
5. **Explain** — SHAP global importance on an eval sample; background sample embedded in
   the registered payload for runtime explanations.
6. **Register** — model + artifacts + lineage (data hash, git SHA, metrics, fairness
   evidence, approvals) under `artifacts/registry/kyc_risk_model/vN/`; model card
   generated; audit events appended at every step.
7. **Serve** — decisions banded approve/refer/decline from the registered policy snapshot;
   every decision audited (feature content hash, never raw values); refer band routed to
   the human review queue.
8. **Monitor** — drift scenarios scored with the registered model; per-feature PSI/JS plus
   fairness drift against the registration baseline; alerts audited.

## Security and integrity model

- **Audit trail**: SHA-256 hash chain makes retroactive edits detectable
  (`feg audit-verify` exits non-zero on tampering). Single-process with a thread lock;
  see production notes.
- **Model integrity**: registry manifests record the model file hash;
  `ModelRegistry.verify_integrity()` re-derives it on demand.
- **Privacy**: production decisions log a salted-free content hash of the feature vector
  and reason-code names only — never raw applicant features.
- **Supply chain**: CI runs `pip-audit`; images build from a pinned base with a non-root
  user and a healthcheck.

## Production hardening notes

This is a reference implementation: correct mechanisms, deliberately boring infrastructure.
For a production deployment, replace the marked components without touching policy logic:

| Replace | With | Why |
|---|---|---|
| File audit trail | Append-only DB table / QLDB-style ledger with the same hash chain | Multi-writer safety, retention, queryability |
| File model registry | MLflow / SageMaker / Vertex registry | Cross-team access, staging lifecycle |
| Synthetic generator | Feature store / warehouse adapters implementing the same `DataFrame` contract | Real data |
| JSONL review queue | Task queue (e.g. a tickets table with SLAs) | Concurrency, notifications |
| Process-local drift runs | Scheduled jobs writing metrics to Prometheus/Grafana | Continuous (not simulated) monitoring |
| Single protected attribute | All protected attributes in regulatory scope | Policy config takes the list as-is |

The policy layer (`configs/` + `feg_mlops.config` + `fairness.gates`) is intentionally
isolated so none of these replacements change governance semantics.
