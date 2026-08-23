# Model Cards in feg-mlops

Model cards here are **generated, not hand-written**: every registered version gets one at
`artifacts/registry/kyc_risk_model/vN/model_card.md` (generator:
`feg_mlops/governance/model_card.py`), and the API serves it via `GET /v1/models/{version}`.

Structure of a generated card:

1. **Header** — name, version, status (candidate/approved), creation time, package
   version, git commit.
2. **Intended use** — the deployment scope, including the human-review requirement for
   the refer band.
3. **Training data** — dataset provenance (synthetic, seeded), data hash, mitigation
   applied.
4. **Quantitative analysis** — AUC, balanced accuracy, Brier score, approval rate.
5. **Fairness** — disparate impact, statistical parity, equal opportunity, and the full
   per-group table (n, selection rate, TPR, FPR, accuracy).
6. **Top risk drivers** — mean |attribution| per feature (from the versioned SHAP
   artifacts).
7. **Limitations and ethical considerations** — synthetic-data caveat, single-attribute
   monitoring scope, drift/retraining obligations.
8. **Approvals** — named human sign-offs with reasons and timestamps.

The intent follows Mitchell et al. (2019), *Model Cards for Model Reporting*, and provides
the evidence trail expected by EU AI Act Article 11-style technical documentation — see
[ADR 0003](0003-file-based-registry-and-audit-trail.md) for the registry design that keeps
cards tamper-evident (hashes recorded in the version manifest).
