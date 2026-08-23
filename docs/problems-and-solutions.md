# Problems Derived from the Paper, and How This Repository Solves Them

Source: Oduro-Gyan, J., Eleweke, I., Ajuwon, S., Bello, A., & Arotayo, A.-L. (2025).
*Embedding Responsible AI into MLOps Pipelines: Ensuring Fairness, Explainability, and Governance in KYC and FinTech Decisioning.* Journal of Scientific Research and Reports, 31(12), 563–581. DOI: [10.9734/jsrr/2025/v31i123797](https://doi.org/10.9734/jsrr/2025/v31i123797)

The paper is an integrative review of 25 studies (2015–2025). Its central claim: fairness,
explainability and governance (FEG) exist as ethics documents and point-in-time audits, but
are **not operationalized inside MLOps pipelines** — and no working reference implementation
exists to validate the idea. This repository takes the paper as a requirements document and
answers each identified problem with executable, tested code.

> **Reading guide:** each problem below cites where the paper documents it, states the
> solution implemented here, and names the artifact/test that proves the solution works.
> Problems P1–P7 come from the paper's findings; P8–P9 are weaknesses of the paper itself,
> addressed so this implementation does not inherit them.

---

## P1 — Responsible-AI principles are not operationalized in pipelines

**Paper evidence:** Section 1.2 ("most FinTech organizations lack a broad-based approach to
integrating FEG throughout the MLOps lifecycle"); Section 4.3 (governance limited to
high-level ethics principles, disconnected from CI/CD).

**Solution:** Policy-as-code. Every gate — fairness limits, performance floors, drift alert
levels, decision bands — lives in `configs/*.yaml`, is validated by pydantic at load time
(`feg_mlops/config`), and is enforced *as code* at three points: the training pipeline
(exits non-zero on violation, model is never registered), CI (the `governance` job runs
`feg train --smoke`, so a discriminatory model fails the build), and serving (bands and
sign-off state come from the registered policy snapshot).

**Verified by:** `tests/integration/test_pipeline_and_api.py::test_pipeline_blocks_when_policy_impossible`
(an impossible fairness policy blocks registration) and the
`governance` job in `.github/workflows/ci.yml`.

---

## P2 — Algorithmic bias in KYC/credit decisioning, transmitted by historical labels and proxies

**Paper evidence:** Section 3.2 (sources of algorithmic bias: unrepresentative data,
historical discrimination, proxy features); recommendation 1 (multi-stage bias detection,
fairness-aware preprocessing).

**Solution:** A deterministic synthetic KYC generator with *documented, configurable* bias
injection (historical label flips against the unprivileged age band + proxy correlation via
credit-history length). Bias is therefore reproducible on demand — the demo always has a
real bias to detect. Detection is multi-stage: data-quality gates check group
representativeness *before* training (`feg_mlops/data/quality.py`), and the fairness gate
engine measures disparate impact (four-fifths rule), statistical parity, equal opportunity
and per-subgroup recall *before* promotion (`feg_mlops/fairness`). Mitigation is automatic:
Kamiran–Calders reweighing, then policy-aware group-threshold post-processing; only a
mitigated model that passes all gates can be registered.

**Verified by:** the default full-size run: unmitigated DI 0.572 → registered DI 0.878 at
AUC 0.830 (`test_pipeline_detects_and_mitigates_bias` asserts unmitigated < 0.80 and
registered ≥ 0.80 on the smoke run); reweighing math is property-tested in
`test_mitigation.py` (weighted joint independence).

---

## P3 — Black-box opacity: customers, underwriters and regulators cannot interrogate decisions

**Paper evidence:** Section 3.4 (XAI rarely operationalized in live decisioning; SHAP, LIME
and counterfactuals named as the toolset); recommendation 2 (embed XAI into CI/CD).

**Solution:** Explainability-by-design at two levels. **Pipeline level:** every registered
model version stores SHAP global importance artifacts (JSON + Markdown) versioned beside the
model. **Decision level:** each API response carries adverse-action-style reason codes
mapped from SHAP attributions to plain language, and declines receive an *actionable*
counterfactual search restricted to mutable features within schema bounds
("risk 0.94 → 0.46 by reducing utilization to 0.30 and on-time payments to 0.96"). A SHAP
adapter with a numpy fallback keeps the pipeline functional where SHAP cannot install.

**Verified by:** `registry artifacts include global_importance.json`
(`test_registry_artifacts_complete`), reason-code ranking
(`test_reason_codes_rank_risk_factors`), counterfactual actionability constraints
(`test_counterfactual_finds_target_and_respects_constraints`).

---

## P4 — Missing traceability: no audit trails, lineage or versioning

**Paper evidence:** Section 4.3 (need for model registries, versioning, systematic logging,
audit trails); Sculley et al. (2015) technical-debt discussion cited as motivation.

**Solution:** A hash-chained, append-only audit trail (SHA-256 per record over the previous
record's hash) records every material event: data generation (with content hashes), quality
results, fairness evaluations, mitigation triggers, registrations, approvals, production
decisions and drift alerts. Any retroactive edit breaks the chain and is detected by
`feg audit-verify`. The file-based registry stores per version: model + data hashes, git
SHA, metrics, fairness evidence, model card and approvals — the lineage record EU AI Act
Article 11-style documentation expects.

**Verified by:** `test_audit_trail_covers_the_pipeline` (chain intact, all event types
present), `test_tamper_detection_value_edit` / `test_tamper_detection_line_deletion`
(edits and deletions detected), `test_decisions_are_audited_with_feature_hashes`
(raw features never logged — only a content hash, privacy by design).

---

## P5 — Post-deployment fairness drift goes undetected

**Paper evidence:** Section 3.3 and recommendation 1 (continuous bias monitoring, adaptive
retraining); Ramamoorthy et al. (2025) gap noted as "continuous governance feedback loops".

**Solution:** `feg monitor` recomputes the full fairness posture on production windows and
compares it against the *reference window recorded at registration* (lineage-based
baselines), alongside per-feature data drift: PSI (quantile-binned), Jensen–Shannon for
categoricals, two-sided warning/critical thresholds, structured alerts appended to the audit
trail. Because the data generator is seeded, the reference window is exactly reproducible.

**Verified by:** `test_monitoring_flags_severe_window_and_not_mild` — the severe scenario
raises critical data-drift and fairness alerts; the mild scenario raises none.

---

## P6 — No human oversight or recourse in automated decisioning

**Paper evidence:** Section 3.5 (human-in-the-loop review and appeal mechanisms);
recommendation 3 (role-based oversight); EU AI Act human-oversight expectations.

**Solution:** Decision banding routes borderline risk scores to a `refer` band that lands in
a human review queue; underwriters resolve via API with an accountable actor name, and the
resolution is audited. Model promotion has the same control: registry versions are
`candidate` until a named human approves them (`feg approve`), and serving prefers approved
versions.

**Verified by:** `test_review_queue_lifecycle` (queue → resolve → 409 on double-resolution)
and `test_registry...` approval workflow tests in `test_registry_and_bands.py`.

---

## P7 — Aggregate metrics hide subgroup harm

**Paper evidence:** Section 3.2 (subgroup-specific performance metrics, not just aggregate
scores); Holstein et al. (2019) practitioner findings.

**Solution:** All evaluation is subgroup-stratified by construction: `performance_report`
returns per-group accuracy/TPR/FPR/selection-rate tables, gates include per-group recall
floors and minimum group counts, and generated model cards publish the full subgroup table.

**Verified by:** the model card content check in `test_registry_artifacts_complete`; gate
tests in `test_gates_and_policy.py`.

---

## P8 — (Weakness of the paper) No empirical validation of the framework itself

**Paper evidence:** Section 5 — the authors' own limitation: "a significant lack of
empirical evidence from real-world, production-grade deployments", with pilot
implementations listed as future work.

**Solution:** This repository *is* the pilot: an executable FEG pipeline with measurable
before/after evidence (unmitigated vs mitigated fairness metrics, AUC cost of mitigation,
drift alerts on shifted windows), a test suite (58 tests, 92% coverage) that fails if the
guarantees regress, and CI that re-proves the whole story on every push. Absolute numbers
are illustrative (synthetic data), but the *mechanisms* are validated end-to-end — the
empirical gap the paper identifies for the field is closed at reference-implementation
scale.

---

## P9 — (Weakness of the paper) Reproducibility gaps: no spec, no code, no data

**Paper evidence:** the paper provides no formal framework specification (Fig. 1 is three
bullet boxes), no codebook, no dataset; Section 5 concedes two teams would implement very
different "FEG pipelines" from the same text.

**Solution:** Everything here is deterministic and inspectable: seeded synthetic data
(stable content hashes across runs), pinned dependency floors, strict type checking (mypy
--strict) and lint (ruff), architecture decision records for every material trade-off
(`docs/adr/`), an OpenAPI-specified API, and tests that pin the semantics (schema-drift
guard between the API contract and the feature schema, exact metric math on hand-computed
examples).

---

## Residual limitations (honest scope statement)

- **Synthetic data only** — bias injection is controlled, which is the point, but absolute
  metric values are not market benchmarks. The data layer is deliberately replaceable:
  `SyntheticKycGenerator` is the only module that knows how data originates.
- **Single protected attribute (age band)** — production systems must monitor every
  protected attribute in scope; the metrics module is attribute-agnostic and takes the
  attribute name from policy.
- **File-based registry and audit trail** — correct for a reference implementation; a
  multi-worker deployment should back the audit trail with a database and the registry
  with MLflow/Sagemaker/GCP-style tooling (see `docs/architecture.md`, production notes).
- **Counterfactuals are heuristic** — greedy search over quantile actions; DiCE-style
  optimization would produce tighter counterfactuals at higher cost.
