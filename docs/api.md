# Decision API

Base URL: `http://localhost:8000` (default bind). Interactive docs at `/docs` (Swagger UI).

## POST /v1/decisions

Score one applicant. Returns the decision band, risk score, adverse-action reason codes
and, for declines, an actionable counterfactual. Borderline scores are routed to the human
review queue.

```bash
curl -s http://localhost:8000/v1/decisions -H 'Content-Type: application/json' -d '{
  "applicant_ref": "APP-1001",
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

Response:

```json
{
  "decision_id": "DEC-9f2c1b6a0d4e5f70",
  "decision": "decline",
  "risk_score": 0.9412,
  "model_version": 1,
  "model_name": "kyc_risk_model",
  "reason_codes": [
    {"code": "credit_utilization", "text": "Credit utilization is high compared with approved applicants", "contribution": 0.21},
    {"code": "pct_on_time_payments", "text": "Share of payments made on time is below the approved-applicant benchmark", "contribution": 0.17},
    {"code": "debt_to_income", "text": "Debt-to-income ratio is elevated relative to approved applicants", "contribution": 0.09}
  ],
  "counterfactual": {
    "found": true,
    "narrative": "Original predicted risk 0.941; target ≤ 0.600. Reaching the target (risk → 0.460) is achievable by: credit_utilization: 0.800 → 0.304; pct_on_time_payments: 0.780 → 0.964"
  },
  "review_required": false
}
```

Errors: `422` when any feature is missing or out of schema bounds; `503` when no model is
registered (run `feg train` first).

## GET /v1/reviews?status=open

The human-review queue (decisions in the `refer` band).

```json
[{
  "review_id": "REV-3f9d2a1c4b5e",
  "decision_id": "DEC-1a2b3c4d5e6f7788",
  "applicant_ref": "APP-1002",
  "risk_score": 0.4311,
  "status": "open",
  "reason_codes": [{"code": "credit_utilization", "text": "...", "contribution": 0.08}],
  "created_at": "2026-08-23T12:00:00+00:00"
}]
```

## POST /v1/reviews/{review_id}/resolve

Underwriter resolution; the actor and outcome are appended to the audit trail.

```bash
curl -s -X POST http://localhost:8000/v1/reviews/REV-3f9d2a1c4b5e/resolve \
  -H 'Content-Type: application/json' \
  -d '{"outcome": "approved", "reviewer": "underwriter-7", "note": "income verified manually"}'
```

Errors: `404` unknown review; `409` already resolved.

## GET /v1/models — GET /v1/models/{version}

Registry listing with fairness evidence per version, and per-version manifest + model card.

## GET /healthz — GET /readyz

Liveness and readiness (`readyz` reports `503` until a model is loaded).

## Related CLI

```
feg train          # generate -> gates -> mitigate -> explain -> register
feg monitor        # drift + fairness drift scenarios on the deployed model
feg audit-verify   # verify the audit trail hash chain
feg approve --version 1 --actor NAME --reason TEXT   # human sign-off
```
