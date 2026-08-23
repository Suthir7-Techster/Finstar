"""End-to-end integration tests (smoke-sized pipeline run).

The expensive artifact (a trained + registered model) is produced once per
session and shared across these tests.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from feg_mlops.governance.audit import AuditTrail
from feg_mlops.modeling.registry import ModelRegistry
from feg_mlops.pipeline.cli import main as cli_main
from feg_mlops.pipeline.run_monitoring import run_monitoring_pipeline
from feg_mlops.pipeline.run_training import PipelineError, run_training_pipeline
from feg_mlops.serving.app import create_app

pytestmark = pytest.mark.integration

GOOD = {
    "income_annual": 95000,
    "debt_to_income": 0.12,
    "credit_history_months": 220,
    "num_open_accounts": 8,
    "num_delinquencies_24m": 0,
    "employment_tenure_months": 120,
    "months_since_last_delinquency": 240,
    "kyc_docs_verified": 1,
    "device_trust_score": 0.9,
    "ip_country_risk": 0.05,
    "velocity_apps_90d": 0,
    "avg_balance_6m": 9000,
    "balance_volatility": 0.1,
    "pct_on_time_payments": 0.99,
    "credit_utilization": 0.1,
    "has_previous_default": 0,
}
RISKY = {
    "income_annual": 24000,
    "debt_to_income": 0.55,
    "credit_history_months": 24,
    "num_open_accounts": 12,
    "num_delinquencies_24m": 3,
    "employment_tenure_months": 8,
    "months_since_last_delinquency": 2,
    "kyc_docs_verified": 0,
    "device_trust_score": 0.3,
    "ip_country_risk": 0.7,
    "velocity_apps_90d": 6,
    "avg_balance_6m": 300,
    "balance_volatility": 0.8,
    "pct_on_time_payments": 0.68,
    "credit_utilization": 0.93,
    "has_previous_default": 1,
}
BORDERLINE = dict(GOOD, credit_utilization=0.55, pct_on_time_payments=0.82, debt_to_income=0.4)


@pytest.fixture(scope="session")
def trained(tmp_path_factory, config_dir):
    """One smoke pipeline run shared by all integration tests."""
    artifacts = tmp_path_factory.mktemp("artifacts")
    outcome = run_training_pipeline(config_dir=config_dir, artifacts_dir=artifacts, smoke=True)
    assert outcome.passed, outcome.summary
    return artifacts, outcome


@pytest.fixture(scope="session")
def client(trained, config_dir):
    artifacts, _ = trained
    return TestClient(create_app(config_dir=config_dir, artifacts_dir=artifacts))


# ----------------------------------------------------------------------
# Training pipeline


def test_pipeline_detects_and_mitigates_bias(trained):
    _, outcome = trained
    # The unmitigated candidate must fail the four-fifths gate ...
    assert outcome.unmitigated_fairness is not None
    assert outcome.unmitigated_fairness.disparate_impact_ratio < 0.80
    # ... and the registered (mitigated) model must clear it.
    registered_di = outcome.champion_evaluation.fairness.disparate_impact_ratio
    assert registered_di >= 0.80
    assert "reweighing" in outcome.mitigation_applied
    assert outcome.champion_evaluation.auc >= 0.68


def test_registry_artifacts_complete(trained):
    artifacts, outcome = trained
    registry = ModelRegistry(artifacts / "registry")
    model = registry.get("kyc_risk_model", outcome.registered.version)
    version_dir = model.manifest_path.parent
    for artifact in (
        "model.joblib",
        "manifest.json",
        "model_card.md",
        "fairness_report.md",
        "fairness_report.json",
        "performance_report.md",
        "global_importance.json",
    ):
        assert (version_dir / artifact).exists(), artifact
    assert registry.verify_integrity("kyc_risk_model", model.version)
    manifest = model.manifest
    assert len(manifest["data_hash"]) == 64
    assert manifest["gates"]["passed"] is True
    card = (version_dir / "model_card.md").read_text(encoding="utf-8")
    assert "Disparate impact ratio" in card and "Limitations" in card


def test_audit_trail_covers_the_pipeline(trained):
    artifacts, _ = trained
    trail = AuditTrail(artifacts / "audit" / "audit_trail.jsonl")
    verification = trail.verify()
    assert verification.valid and verification.records >= 8
    event_types = {r["event_type"] for r in trail.records()}
    assert {
        "pipeline.start",
        "data.generated",
        "quality.checked",
        "fairness.evaluated",
        "fairness.mitigation_started",
        "explainability.artifacts_generated",
        "model.registered",
        "pipeline.finished",
    } <= event_types


def test_pipeline_blocks_when_policy_impossible(tmp_path, modified_config_dir):
    strict_dir = modified_config_dir({"fairness": {"gates": {"disparate_impact_min": 0.99}}})
    outcome = run_training_pipeline(
        config_dir=strict_dir, artifacts_dir=tmp_path / "strict", smoke=True
    )
    assert not outcome.passed
    assert outcome.registered is None
    assert outcome.summary["status"] == "blocked"


def test_pipeline_fails_closed_on_bad_quality(tmp_path, modified_config_dir):
    # Note: smoke=False so the tiny dataset sizes are not overridden.
    tiny_dir = modified_config_dir({"data": {"data": {"n_train": 60, "n_eval": 30}}})
    with pytest.raises(PipelineError, match="quality gate failed"):
        run_training_pipeline(config_dir=tiny_dir, artifacts_dir=tmp_path / "tiny", smoke=False)


def test_cli_audit_verify_exits_zero(trained, capsys):
    artifacts, _ = trained
    code = cli_main(["audit-verify", "--artifacts-dir", str(artifacts)])
    assert code == 0
    assert "hash chain intact" in capsys.readouterr().out


# ----------------------------------------------------------------------
# Monitoring


def test_monitoring_flags_severe_window_and_not_mild(trained, config_dir):
    artifacts, _ = trained
    outcome = run_monitoring_pipeline(config_dir=config_dir, artifacts_dir=artifacts)
    by_scenario = {w.scenario: w for w in outcome.windows}
    assert "mild" in by_scenario and "severe" in by_scenario

    mild_alerts = by_scenario["mild"].alert_dicts
    severe_alerts = by_scenario["severe"].alert_dicts
    assert not any(a["severity"] == "critical" for a in mild_alerts)
    assert any(
        a["severity"] == "critical" and a["component"] == "data-drift" for a in severe_alerts
    )

    severe_fairness = by_scenario["severe"].fairness_drift_dict
    assert severe_fairness["level"] in ("warning", "critical")


# ----------------------------------------------------------------------
# Decision API


def test_approve_decline_refer_bands(client):
    approve = client.post("/v1/decisions", json={"features": GOOD})
    assert approve.status_code == 200
    body = approve.json()
    assert body["decision"] == "approve"
    assert body["review_required"] is False
    assert body["decision_id"].startswith("DEC-")

    decline = client.post("/v1/decisions", json={"features": RISKY}).json()
    assert decline["decision"] == "decline"
    assert decline["reason_codes"], "declines must carry adverse-action reasons"
    assert decline["counterfactual"] is not None

    refer = client.post(
        "/v1/decisions", json={"features": BORDERLINE, "applicant_ref": "APP-1"}
    ).json()
    assert refer["decision"] == "refer"
    assert refer["review_required"] is True


def test_review_queue_lifecycle(client):
    before = len(client.get("/v1/reviews", params={"status": "open"}).json())
    client.post("/v1/decisions", json={"features": BORDERLINE, "applicant_ref": "APP-2"})
    open_reviews = client.get("/v1/reviews", params={"status": "open"}).json()
    assert len(open_reviews) == before + 1

    review = open_reviews[-1]
    resolved = client.post(
        f"/v1/reviews/{review['review_id']}/resolve",
        json={"outcome": "declined", "reviewer": "uw-test", "note": "income unverifiable"},
    )
    assert resolved.status_code == 200
    assert resolved.json()["status"] == "declined"

    again = client.post(
        f"/v1/reviews/{review['review_id']}/resolve",
        json={"outcome": "approved", "reviewer": "uw-test", "note": "retry"},
    )
    assert again.status_code == 409  # already resolved

    missing = client.post(
        "/v1/reviews/REV-nope/resolve",
        json={"outcome": "approved", "reviewer": "uw", "note": ""},
    )
    assert missing.status_code == 404


def test_validation_and_model_endpoints(client):
    invalid = dict(GOOD, credit_utilization=9.0)
    assert client.post("/v1/decisions", json={"features": invalid}).status_code == 422
    assert client.post("/v1/decisions", json={"features": {}}).status_code == 422

    assert client.get("/healthz").status_code == 200
    ready = client.get("/readyz")
    assert ready.status_code == 200 and ready.json()["status"] == "ready"

    models = client.get("/v1/models").json()
    assert models and models[0]["fairness"]["disparate_impact_ratio"] >= 0.80

    detail = client.get(f"/v1/models/{models[0]['version']}")
    assert detail.status_code == 200
    assert "model_card" in detail.json()
    assert client.get("/v1/models/999").status_code == 404


def test_decisions_are_audited_with_feature_hashes(trained, config_dir):
    artifacts, _ = trained
    client = TestClient(create_app(config_dir=config_dir, artifacts_dir=artifacts))
    client.post("/v1/decisions", json={"features": GOOD})
    trail = AuditTrail(artifacts / "audit" / "audit_trail.jsonl")
    decision_events = [r for r in trail.records() if r["event_type"] == "decision.made"]
    assert decision_events
    last = decision_events[-1]["payload"]
    assert len(last["features_sha256"]) == 64
    assert "income_annual" not in last  # raw features are never logged
