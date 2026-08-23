"""Model registry: versioning, approval, integrity; decision bands; config."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from feg_mlops.config import load_configs
from feg_mlops.modeling.registry import ModelRegistry, frame_hash
from feg_mlops.serving.decision import DecisionBands


class _FakeEstimator:
    def predict_proba(self, X):
        return [[0.5, 0.5]] * len(X)


def test_register_list_approve_and_integrity(tmp_path):
    registry = ModelRegistry(tmp_path / "registry")
    payload = {
        "estimator": _FakeEstimator(),
        "threshold": 0.4,
        "group_thresholds": None,
        "feature_columns": ["a"],
        "background": None,
    }
    v1 = registry.register(
        "m",
        payload,
        artifacts={"card.md": "# card"},
        metrics={"auc": 0.8},
        data_hash="abc",
    )
    v2 = registry.register("m", payload, metrics={"auc": 0.82}, data_hash="def")
    assert (v1.version, v2.version) == (1, 2)
    assert [m.version for m in registry.list_versions("m")] == [1, 2]
    assert registry.latest("m").version == 2

    # Integrity: hash recorded and verifiable.
    assert registry.verify_integrity("m", 1)

    # latest_approved falls back to newest candidate when none approved.
    assert registry.latest_approved("m").version == 2

    approved = registry.approve("m", 1, actor="risk-officer", reason="sign-off")
    assert approved.status == "approved"
    assert approved.manifest["approvals"][0]["actor"] == "risk-officer"
    assert registry.latest_approved("m").version == 1

    with pytest.raises(KeyError):
        registry.get("m", 99)
    with pytest.raises(KeyError):
        registry.get("missing", 1)


def test_artifact_hashes_recorded(tmp_path):
    registry = ModelRegistry(tmp_path / "registry")
    model = registry.register("m", {"estimator": None}, artifacts={"r.md": "hello"})
    entry = model.manifest["artifacts"]["r.md"]
    assert len(entry["sha256"]) == 64


def test_frame_hash_stable_and_order_sensitive(tmp_path):
    import pandas as pd

    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    assert frame_hash(df) == frame_hash(df.copy())
    assert frame_hash(df) != frame_hash(df.assign(a=[2, 1]))
    assert frame_hash(df) != frame_hash(df.assign(b=["x", "z"]))


class TestDecisionBands:
    bands = DecisionBands(approve_below=0.35, refer_low=0.35, refer_high=0.60, decline_above=0.60)

    def test_boundaries(self):
        assert self.bands.band(0.10) == "approve"
        assert self.bands.band(0.3499) == "approve"
        assert self.bands.band(0.35) == "refer"
        assert self.bands.band(0.5999) == "refer"
        assert self.bands.band(0.60) == "decline"
        assert self.bands.band(0.95) == "decline"


def test_config_validation_rejects_bad_policy(config_dir, modified_config_dir):
    bad_dir = modified_config_dir({"fairness": {"gates": {"disparate_impact_min": 1.5}}})
    with pytest.raises(ValidationError):
        load_configs(bad_dir)


def test_config_rejects_unordered_refer_band(config_dir, modified_config_dir):
    bad_dir = modified_config_dir({"model": {"decisions": {"refer_between": [0.6, 0.35]}}})
    with pytest.raises(ValidationError):
        load_configs(bad_dir)


def test_config_loads_and_defaults(config_dir):
    cfg = load_configs(config_dir)
    assert cfg.data.seed > 0
    assert cfg.fairness.gates.disparate_impact_min == pytest.approx(0.80)
    assert cfg.model_settings.decisions.max_reason_codes >= 1
    assert set(cfg.data_settings.drift_scenarios) >= {"mild", "severe"}
