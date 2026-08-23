"""API schema tests, including the schema-drift guard."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from feg_mlops.serving.schemas import FeatureValues, feature_values_matches_schema

VALID = dict(
    income_annual=50000.0,
    debt_to_income=0.3,
    credit_history_months=120.0,
    num_open_accounts=6,
    num_delinquencies_24m=0,
    employment_tenure_months=60.0,
    months_since_last_delinquency=240.0,
    kyc_docs_verified=1,
    device_trust_score=0.8,
    ip_country_risk=0.1,
    velocity_apps_90d=1,
    avg_balance_6m=4000.0,
    balance_volatility=0.3,
    pct_on_time_payments=0.9,
    credit_utilization=0.4,
    has_previous_default=0,
)


def test_feature_values_mirrors_canonical_schema():
    """Guards against FeatureValues drifting from data.schema.FEATURE_SPECS."""
    assert feature_values_matches_schema(), (
        "FeatureValues is out of sync with feg_mlops.data.schema.FEATURE_SPECS"
    )


def test_valid_features_accepted():
    model = FeatureValues(**VALID)
    dumped = model.model_dump()
    assert dumped["credit_utilization"] == 0.4
    assert dumped["num_open_accounts"] == 6


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("credit_utilization", 1.5),  # above hi
        ("income_annual", 100.0),  # below lo
        ("num_delinquencies_24m", 99),
        ("kyc_docs_verified", 2),
    ],
)
def test_out_of_bounds_rejected(field: str, value: float):
    broken = dict(VALID, **{field: value})
    with pytest.raises(ValidationError):
        FeatureValues(**broken)


def test_missing_fields_rejected():
    partial = dict(VALID)
    del partial["credit_utilization"]
    with pytest.raises(ValidationError):
        FeatureValues(**partial)
