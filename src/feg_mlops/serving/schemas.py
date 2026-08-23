"""API request/response schemas.

``FeatureValues`` mirrors the canonical feature schema in
``feg_mlops.data.schema`` (bounds and integer-ness included) so the OpenAPI
contract and the model's input contract stay aligned; a unit test guards
the two definitions against drift.
"""

from __future__ import annotations

import pandas as pd
from pydantic import BaseModel, Field

from feg_mlops.data.schema import FEATURE_SPECS


class FeatureValues(BaseModel):
    income_annual: float = Field(ge=3_000, le=1_000_000)
    debt_to_income: float = Field(ge=0, le=1)
    credit_history_months: float = Field(ge=0, le=720)
    num_open_accounts: int = Field(ge=0, le=30)
    num_delinquencies_24m: int = Field(ge=0, le=12)
    employment_tenure_months: float = Field(ge=0, le=600)
    months_since_last_delinquency: float = Field(ge=0, le=240)
    kyc_docs_verified: int = Field(ge=0, le=1)
    device_trust_score: float = Field(ge=0, le=1)
    ip_country_risk: float = Field(ge=0, le=1)
    velocity_apps_90d: int = Field(ge=0, le=20)
    avg_balance_6m: float = Field(ge=0, le=2_000_000)
    balance_volatility: float = Field(ge=0, le=1)
    pct_on_time_payments: float = Field(ge=0, le=1)
    credit_utilization: float = Field(ge=0, le=1)
    has_previous_default: int = Field(ge=0, le=1)


def feature_values_matches_schema() -> bool:
    """True when FeatureValues fields exactly mirror FEATURE_SPECS."""
    fields = FeatureValues.model_fields
    if set(fields) != {spec.name for spec in FEATURE_SPECS}:
        return False
    for spec in FEATURE_SPECS:
        info = fields[spec.name]
        ge = next((m.ge for m in info.metadata if hasattr(m, "ge")), None)
        le = next((m.le for m in info.metadata if hasattr(m, "le")), None)
        if ge != spec.lo or le != spec.hi:
            return False
        expected_type = int if spec.integer else float
        if info.annotation is not expected_type:
            return False
    return True


class DecisionRequest(BaseModel):
    features: FeatureValues
    applicant_ref: str | None = Field(
        default=None, max_length=64, description="Optional applicant reference (not PII)"
    )


class ReasonCodeOut(BaseModel):
    code: str
    text: str
    contribution: float


class CounterfactualOut(BaseModel):
    found: bool
    narrative: str


class DecisionResponse(BaseModel):
    decision_id: str
    decision: str = Field(description="approve | refer | decline")
    risk_score: float
    model_version: int
    model_name: str
    reason_codes: list[ReasonCodeOut]
    counterfactual: CounterfactualOut | None = None
    review_required: bool


class ReviewOut(BaseModel):
    review_id: str
    decision_id: str
    applicant_ref: str | None
    risk_score: float
    status: str  # open | approved | declined
    reason_codes: list[ReasonCodeOut]
    created_at: str


class ResolveReviewRequest(BaseModel):
    outcome: str = Field(pattern="^(approved|declined)$")
    reviewer: str = Field(max_length=64)
    note: str = Field(default="", max_length=500)


class ModelVersionOut(BaseModel):
    name: str
    version: int
    status: str
    created_at: str
    metrics: dict[str, object]
    fairness: dict[str, object]


def feature_frame(features: BaseModel) -> pd.DataFrame:
    return pd.DataFrame([features.model_dump()])
