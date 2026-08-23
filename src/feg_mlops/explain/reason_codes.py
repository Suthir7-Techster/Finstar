"""Adverse-action style reason codes derived from attributions.

Regulators and customers need human-readable reasons for unfavorable
decisions (paper Section 3.4: stakeholder-sensitive explanations). This
module translates attribution vectors into plain-language reason codes,
reporting only factors that pushed the decision toward higher risk.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

REASON_TEMPLATES: dict[str, str] = {
    "credit_utilization": "Credit utilization is high compared with approved applicants",
    "debt_to_income": "Debt-to-income ratio is elevated relative to approved applicants",
    "pct_on_time_payments": (
        "Share of payments made on time is below the approved-applicant benchmark"
    ),
    "num_delinquencies_24m": "Recent delinquencies in the last 24 months",
    "has_previous_default": "A previous default is on record",
    "income_annual": "Reported income is low relative to requested credit terms",
    "avg_balance_6m": "Average account balance over the last 6 months is low",
    "credit_history_months": ("Credit history is shorter than the approved-applicant benchmark"),
    "employment_tenure_months": (
        "Employment tenure is shorter than the approved-applicant benchmark"
    ),
    "months_since_last_delinquency": "Most recent delinquency occurred recently",
    "velocity_apps_90d": "High number of credit applications in the last 90 days",
    "ip_country_risk": "Application originated from a higher-risk network/location profile",
    "device_trust_score": "Device trust signals for this application were weak",
    "kyc_docs_verified": "Identity documentation could not be fully verified",
    "balance_volatility": "Account balance volatility is high",
    "num_open_accounts": "Number of open credit accounts",
}

FALLBACK_TEMPLATE = "Factor `{feature}` contributed negatively to this decision"


@dataclass(frozen=True)
class ReasonCode:
    code: str  # feature name
    text: str
    contribution: float  # attribution to predicted risk (positive = riskier)
    feature_value: float


def reason_codes(
    row: pd.Series,
    attributions: dict[str, float],
    max_codes: int = 3,
) -> list[ReasonCode]:
    """Top factors that pushed predicted risk upward, human-readable."""
    risk_factors = sorted(
        ((feat, float(a)) for feat, a in attributions.items() if a > 0),
        key=lambda kv: kv[1],
        reverse=True,
    )
    codes: list[ReasonCode] = []
    for feat, attribution in risk_factors[:max_codes]:
        template = REASON_TEMPLATES.get(feat, FALLBACK_TEMPLATE.replace("{feature}", feat))
        value = float(row[feat]) if feat in row.index else float("nan")
        codes.append(
            ReasonCode(
                code=feat,
                text=template,
                contribution=round(attribution, 4),
                feature_value=value,
            )
        )
    return codes
