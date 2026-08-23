"""KYC applicant record schema: feature inventory, bounds and constraints.

The schema is the single source of truth shared by data generation, schema
validation, the serving API, the counterfactual engine and monitoring. Bounds
double as plausibility checks for the data-quality gates and as search limits
for actionable counterfactuals.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

SCHEMA_VERSION = "1.0.0"

LABEL_COLUMN = "defaulted"  # 1 = applicant defaulted (unfavorable), 0 = good
PROTECTED_COLUMNS = ("age", "age_band")
ID_COLUMN = "applicant_id"


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    lo: float
    hi: float
    integer: bool = False
    # Counterfactual mutability: can the *applicant* change this feature through
    # a reasonable action? Immutable features never appear in counterfactuals.
    mutable: bool = True
    # Typical "improving" direction for counterfactual search: -1 means lower
    # values reduce predicted risk, +1 means higher values reduce it.
    improve_direction: int = -1


FEATURE_SPECS: tuple[FeatureSpec, ...] = (
    FeatureSpec("income_annual", 3_000.0, 1_000_000.0, improve_direction=-1),
    FeatureSpec("debt_to_income", 0.0, 1.0, improve_direction=-1),
    FeatureSpec("credit_history_months", 0.0, 720.0, mutable=False, improve_direction=-1),
    FeatureSpec("num_open_accounts", 0.0, 30.0, integer=True, improve_direction=1),
    FeatureSpec("num_delinquencies_24m", 0.0, 12.0, integer=True, improve_direction=-1),
    FeatureSpec("employment_tenure_months", 0.0, 600.0, improve_direction=1),
    FeatureSpec("months_since_last_delinquency", 0.0, 240.0, improve_direction=1),
    FeatureSpec("kyc_docs_verified", 0.0, 1.0, integer=True, improve_direction=1),
    FeatureSpec("device_trust_score", 0.0, 1.0, improve_direction=1),
    FeatureSpec("ip_country_risk", 0.0, 1.0, improve_direction=-1),
    FeatureSpec("velocity_apps_90d", 0.0, 20.0, integer=True, improve_direction=-1),
    FeatureSpec("avg_balance_6m", 0.0, 2_000_000.0, improve_direction=1),
    FeatureSpec("balance_volatility", 0.0, 1.0, improve_direction=-1),
    FeatureSpec("pct_on_time_payments", 0.0, 1.0, improve_direction=1),
    FeatureSpec("credit_utilization", 0.0, 1.0, improve_direction=-1),
    FeatureSpec(
        "has_previous_default", 0.0, 1.0, integer=True, mutable=False, improve_direction=-1
    ),
)

FEATURE_COLUMNS: tuple[str, ...] = tuple(spec.name for spec in FEATURE_SPECS)
FEATURE_SPECS_BY_NAME: dict[str, FeatureSpec] = {spec.name: spec for spec in FEATURE_SPECS}
MUTABLE_FEATURES: tuple[str, ...] = tuple(spec.name for spec in FEATURE_SPECS if spec.mutable)


@dataclass(frozen=True)
class SchemaIssue:
    column: str
    issue: str
    offending: int  # number of offending rows


def validate_schema(df: pd.DataFrame) -> list[SchemaIssue]:
    """Vectorized schema validation of a KYC dataframe.

    Checks required columns, dtypes, numeric bounds and integer-typedness.
    Returns a list of issues; an empty list means the dataframe is valid.
    """
    issues: list[SchemaIssue] = []

    required = [ID_COLUMN, *PROTECTED_COLUMNS, *FEATURE_COLUMNS, LABEL_COLUMN]
    for col in required:
        if col not in df.columns:
            issues.append(SchemaIssue(col, "missing required column", len(df)))
    if issues:
        return issues

    for spec in FEATURE_SPECS:
        raw = df[spec.name]
        if not pd.api.types.is_numeric_dtype(raw):
            issues.append(SchemaIssue(spec.name, "non-numeric dtype", len(df)))
            continue
        numeric = pd.to_numeric(raw)
        out_of_bounds = int(((numeric < spec.lo) | (numeric > spec.hi)).sum())
        if out_of_bounds:
            issues.append(
                SchemaIssue(spec.name, f"value out of [{spec.lo}, {spec.hi}]", out_of_bounds)
            )

    bad_label = int((~df[LABEL_COLUMN].isin([0, 1])).sum())
    if bad_label:
        issues.append(SchemaIssue(LABEL_COLUMN, "label not in {0, 1}", bad_label))
    return issues
