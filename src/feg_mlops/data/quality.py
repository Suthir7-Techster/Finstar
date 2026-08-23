"""Data-quality gates executed before any model sees the data.

Implements the paper's "Data Collection & Preprocessing" stage controls:
bias detection inputs, data quality checks and representativeness constraints
(Fig. 1, stage 1; Section 4.1).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from feg_mlops.config import DataSettings
from feg_mlops.data.schema import (
    FEATURE_COLUMNS,
    ID_COLUMN,
    LABEL_COLUMN,
    validate_schema,
)
from feg_mlops.data.synthetic import SyntheticKycGenerator


@dataclass(frozen=True)
class QualityIssue:
    check: str
    detail: str
    severity: str = "error"  # error blocks the pipeline; warning only reports


@dataclass(frozen=True)
class QualityReport:
    dataset: str
    rows: int
    passed: bool
    issues: list[QualityIssue] = field(default_factory=list)

    @property
    def errors(self) -> list[QualityIssue]:
        return [i for i in self.issues if i.severity == "error"]


class QualityChecker:
    """Runs all data-quality gates for one dataset."""

    def __init__(self, settings: DataSettings) -> None:
        self._cfg = settings

    def check(self, df: pd.DataFrame, dataset: str) -> QualityReport:
        issues: list[QualityIssue] = []
        q = self._cfg.quality
        d = self._cfg.data

        # 1. Schema conformance (columns, dtypes, bounds).
        for schema_issue in validate_schema(df):
            issues.append(QualityIssue("schema", f"{schema_issue.column}: {schema_issue.issue}"))

        # 2. Row count floor.
        if len(df) < q.min_rows:
            issues.append(QualityIssue("min_rows", f"{len(df)} < {q.min_rows}"))

        # 3. Null fractions.
        for col in (ID_COLUMN, *FEATURE_COLUMNS, LABEL_COLUMN):
            if col not in df.columns:
                continue
            frac = float(df[col].isna().mean())
            if frac > q.max_null_fraction:
                issues.append(QualityIssue("nulls", f"{col}: {frac:.3f} > {q.max_null_fraction}"))

        # 4. Duplicate ids.
        dup_frac = float(df[ID_COLUMN].duplicated().mean()) if ID_COLUMN in df.columns else 0.0
        if dup_frac > q.max_duplicate_fraction:
            issues.append(
                QualityIssue("duplicates", f"{dup_frac:.3f} > {q.max_duplicate_fraction}")
            )

        # 5. Class balance.
        if LABEL_COLUMN in df.columns:
            class_share = float(df[LABEL_COLUMN].mean())
            minority_share = min(class_share, 1.0 - class_share)
            if minority_share < q.min_class_share:
                issues.append(
                    QualityIssue(
                        "class_balance",
                        f"minority share {minority_share:.3f} < {q.min_class_share}",
                    )
                )

        # 6. Protected-group representativeness (bias-detection precondition).
        attr = d.protected_attribute
        if attr in df.columns:
            shares = df[attr].value_counts(normalize=True)
            for group, share in shares.items():
                if float(share) < q.min_group_share:
                    issues.append(
                        QualityIssue(
                            "group_representation",
                            f"{group}: {float(share):.3f} < {q.min_group_share}",
                        )
                    )
            missing_groups = {d.privileged_group, d.unprivileged_group} - set(shares.index)
            if missing_groups:
                issues.append(
                    QualityIssue(
                        "group_representation", f"missing groups: {sorted(missing_groups)}"
                    )
                )

        passed = not any(i.severity == "error" for i in issues)
        return QualityReport(dataset=dataset, rows=len(df), passed=passed, issues=issues)


def generate_datasets(settings: DataSettings) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate train + eval frames through the quality gates contract.

    Callers are expected to run :class:`QualityChecker` on the outputs; this
    helper only centralizes generation so pipeline and tests share one path.
    """
    generator = SyntheticKycGenerator(settings.data)
    return generator.train_frame(), generator.eval_frame()
