"""Fairness drift: has the deployed model's fairness posture moved?

Recomputes the fairness metrics on a production window and compares them
against the reference window recorded at registration — the paper's
"continuous bias monitoring" recommendation (Section 3.3, rec. 1).
"""

from __future__ import annotations

from dataclasses import dataclass

from feg_mlops.config import FairnessDriftThresholds
from feg_mlops.fairness import FairnessReport


@dataclass(frozen=True)
class FairnessDriftResult:
    di_current: float
    di_reference: float
    di_drop: float  # positive when current is worse
    spd_delta: float
    eod_delta: float
    level: str  # "ok" | "warning" | "critical"

    def to_dict(self) -> dict[str, object]:
        return {
            "disparate_impact_current": round(self.di_current, 4),
            "disparate_impact_reference": round(self.di_reference, 4),
            "disparate_impact_drop": round(self.di_drop, 4),
            "statistical_parity_delta": round(self.spd_delta, 4),
            "equal_opportunity_delta": round(self.eod_delta, 4),
            "level": self.level,
        }

    def render_markdown(self) -> str:
        return (
            f"Disparate impact: reference **{self.di_reference:.3f}** → current "
            f"**{self.di_current:.3f}** (drop {self.di_drop:+.3f}, level `{self.level}`)\n\n"
            f"Statistical parity delta vs reference: {self.spd_delta:+.3f} · "
            f"equal-opportunity delta: {self.eod_delta:+.3f}\n"
        )


def fairness_drift(
    reference: FairnessReport, current: FairnessReport, thresholds: FairnessDriftThresholds
) -> FairnessDriftResult:
    di_drop = reference.disparate_impact_ratio - current.disparate_impact_ratio
    spd_delta = current.statistical_parity_difference - reference.statistical_parity_difference
    eod_delta = current.equal_opportunity_difference - reference.equal_opportunity_difference

    if di_drop >= thresholds.disparate_impact_abs_drop_critical or (
        abs(spd_delta) >= thresholds.statistical_parity_abs_delta_critical
    ):
        level = "critical"
    elif di_drop >= thresholds.disparate_impact_abs_drop_warning or (
        abs(spd_delta) >= thresholds.statistical_parity_abs_delta_warning
    ):
        level = "warning"
    else:
        level = "ok"
    return FairnessDriftResult(
        di_current=current.disparate_impact_ratio,
        di_reference=reference.disparate_impact_ratio,
        di_drop=round(di_drop, 4),
        spd_delta=round(spd_delta, 4),
        eod_delta=round(eod_delta, 4),
        level=level,
    )
