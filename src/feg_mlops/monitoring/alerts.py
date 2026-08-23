"""Alert evaluation and rendering for monitoring runs."""

from __future__ import annotations

from dataclasses import dataclass

from feg_mlops.config import MonitoringSettings
from feg_mlops.monitoring.drift import FeatureDrift
from feg_mlops.monitoring.fairness_drift import FairnessDriftResult


@dataclass(frozen=True)
class Alert:
    severity: str  # "warning" | "critical"
    component: str  # "data-drift" | "fairness-drift"
    message: str

    def to_dict(self) -> dict[str, object]:
        return {"severity": self.severity, "component": self.component, "message": self.message}


def evaluate_alerts(
    drifts: list[FeatureDrift],
    fairness: FairnessDriftResult,
    settings: MonitoringSettings,
) -> list[Alert]:
    alerts: list[Alert] = []

    critical = [d for d in drifts if d.level == "critical"]
    warning = [d for d in drifts if d.level == "warning"]
    if critical:
        alerts.append(
            Alert(
                "critical",
                "data-drift",
                f"{len(critical)} features critically drifted: "
                + ", ".join(d.feature for d in critical),
            )
        )
    elif len(warning) >= settings.alerts.min_features_for_dataset_alert:
        alerts.append(
            Alert(
                "warning",
                "data-drift",
                f"{len(warning)} features drifted past warning thresholds: "
                + ", ".join(d.feature for d in warning),
            )
        )

    if fairness.level == "critical":
        alerts.append(
            Alert(
                "critical",
                "fairness-drift",
                f"disparate impact dropped {fairness.di_drop:+.3f} vs reference "
                f"(now {fairness.di_current:.3f}); retrain or rollback required",
            )
        )
    elif fairness.level == "warning":
        alerts.append(
            Alert(
                "warning",
                "fairness-drift",
                f"disparate impact moved {fairness.di_drop:+.3f} vs reference "
                f"(now {fairness.di_current:.3f}); investigate subgroup mix and inputs",
            )
        )
    return alerts


def render_alerts_markdown(alerts: list[Alert]) -> str:
    if not alerts:
        return "No alerts — all monitored dimensions within policy.\n"
    lines = [f"- **{a.severity.upper()}** [{a.component}] {a.message}" for a in alerts]
    return "\n".join(lines) + "\n"
