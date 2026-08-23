"""Post-deployment monitoring run.

Simulates production windows under the configured drift scenarios, scores
them with the registered model, and evaluates:

- per-feature data drift (PSI / Jensen-Shannon) vs the reference window,
- fairness drift (disparate impact, statistical parity) vs the fairness
  posture recorded in the model's registry manifest,
- alerts at warning/critical levels, appended to the audit trail.

Because the data generator is seeded, the reference window is exactly the
evaluation set from training — demonstrating lineage-based monitoring.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from feg_mlops.config import load_configs
from feg_mlops.data.schema import FEATURE_COLUMNS
from feg_mlops.data.synthetic import SyntheticKycGenerator
from feg_mlops.fairness import fairness_report
from feg_mlops.governance.audit import AuditTrail
from feg_mlops.modeling.registry import ModelRegistry
from feg_mlops.monitoring.alerts import evaluate_alerts, render_alerts_markdown
from feg_mlops.monitoring.drift import compute_feature_drift, drift_report_markdown
from feg_mlops.monitoring.fairness_drift import fairness_drift


@dataclass(frozen=True)
class WindowResult:
    scenario: str
    n: int
    alert_dicts: list[dict[str, Any]]
    fairness_drift_dict: dict[str, Any]
    report_path: Path


@dataclass
class MonitoringOutcome:
    windows: list[WindowResult]
    total_alerts: int
    report_dir: Path


def run_monitoring_pipeline(
    config_dir: str | Path = "configs",
    artifacts_dir: str | Path = "artifacts",
    model_name: str = "kyc_risk_model",
) -> MonitoringOutcome:
    config = load_configs(config_dir)
    artifacts_dir = Path(artifacts_dir)
    audit = AuditTrail(artifacts_dir / "audit" / "audit_trail.jsonl")
    registry = ModelRegistry(artifacts_dir / "registry")

    registered = registry.latest(model_name)
    payload = registered.load(registry.root)
    estimator = payload["estimator"]
    threshold = float(payload["threshold"])
    group_thresholds = payload.get("group_thresholds")

    fs = config.fairness_settings
    generator = SyntheticKycGenerator(config.data)
    reference = generator.eval_frame()  # deterministic: identical to training eval

    def score(frame: pd.DataFrame) -> np.ndarray:
        return np.asarray(estimator.predict_proba(frame[list(FEATURE_COLUMNS)])[:, 1])

    def approvals(frame: pd.DataFrame) -> np.ndarray:
        s = score(frame)
        if group_thresholds:
            from feg_mlops.fairness.mitigation import apply_group_thresholds

            return apply_group_thresholds(frame[fs.protected_attribute], s, group_thresholds)
        return s < threshold

    reference_fairness = fairness_report(
        groups=reference[fs.protected_attribute],
        approved=approvals(reference),
        labels=reference["defaulted_ground_truth"],
        dataset="reference",
        protected_attribute=fs.protected_attribute,
        privileged_group=fs.privileged_group,
        unprivileged_group=fs.unprivileged_group,
    )

    report_dir = artifacts_dir / "monitoring"
    report_dir.mkdir(parents=True, exist_ok=True)
    audit.append(
        "monitoring.started",
        {
            "model": model_name,
            "version": registered.version,
            "reference_di": round(reference_fairness.disparate_impact_ratio, 4),
        },
    )

    windows: list[WindowResult] = []
    total_alerts = 0
    for index, (scenario_name, scenario) in enumerate(config.data_settings.drift_scenarios.items()):
        window = generator.monitoring_window(scenario, index)
        current_fairness = fairness_report(
            groups=window[fs.protected_attribute],
            approved=approvals(window),
            labels=window["defaulted_ground_truth"],
            dataset=f"window:{scenario_name}",
            protected_attribute=fs.protected_attribute,
            privileged_group=fs.privileged_group,
            unprivileged_group=fs.unprivileged_group,
        )
        drifts = compute_feature_drift(
            reference[list(FEATURE_COLUMNS)],
            window[list(FEATURE_COLUMNS)],
            FEATURE_COLUMNS,
            config.monitoring_settings.drift,
        )
        fdrift = fairness_drift(
            reference_fairness, current_fairness, config.monitoring_settings.fairness_drift
        )
        alerts = evaluate_alerts(drifts, fdrift, config.monitoring_settings)

        drift_levels = {d.feature: d.level for d in drifts}
        audit.append(
            "monitoring.window_evaluated",
            {
                "scenario": scenario_name,
                "rows": len(window),
                "features_drifted_warning": sum(1 for v in drift_levels.values() if v == "warning"),
                "features_drifted_critical": sum(
                    1 for v in drift_levels.values() if v == "critical"
                ),
                "fairness_level": fdrift.level,
                "alerts": [a.to_dict() for a in alerts],
            },
        )

        report_path = report_dir / f"window_{scenario_name}.md"
        report_path.write_text(
            "# Monitoring window — "
            f"{scenario_name}\n\n## Fairness drift\n\n{fdrift.render_markdown()}\n"
            f"## Feature drift\n\n{drift_report_markdown(drifts)}\n"
            f"## Alerts\n\n{render_alerts_markdown(alerts)}",
            encoding="utf-8",
        )
        (report_dir / f"window_{scenario_name}.json").write_text(
            json.dumps(
                {
                    "scenario": scenario_name,
                    "rows": len(window),
                    "feature_drift": [d.to_dict() for d in drifts],
                    "fairness_drift": fdrift.to_dict(),
                    "alerts": [a.to_dict() for a in alerts],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        windows.append(
            WindowResult(
                scenario=scenario_name,
                n=len(window),
                alert_dicts=[a.to_dict() for a in alerts],
                fairness_drift_dict=fdrift.to_dict(),
                report_path=report_path,
            )
        )
        total_alerts += len(alerts)

    audit.append("monitoring.finished", {"windows": len(windows), "alerts": total_alerts})
    return MonitoringOutcome(windows=windows, total_alerts=total_alerts, report_dir=report_dir)
