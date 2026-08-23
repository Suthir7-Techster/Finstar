"""Automatic model card generation.

Every registered model version gets a generated model card (Mitchell et al.
2019 style) capturing quantitative metrics, fairness evidence, training data
lineage and known limitations — the documentation artifact the paper's
governance stage requires and the EU AI Act's Article 11 technical
documentation expectation.
"""

from __future__ import annotations

from typing import Any

from feg_mlops.fairness import FairnessReport
from feg_mlops.modeling.evaluate import PerformanceReport


def generate_model_card(
    model_name: str,
    version: int,
    manifest: dict[str, Any],
    performance: PerformanceReport,
    fairness: FairnessReport,
    explanation_summary: dict[str, float],
    mitigation_applied: str,
    intended_use: str = (
        "Automated KYC/credit pre-screening for consumer lending. Decisions in the "
        "refer band must be reviewed by a human underwriter before any adverse action."
    ),
    limitations: tuple[str, ...] = (
        "Trained on synthetic data with a documented injected historical bias; "
        "absolute metric values are illustrative, not benchmarks.",
        "Fairness gates cover age-band groups only; production systems must extend "
        "monitoring to all protected attributes in scope.",
        "Post-deployment, population drift invalidates calibration — re-run "
        "`feg monitor` continuously and retrain on policy-defined cadence.",
    ),
) -> str:
    f = fairness
    metrics = manifest.get("metrics", {})
    lines = [
        f"# Model card — {model_name} v{version}",
        "",
        f"**Status:** {manifest.get('status', 'candidate')} · "
        f"**Created:** {manifest.get('created_at', 'n/a')} · "
        f"**Package:** feg-mlops {manifest.get('package_version', '')} · "
        f"**Commit:** `{str(manifest.get('git_sha', 'n/a'))[:12]}`",
        "",
        "## Intended use",
        "",
        intended_use,
        "",
        "## Training data",
        "",
        f"- Synthetic KYC dataset, data hash `{manifest.get('data_hash', 'n/a')[:16]}…`",
        f"- Mitigation applied: {mitigation_applied}",
        "",
        "## Quantitative analysis",
        "",
        f"- AUC (ground-truth outcomes): **{metrics.get('auc', float('nan')):.3f}**",
        f"- Balanced accuracy at operating threshold: "
        f"{metrics.get('balanced_accuracy', float('nan')):.3f}",
        f"- Brier score: {metrics.get('brier', float('nan')):.3f}",
        f"- Approval rate: {metrics.get('approval_rate', float('nan')):.3f}",
        "",
        "## Fairness",
        "",
        f"- Protected attribute: `{f.protected_attribute}` "
        f"(privileged `{f.privileged_group}` vs unprivileged `{f.unprivileged_group}`)",
        f"- Disparate impact ratio: **{f.disparate_impact_ratio:.3f}**",
        f"- Statistical parity difference: {f.statistical_parity_difference:+.3f}",
        f"- Equal opportunity difference: {f.equal_opportunity_difference:+.3f}",
        "",
        "| Group | n | Selection rate | TPR | FPR | Accuracy |",
        "|---|---|---|---|---|---|",
    ]
    for sg in f.subgroups:
        lines.append(
            f"| {sg.group} | {sg.n} | {sg.selection_rate:.3f} | {sg.tpr:.3f} "
            f"| {sg.fpr:.3f} | {sg.accuracy:.3f} |"
        )
    lines += [
        "",
        "## Top risk drivers (mean |attribution|)",
        "",
        "| Feature | Importance |",
        "|---|---|",
    ]
    for feat, imp in list(explanation_summary.items())[:8]:
        lines.append(f"| `{feat}` | {imp:.4f} |")
    lines += ["", "## Limitations and ethical considerations", ""]
    lines += [f"- {lim}" for lim in limitations]
    lines += [
        "",
        "## Approvals",
        "",
    ]
    approvals = manifest.get("approvals", [])
    if isinstance(approvals, list) and approvals:
        for a in approvals:
            if isinstance(a, dict):
                lines.append(
                    f"- **{a.get('actor', 'unknown')}**: {a.get('reason', '')} ({a.get('at', '')})"
                )
    else:
        lines.append("- _None yet — this version is awaiting sign-off._")
    return "\n".join(lines) + "\n"
