"""Fairness gates: hard policy checks that block model promotion.

A gate failure must stop the pipeline (non-zero exit) — this is the paper's
"validation gates" governance control (Section 4.3) made executable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from feg_mlops.config import FairnessSettings
from feg_mlops.fairness import FairnessReport


@dataclass(frozen=True)
class GateResult:
    name: str
    passed: bool
    observed: float
    threshold: str  # human-readable threshold description


@dataclass(frozen=True)
class GateEvaluation:
    report: FairnessReport
    results: list[GateResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(r.passed for r in self.results)

    @property
    def failures(self) -> list[GateResult]:
        return [r for r in self.results if not r.passed]

    def render_markdown(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        lines = [
            f"### Fairness gates — {status}",
            "",
            "| Gate | Observed | Policy | Result |",
            "|---|---|---|---|",
        ]
        for r in self.results:
            lines.append(
                f"| {r.name} | {r.observed:.3f} | {r.threshold} | "
                f"{'✅ pass' if r.passed else '❌ fail'} |"
            )
        return "\n".join(lines) + "\n"


def evaluate_fairness_gates(report: FairnessReport, settings: FairnessSettings) -> GateEvaluation:
    gates = settings.gates
    results = [
        GateResult(
            name="disparate_impact_min",
            passed=report.disparate_impact_ratio >= gates.disparate_impact_min,
            observed=report.disparate_impact_ratio,
            threshold=f">= {gates.disparate_impact_min} (four-fifths rule)",
        ),
        GateResult(
            name="statistical_parity_abs_max",
            passed=abs(report.statistical_parity_difference) <= gates.statistical_parity_abs_max,
            observed=abs(report.statistical_parity_difference),
            threshold=f"abs <= {gates.statistical_parity_abs_max}",
        ),
        GateResult(
            name="equal_opportunity_abs_max",
            passed=abs(report.equal_opportunity_difference) <= gates.equal_opportunity_abs_max,
            observed=abs(report.equal_opportunity_difference),
            threshold=f"abs <= {gates.equal_opportunity_abs_max}",
        ),
    ]
    # Subgroup floors: every protected group must clear recall/count minimums.
    for sg in report.subgroups:
        results.append(
            GateResult(
                name=f"subgroup_recall[{sg.group}]",
                passed=sg.tpr >= gates.min_subgroup_recall,
                observed=sg.tpr,
                threshold=f">= {gates.min_subgroup_recall}",
            )
        )
        results.append(
            GateResult(
                name=f"subgroup_count[{sg.group}]",
                passed=sg.n >= gates.min_subgroup_count,
                observed=float(sg.n),
                threshold=f">= {gates.min_subgroup_count}",
            )
        )
    return GateEvaluation(report=report, results=results)
