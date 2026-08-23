"""Policy engine: the single authority for go/no-go decisions.

Wraps fairness gates and performance floors into structured
:class:`PolicyDecision` objects. The pipeline consults it before promotion;
CI runs the same code path — "governance as code" (paper Section 4.3).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from feg_mlops.config import FairnessSettings, ModelSettings
from feg_mlops.fairness import FairnessReport
from feg_mlops.fairness.gates import GateEvaluation, evaluate_fairness_gates


@dataclass(frozen=True)
class PolicyDecision:
    action: str  # "promote" | "block"
    reasons: list[str] = field(default_factory=list)
    gate_evaluation: GateEvaluation | None = None

    @property
    def allowed(self) -> bool:
        return self.action == "promote"


class PolicyEngine:
    def __init__(self, model_settings: ModelSettings, fairness_settings: FairnessSettings) -> None:
        self._model = model_settings
        self._fairness = fairness_settings

    def fairness_gates(self, report: FairnessReport) -> GateEvaluation:
        return evaluate_fairness_gates(report, self._fairness)

    def promotion_decision(
        self,
        report: FairnessReport,
        auc: float,
    ) -> PolicyDecision:
        """Combined fairness + performance verdict for model promotion."""
        reasons: list[str] = []
        gates = self.fairness_gates(report)
        for failure in gates.failures:
            reasons.append(
                f"fairness gate {failure.name} failed: observed {failure.observed:.3f} "
                f"against policy {failure.threshold}"
            )
        floor = self._model.selection.min_auc
        if auc < floor:
            reasons.append(f"performance floor failed: AUC {auc:.3f} < {floor}")
        action = "promote" if not reasons else "block"
        return PolicyDecision(action=action, reasons=reasons, gate_evaluation=gates)
