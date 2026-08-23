"""Decision engine: score, band, explain, audit, route to human review.

Bands come from the policy snapshot stored with the registered model:

- ``approve`` — risk below ``approve_below``
- ``refer``   — risk inside the refer band; routed to the human review queue
                (human-in-the-loop, paper Section 3.5 / recommendation 3)
- ``decline`` — risk above ``decline_above``; adverse-action reason codes
                plus an actionable counterfactual are always returned

Every decision is appended to the audit trail. Features themselves are
never logged — only a salt-free content hash (privacy by design).
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from feg_mlops.config import FegConfig
from feg_mlops.explain.counterfactual import find_counterfactual
from feg_mlops.explain.explainer import (
    MarginalAblationExplainer,
    ShapExplainer,
    build_explainer,
)
from feg_mlops.explain.reason_codes import reason_codes
from feg_mlops.governance.audit import AuditTrail
from feg_mlops.modeling.registry import ModelRegistry, RegisteredModel
from feg_mlops.serving.schemas import (
    CounterfactualOut,
    DecisionResponse,
    ReasonCodeOut,
    ReviewOut,
)


class ServingError(RuntimeError):
    """Raised when no model is registered or the payload is unusable."""


@dataclass(frozen=True)
class DecisionBands:
    approve_below: float
    refer_low: float
    refer_high: float
    decline_above: float

    def band(self, risk: float) -> str:
        if risk < self.approve_below:
            return "approve"
        if self.refer_low <= risk < self.refer_high:
            return "refer"
        return "decline"


class DecisionEngine:
    def __init__(self, config: FegConfig, artifacts_dir: str | Path) -> None:
        self._config = config
        self._artifacts = Path(artifacts_dir)
        self._registry = ModelRegistry(self._artifacts / "registry")
        self._audit = AuditTrail(self._artifacts / "audit" / "audit_trail.jsonl")
        self._reviews_path = self._artifacts / "serving" / "reviews.jsonl"
        self._reviews_path.parent.mkdir(parents=True, exist_ok=True)
        self.reload()

    # ------------------------------------------------------------------

    def reload(self) -> None:
        """Load the newest model version (approved preferred)."""
        try:
            self._model = self._registry.latest_approved("kyc_risk_model")
        except KeyError as exc:
            raise ServingError("no registered model — run `feg train` first") from exc
        payload = self._model.load(self._registry.root)
        self._estimator = payload["estimator"]
        self._feature_columns = list(payload["feature_columns"])
        decisions: dict[str, Any] = dict(payload.get("decisions", {}))
        refer = list(decisions["refer_between"])
        self._bands = DecisionBands(
            approve_below=float(decisions["approve_below"]),
            refer_low=float(refer[0]),
            refer_high=float(refer[1]),
            decline_above=float(decisions["decline_above"]),
        )
        self._max_reason_codes = int(decisions.get("max_reason_codes", 3))
        background = payload.get("background")
        if not isinstance(background, pd.DataFrame) or background.empty:
            raise ServingError("registered model lacks a background sample")
        self._background = background
        self._explainer: ShapExplainer | MarginalAblationExplainer = build_explainer(
            self.risk_scores, background
        )
        self._audit.append(
            "serving.model_loaded",
            {"version": self._model.version, "status": self._model.status},
        )

    @property
    def model(self) -> RegisteredModel:
        return self._model

    def risk_scores(self, frame: pd.DataFrame) -> np.ndarray:
        scores = self._estimator.predict_proba(frame[self._feature_columns])[:, 1]
        return np.asarray(scores, dtype=float)

    # ------------------------------------------------------------------

    def decide(
        self, features: dict[str, float], applicant_ref: str | None = None
    ) -> DecisionResponse:
        row = pd.DataFrame([features])[self._feature_columns].iloc[0]
        frame = pd.DataFrame([features])
        risk = float(self.risk_scores(frame)[0])
        band = self._bands.band(risk)
        decision_id = f"DEC-{uuid.uuid4().hex[:16]}"

        attributions = self._explainer.explain_row(row)
        codes = reason_codes(row, attributions, max_codes=self._max_reason_codes)
        reason_out = [
            ReasonCodeOut(code=c.code, text=c.text, contribution=c.contribution) for c in codes
        ]

        counterfactual: CounterfactualOut | None = None
        if band == "decline":
            cf = find_counterfactual(
                row=row,
                score_fn=self.risk_scores,
                target_score=self._bands.decline_above,  # exit the decline band
                background=self._background,
                attributions=attributions,
            )
            counterfactual = CounterfactualOut(found=cf.found, narrative=cf.render())

        self._audit.append(
            "decision.made",
            {
                "decision_id": decision_id,
                "model_version": self._model.version,
                "decision": band,
                "risk_score": round(risk, 4),
                "applicant_ref": applicant_ref,
                # Privacy: hash of the feature vector, never raw values.
                "features_sha256": hashlib.sha256(
                    json.dumps(features, sort_keys=True).encode()
                ).hexdigest(),
                "reason_codes": [c.code for c in codes],
            },
            actor="decision-engine",
        )

        review_required = band == "refer"
        if review_required:
            self._enqueue_review(decision_id, applicant_ref, risk, reason_out)
        return DecisionResponse(
            decision_id=decision_id,
            decision=band,
            risk_score=round(risk, 4),
            model_version=self._model.version,
            model_name=self._model.name,
            reason_codes=reason_out,
            counterfactual=counterfactual,
            review_required=review_required,
        )

    # ------------------------------------------------------------------
    # Human review queue

    def _enqueue_review(
        self,
        decision_id: str,
        applicant_ref: str | None,
        risk: float,
        codes: list[ReasonCodeOut],
    ) -> None:
        record = {
            "review_id": f"REV-{uuid.uuid4().hex[:12]}",
            "decision_id": decision_id,
            "applicant_ref": applicant_ref,
            "risk_score": round(risk, 4),
            "status": "open",
            "reason_codes": [c.model_dump() for c in codes],
            "created_at": datetime.now(UTC).isoformat(),
        }
        with self._reviews_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")

    def _reviews(self) -> list[dict[str, Any]]:
        if not self._reviews_path.exists():
            return []
        out = []
        for line in self._reviews_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                out.append(json.loads(line))
        return out

    def list_reviews(self, status: str | None = None) -> list[ReviewOut]:
        reviews = self._reviews()
        if status:
            reviews = [r for r in reviews if r["status"] == status]
        return [self._to_review_out(r) for r in reviews]

    def resolve_review(self, review_id: str, outcome: str, reviewer: str, note: str) -> ReviewOut:
        reviews = self._reviews()
        target = next((r for r in reviews if r["review_id"] == review_id), None)
        if target is None:
            raise KeyError(review_id)
        if target["status"] != "open":
            raise ValueError(f"review {review_id} already {target['status']}")
        target["status"] = outcome
        target["resolved_by"] = reviewer
        target["resolution_note"] = note
        target["resolved_at"] = datetime.now(UTC).isoformat()
        with self._reviews_path.open("w", encoding="utf-8") as fh:
            for r in reviews:
                fh.write(json.dumps(r) + "\n")
        self._audit.append(
            "review.resolved",
            {
                "review_id": review_id,
                "decision_id": target["decision_id"],
                "outcome": outcome,
                "reviewer": reviewer,
                "note": note,
            },
            actor=reviewer,
        )
        return next(self._to_review_out(r) for r in self._reviews() if r["review_id"] == review_id)

    @staticmethod
    def _to_review_out(r: dict[str, Any]) -> ReviewOut:
        return ReviewOut(
            review_id=r["review_id"],
            decision_id=r["decision_id"],
            applicant_ref=r.get("applicant_ref"),
            risk_score=r["risk_score"],
            status=r["status"],
            reason_codes=[ReasonCodeOut(**rc) for rc in r["reason_codes"]],
            created_at=r["created_at"],
        )

    # ------------------------------------------------------------------
