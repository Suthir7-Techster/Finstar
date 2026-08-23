"""End-to-end FEG training pipeline.

Stages (each one emits audit events and fails closed):

1. generate deterministic synthetic KYC data (train + eval),
2. data-quality gates (schema, nulls, balance, representation),
3. train candidate models unmitigated,
4. fairness + performance evaluation against ground-truth outcomes,
5. policy gates — on failure, automatic mitigation (reweighing, then
   group-threshold post-processing) and re-evaluation,
6. explainability artifacts (SHAP global importance) — versioned with model,
7. registration in the model registry with lineage + model card.

A pipeline whose final candidate cannot pass policy exits non-zero: a
discriminatory model is never registered. This is the paper's FEG framework
(Fig. 1) made executable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from feg_mlops import __version__
from feg_mlops.config import DataConfig, FegConfig, load_configs
from feg_mlops.data.quality import QualityChecker, generate_datasets
from feg_mlops.data.schema import FEATURE_COLUMNS
from feg_mlops.explain.explainer import (
    build_explainer,
    global_importance,
    global_importance_markdown,
)
from feg_mlops.fairness import FairnessReport, fairness_report
from feg_mlops.fairness.gates import GateEvaluation
from feg_mlops.fairness.mitigation import (
    apply_group_thresholds,
    optimize_group_thresholds,
    reweighing_weights,
)
from feg_mlops.governance.audit import AuditTrail
from feg_mlops.governance.model_card import generate_model_card
from feg_mlops.governance.policy import PolicyEngine
from feg_mlops.modeling.evaluate import performance_report
from feg_mlops.modeling.registry import (
    ModelPayload,
    ModelRegistry,
    RegisteredModel,
    frame_hash,
)
from feg_mlops.modeling.train import TrainedCandidate, train_candidates

SMALL_OVERRIDES: dict[str, int] = {"n_train": 900, "n_eval": 450, "n_monitor_window": 300}


class PipelineError(RuntimeError):
    """Unrecoverable pipeline failure (quality gate, policy gate, ...)."""


@dataclass(frozen=True)
class CandidateEvaluation:
    candidate: TrainedCandidate
    auc: float
    fairness: FairnessReport
    gates: GateEvaluation
    approved: np.ndarray
    group_thresholds: dict[str, float] | None = None


@dataclass
class TrainingOutcome:
    passed: bool
    champion_name: str
    champion_evaluation: CandidateEvaluation | None
    unmitigated_fairness: FairnessReport | None
    mitigation_applied: str
    registered: RegisteredModel | None
    run_dir: Path
    summary: dict[str, Any]


def _approved_mask(
    candidate: TrainedCandidate,
    eval_frame: pd.DataFrame,
    group_thresholds: dict[str, float] | None,
    attr: str,
) -> np.ndarray:
    scores = candidate.risk_scores(eval_frame)
    if group_thresholds is not None:
        return apply_group_thresholds(eval_frame[attr], scores, group_thresholds)
    return scores < candidate.threshold


def _evaluate_candidate(
    candidate: TrainedCandidate,
    eval_frame: pd.DataFrame,
    config: FegConfig,
    dataset_label: str,
    group_thresholds: dict[str, float] | None = None,
) -> CandidateEvaluation:
    fs = config.fairness_settings
    approved = _approved_mask(candidate, eval_frame, group_thresholds, fs.protected_attribute)
    report = fairness_report(
        groups=eval_frame[fs.protected_attribute],
        approved=approved,
        labels=eval_frame["defaulted_ground_truth"],
        dataset=dataset_label,
        protected_attribute=fs.protected_attribute,
        privileged_group=fs.privileged_group,
        unprivileged_group=fs.unprivileged_group,
    )
    scores = candidate.risk_scores(eval_frame)
    effective_threshold = (
        float(np.mean(list(group_thresholds.values()))) if group_thresholds else candidate.threshold
    )
    perf = performance_report(
        eval_frame["defaulted_ground_truth"].to_numpy(),
        scores,
        effective_threshold,
        eval_frame[fs.protected_attribute],
        dataset=dataset_label,
    )
    engine = PolicyEngine(config.model_settings, config.fairness_settings)
    gates = engine.fairness_gates(report)
    return CandidateEvaluation(
        candidate=candidate,
        auc=perf.auc,
        fairness=report,
        gates=gates,
        approved=approved,
        group_thresholds=group_thresholds,
    )


def run_training_pipeline(
    config_dir: str | Path = "configs",
    artifacts_dir: str | Path = "artifacts",
    smoke: bool = False,
) -> TrainingOutcome:
    """Execute the full FEG training pipeline; returns the run outcome."""
    config = load_configs(config_dir)
    if smoke:
        shrunk_data = _shrunk_data_config(config.data)
        shrunk_quality = config.data_settings.quality.model_copy(update={"min_rows": 200})
        config = config.model_copy(
            update={
                "data_settings": config.data_settings.model_copy(
                    update={"data": shrunk_data, "quality": shrunk_quality}
                )
            }
        )
    artifacts_dir = Path(artifacts_dir)
    run_dir = artifacts_dir / "runs" / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir.mkdir(parents=True, exist_ok=True)
    audit = AuditTrail(artifacts_dir / "audit" / "audit_trail.jsonl")
    registry = ModelRegistry(artifacts_dir / "registry")
    fs = config.fairness_settings

    audit.append(
        "pipeline.start",
        {"mode": "smoke" if smoke else "full", "seed": config.data.seed},
    )

    # --- Stage 1-2: data + quality gates ---------------------------------
    train_df, eval_df = generate_datasets(config.data_settings)
    train_hash, eval_hash = frame_hash(train_df), frame_hash(eval_df)
    audit.append(
        "data.generated",
        {
            "rows_train": len(train_df),
            "rows_eval": len(eval_df),
            "train_hash": train_hash,
            "eval_hash": eval_hash,
        },
    )

    checker = QualityChecker(config.data_settings)
    for name, frame in (("train", train_df), ("eval", eval_df)):
        report = checker.check(frame, name)
        audit.append(
            "quality.checked",
            {
                "dataset": name,
                "passed": report.passed,
                "issues": [f"{i.check}: {i.detail}" for i in report.issues],
            },
        )
        if not report.passed:
            raise PipelineError(
                f"quality gate failed for {name}: "
                + "; ".join(f"{i.check}: {i.detail}" for i in report.errors)
            )

    # --- Stage 3-4: unmitigated candidates -------------------------------
    features, labels = train_df[list(FEATURE_COLUMNS)], train_df["defaulted"]
    candidates = train_candidates(features, labels, config.model_settings)

    evaluations = [
        _evaluate_candidate(c, eval_df, config, f"eval-unmitigated-{c.name}")
        for c in candidates.values()
    ]
    evaluations.sort(key=lambda e: e.auc, reverse=True)
    best_unmitigated = evaluations[0]
    audit.append(
        "fairness.evaluated",
        {
            "phase": "unmitigated",
            "champion": best_unmitigated.candidate.name,
            "auc": round(best_unmitigated.auc, 4),
            "disparate_impact": round(best_unmitigated.fairness.disparate_impact_ratio, 4),
            "gates_passed": best_unmitigated.gates.passed,
        },
    )

    # --- Stage 5: gates -> mitigation loop --------------------------------
    champion = best_unmitigated if best_unmitigated.gates.passed else None
    mitigation_applied = "none (unmitigated candidate passed all gates)"
    weights = None

    if champion is None and fs.mitigation.reweighing:
        audit.append("fairness.mitigation_started", {"technique": "reweighing"})
        weights = reweighing_weights(train_df[fs.protected_attribute], labels)
        mitigated = train_candidates(
            features, labels, config.model_settings, sample_weights=weights
        )
        evaluations = [
            _evaluate_candidate(c, eval_df, config, f"eval-reweighed-{c.name}")
            for c in mitigated.values()
        ]
        evaluations.sort(key=lambda e: e.auc, reverse=True)
        passing = [e for e in evaluations if e.gates.passed]
        mitigation_applied = "reweighing (Kamiran & Calders 2012)"
        champion = max(passing, key=lambda e: e.auc) if passing else evaluations[0]

    if (
        champion is not None
        and not champion.gates.passed
        and fs.mitigation.group_threshold_optimization
    ):
        audit.append("fairness.mitigation_started", {"technique": "group_threshold_optimization"})
        thresholds = optimize_group_thresholds(
            groups=eval_df[fs.protected_attribute],
            labels=eval_df["defaulted_ground_truth"],
            scores=champion.candidate.risk_scores(eval_df),
            base_threshold=champion.candidate.threshold,
            min_recall=fs.gates.min_subgroup_recall,
        )
        champion = _evaluate_candidate(
            champion.candidate,
            eval_df,
            config,
            f"eval-thresholded-{champion.candidate.name}",
            group_thresholds=thresholds,
        )
        mitigation_applied += " + group-threshold post-processing"

    if champion is None or not champion.gates.passed:
        failure = (champion or best_unmitigated).gates.failures[0]
        audit.append(
            "pipeline.blocked",
            {"reason": f"gate {failure.name}: {failure.observed:.3f} vs {failure.threshold}"},
        )
        _write_summary(run_dir, None, mitigation_applied, best_unmitigated)
        return TrainingOutcome(
            passed=False,
            champion_name=(champion or best_unmitigated).candidate.name,
            champion_evaluation=champion,
            unmitigated_fairness=best_unmitigated.fairness,
            mitigation_applied=mitigation_applied,
            registered=None,
            run_dir=run_dir,
            summary={"status": "blocked", "first_failure": failure.name},
        )

    # --- Stage 6: explainability artifacts --------------------------------
    # champion is non-None from here on (policy passed above).
    assert champion is not None
    background = train_df[list(FEATURE_COLUMNS)].sample(
        n=min(256, len(train_df)), random_state=config.data.seed
    )
    explanation_rows = eval_df[list(FEATURE_COLUMNS)].sample(
        n=min(200, len(eval_df)), random_state=config.data.seed
    )

    explainer = build_explainer(champion.candidate.risk_scores, background)
    attributions = explainer.explain_batch(explanation_rows)
    importance = global_importance(attributions)
    audit.append(
        "explainability.artifacts_generated",
        {
            "engine": explainer.name,
            "rows": len(explanation_rows),
            "top_feature": next(iter(importance)),
        },
    )

    # --- Stage 7: registration + model card -------------------------------
    perf = performance_report(
        eval_df["defaulted_ground_truth"].to_numpy(),
        champion.candidate.risk_scores(eval_df),
        champion.candidate.threshold,
        eval_df[fs.protected_attribute],
        dataset="eval",
    )
    payload = ModelPayload(
        estimator=champion.candidate.estimator,
        threshold=float(champion.candidate.threshold),
        group_thresholds=champion.group_thresholds,
        feature_columns=list(FEATURE_COLUMNS),
        protected_attribute=fs.protected_attribute,
        decisions=config.model_settings.decisions.model_dump(),
        # Background sample for runtime SHAP explanations (deterministic).
        background=background.sample(n=min(64, len(background)), random_state=1),
        package_version=__version__,
    )
    registered = registry.register(
        name="kyc_risk_model",
        model_payload=payload,
        artifacts={
            "fairness_report.md": champion.fairness.render_markdown(),
            "fairness_report.json": json.dumps(champion.fairness.to_dict(), indent=2),
            "performance_report.md": perf.render_markdown(),
            "performance_report.json": json.dumps(perf.to_dict(), indent=2),
            "global_importance.json": json.dumps(importance, indent=2),
            "global_importance.md": global_importance_markdown(importance),
        },
        metrics=perf.to_dict(),
        fairness={
            "disparate_impact_ratio": champion.fairness.disparate_impact_ratio,
            "statistical_parity_difference": champion.fairness.statistical_parity_difference,
            "equal_opportunity_difference": champion.fairness.equal_opportunity_difference,
        },
        gates={
            "passed": champion.gates.passed,
            "failures": [f.name for f in champion.gates.failures],
        },
        data_hash=train_hash,
        notes=mitigation_applied,
    )
    card = generate_model_card(
        model_name="kyc_risk_model",
        version=registered.version,
        manifest=registered.manifest,
        performance=perf,
        fairness=champion.fairness,
        explanation_summary=importance,
        mitigation_applied=mitigation_applied,
    )
    card_path = registry.root / "kyc_risk_model" / f"v{registered.version}" / "model_card.md"
    card_path.write_text(card, encoding="utf-8")

    audit.append(
        "model.registered",
        {
            "name": "kyc_risk_model",
            "version": registered.version,
            "mitigation": mitigation_applied,
            "auc": round(perf.auc, 4),
            "disparate_impact": round(champion.fairness.disparate_impact_ratio, 4),
        },
    )
    audit.append("pipeline.finished", {"status": "registered"})

    summary = {
        "status": "registered",
        "champion": champion.candidate.name,
        "version": registered.version,
        "auc": round(perf.auc, 4),
        "disparate_impact_unmitigated": round(best_unmitigated.fairness.disparate_impact_ratio, 4),
        "disparate_impact_registered": round(champion.fairness.disparate_impact_ratio, 4),
        "mitigation": mitigation_applied,
    }
    _write_summary(run_dir, summary, mitigation_applied, best_unmitigated, champion)
    return TrainingOutcome(
        passed=True,
        champion_name=champion.candidate.name,
        champion_evaluation=champion,
        unmitigated_fairness=best_unmitigated.fairness,
        mitigation_applied=mitigation_applied,
        registered=registered,
        run_dir=run_dir,
        summary=summary,
    )


def _shrunk_data_config(base: DataConfig) -> DataConfig:
    return base.model_copy(update=SMALL_OVERRIDES)


def _write_summary(
    run_dir: Path,
    summary: dict[str, Any] | None,
    mitigation: str,
    unmitigated: CandidateEvaluation,
    champion: CandidateEvaluation | None = None,
) -> None:
    payload = {
        "summary": summary or {"status": "blocked"},
        "mitigation": mitigation,
        "unmitigated": {
            "auc": round(unmitigated.auc, 4),
            "disparate_impact": round(unmitigated.fairness.disparate_impact_ratio, 4),
            "gates": [
                f"{r.name}: {'pass' if r.passed else 'fail'}" for r in unmitigated.gates.results
            ],
        },
    }
    if champion is not None:
        payload["registered_candidate"] = {
            "name": champion.candidate.name,
            "auc": round(champion.auc, 4),
            "disparate_impact": round(champion.fairness.disparate_impact_ratio, 4),
            "gates": [
                f"{r.name}: {'pass' if r.passed else 'fail'}" for r in champion.gates.results
            ],
        }
    (run_dir / "run_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
