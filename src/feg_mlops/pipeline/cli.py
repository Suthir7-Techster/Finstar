"""Command-line interface: ``feg train | monitor | audit-verify | approve``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from feg_mlops.governance.audit import AuditTrail
from feg_mlops.modeling.registry import ModelRegistry
from feg_mlops.pipeline.run_monitoring import run_monitoring_pipeline
from feg_mlops.pipeline.run_training import PipelineError, run_training_pipeline

DEFAULT_CONFIG_DIR = "configs"
DEFAULT_ARTIFACTS_DIR = "artifacts"


def _cmd_train(args: argparse.Namespace) -> int:
    try:
        outcome = run_training_pipeline(
            config_dir=args.config_dir, artifacts_dir=args.artifacts_dir, smoke=args.smoke
        )
    except PipelineError as exc:
        print(f"PIPELINE BLOCKED: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(outcome.summary, indent=2))
    if not outcome.passed or outcome.registered is None:
        print("model NOT registered: fairness policy not satisfied", file=sys.stderr)
        return 1
    print(
        f"registered kyc_risk_model v{outcome.registered.version} in {args.artifacts_dir}/registry"
    )
    return 0


def _cmd_monitor(args: argparse.Namespace) -> int:
    outcome = run_monitoring_pipeline(config_dir=args.config_dir, artifacts_dir=args.artifacts_dir)
    for window in outcome.windows:
        alerts = (
            ", ".join(f"{a['severity']}:{a['component']}" for a in window.alert_dicts)
            or "no alerts"
        )
        print(f"window {window.scenario} (n={window.n}) -> {alerts}")
    print(f"reports: {outcome.report_dir}")
    return 0


def _cmd_audit_verify(args: argparse.Namespace) -> int:
    trail = AuditTrail(Path(args.artifacts_dir) / "audit" / "audit_trail.jsonl")
    verification = trail.verify()
    print(verification.render())
    return 0 if verification.valid else 2


def _cmd_approve(args: argparse.Namespace) -> int:
    registry = ModelRegistry(Path(args.artifacts_dir) / "registry")
    model = registry.approve(args.model, args.version, actor=args.actor, reason=args.reason)
    trail = AuditTrail(Path(args.artifacts_dir) / "audit" / "audit_trail.jsonl")
    trail.append(
        "model.approved",
        {"model": args.model, "version": args.version, "actor": args.actor, "reason": args.reason},
        actor=args.actor,
    )
    print(f"{model.name} v{model.version} approved by {args.actor}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="feg",
        description="Fairness-Explainability-Governance MLOps for KYC decisioning",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="run the FEG training pipeline")
    train.add_argument("--config-dir", default=DEFAULT_CONFIG_DIR)
    train.add_argument("--artifacts-dir", default=DEFAULT_ARTIFACTS_DIR)
    train.add_argument("--smoke", action="store_true", help="reduced dataset sizes")
    train.set_defaults(func=_cmd_train)

    monitor = sub.add_parser("monitor", help="evaluate drift scenarios on the deployed model")
    monitor.add_argument("--config-dir", default=DEFAULT_CONFIG_DIR)
    monitor.add_argument("--artifacts-dir", default=DEFAULT_ARTIFACTS_DIR)
    monitor.set_defaults(func=_cmd_monitor)

    verify = sub.add_parser("audit-verify", help="verify the audit trail hash chain")
    verify.add_argument("--artifacts-dir", default=DEFAULT_ARTIFACTS_DIR)
    verify.set_defaults(func=_cmd_audit_verify)

    approve = sub.add_parser("approve", help="record a human sign-off on a model version")
    approve.add_argument("--model", default="kyc_risk_model")
    approve.add_argument("--version", type=int, required=True)
    approve.add_argument("--actor", required=True, help="name of the approver")
    approve.add_argument("--reason", required=True, help="why this version is approved")
    approve.add_argument("--artifacts-dir", default=DEFAULT_ARTIFACTS_DIR)
    approve.set_defaults(func=_cmd_approve)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    handler = args.func
    assert handler is not None  # subparsers are required
    return int(handler(args))


if __name__ == "__main__":
    sys.exit(main())
