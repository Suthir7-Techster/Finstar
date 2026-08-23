"""File-based model registry with content hashes and approval workflow.

The paper's governance stage calls for model registries, versioning and
audit trails (Section 4.3). This registry stores, per version:

- the serialized model + its SHA-256 (integrity),
- the data hash it was trained on (lineage),
- metrics, fairness summary and gate outcomes (evidence),
- the generated model card and explanation artifacts,
- an approvals list with actor + timestamp (accountability).

It deliberately has no external service dependencies; swapping in MLflow or
a cloud registry is a production hardening step (see docs/architecture.md).
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypedDict, cast

import joblib
import pandas as pd

from feg_mlops import __version__

GENESIS = "0" * 64


class ModelPayload(TypedDict, total=False):
    """The serialized bundle stored per model version."""

    estimator: Any  # fitted sklearn estimator with predict_proba
    threshold: float
    group_thresholds: dict[str, float] | None
    feature_columns: list[str]
    protected_attribute: str
    decisions: dict[str, Any]
    background: pd.DataFrame
    package_version: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def frame_hash(df: pd.DataFrame) -> str:
    """Deterministic hash of a dataframe's content (order-insensitive rows)."""
    canonical = (
        df.astype(str)
        .sort_index(axis=1)
        .map(lambda v: v if v != "nan" else "<NA>")
        .agg("|".join, axis=1)
        .sort_values()
    )
    return sha256_bytes("\n".join(canonical).encode("utf-8"))


def git_commit_sha() -> str:
    """Best-effort commit SHA of the working tree ('uncommitted' if none)."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        return out.stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return "uncommitted"


@dataclass(frozen=True)
class RegisteredModel:
    name: str
    version: int
    manifest_path: Path
    manifest: dict[str, object]

    @property
    def status(self) -> str:
        return str(self.manifest["status"])

    def model_path(self, registry_root: Path) -> Path:
        return registry_root / self.name / f"v{self.version}" / "model.joblib"

    def load(self, registry_root: Path) -> ModelPayload:
        payload = joblib.load(self.model_path(registry_root))
        assert isinstance(payload, dict)
        return cast(ModelPayload, payload)


class ModelRegistry:
    """Versioned, content-addressed model store under ``root``."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------

    def register(
        self,
        name: str,
        model_payload: ModelPayload,
        artifacts: dict[str, str] | None = None,
        metrics: dict[str, object] | None = None,
        fairness: dict[str, object] | None = None,
        gates: dict[str, object] | None = None,
        data_hash: str = "",
        notes: str = "",
    ) -> RegisteredModel:
        """Persist a new model version with full lineage metadata."""
        version = self._next_version(name)
        version_dir = self.root / name / f"v{version}"
        version_dir.mkdir(parents=True, exist_ok=True)

        model_path = version_dir / "model.joblib"
        joblib.dump(model_payload, model_path)

        artifact_entries: dict[str, dict[str, str]] = {}
        for filename, content in (artifacts or {}).items():
            artifact_path = version_dir / filename
            artifact_path.write_text(content, encoding="utf-8")
            artifact_entries[filename] = {
                "sha256": sha256_file(artifact_path),
            }

        manifest = {
            "name": name,
            "version": version,
            "status": "candidate",
            "created_at": datetime.now(UTC).isoformat(),
            "package_version": __version__,
            "git_sha": git_commit_sha(),
            "data_hash": data_hash,
            "model_sha256": sha256_file(model_path),
            "metrics": metrics or {},
            "fairness": fairness or {},
            "gates": gates or {},
            "artifacts": artifact_entries,
            "notes": notes,
            "approvals": [],
        }
        manifest_path = version_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        return RegisteredModel(
            name=name, version=version, manifest_path=manifest_path, manifest=manifest
        )

    def get(self, name: str, version: int) -> RegisteredModel:
        manifest_path = self.root / name / f"v{version}" / "manifest.json"
        if not manifest_path.exists():
            raise KeyError(f"model {name} v{version} not found")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return RegisteredModel(name, version, manifest_path, manifest)

    def latest(self, name: str) -> RegisteredModel:
        return self.get(name, self._next_version(name) - 1)

    def latest_approved(self, name: str) -> RegisteredModel:
        """Newest approved version; falls back to newest candidate."""
        for version in range(self._next_version(name) - 1, 0, -1):
            model = self.get(name, version)
            if model.status == "approved":
                return model
        return self.latest(name)

    def list_versions(self, name: str) -> list[RegisteredModel]:
        models = []
        for version in range(1, self._next_version(name)):
            models.append(self.get(name, version))
        return models

    def approve(
        self,
        name: str,
        version: int,
        actor: str,
        reason: str,
    ) -> RegisteredModel:
        """Record a human approval — the accountable sign-off gate."""
        model = self.get(name, version)
        manifest = dict(model.manifest)
        manifest["status"] = "approved"
        approvals = cast(list[dict[str, str]], manifest.get("approvals", []))
        approvals = list(approvals)
        approvals.append(
            {
                "actor": actor,
                "reason": reason,
                "at": datetime.now(UTC).isoformat(),
            }
        )
        manifest["approvals"] = approvals
        model.manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
        )
        return self.get(name, version)

    def verify_integrity(self, name: str, version: int) -> bool:
        """Recompute the model hash to detect tampering."""
        model = self.get(name, version)
        return sha256_file(model.model_path(self.root)) == model.manifest["model_sha256"]

    # ------------------------------------------------------------------

    def _next_version(self, name: str) -> int:
        model_dir = self.root / name
        if not model_dir.exists():
            return 1
        versions = [
            int(p.name[1:])
            for p in model_dir.iterdir()
            if p.is_dir() and p.name.startswith("v") and p.name[1:].isdigit()
        ]
        return max(versions, default=0) + 1
