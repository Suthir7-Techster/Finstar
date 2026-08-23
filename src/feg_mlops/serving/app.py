"""FastAPI KYC decision service.

Run: ``uvicorn feg_mlops.serving.app:app`` (after ``feg train``).

Endpoints
---------
- ``POST /v1/decisions``         — score one applicant; returns decision band,
                                   reason codes, counterfactual, audit id
- ``GET  /v1/reviews``           — human-review queue (refer band)
- ``POST /v1/reviews/{id}/resolve`` — underwriter resolution
- ``GET  /v1/models``            — registered versions with fairness evidence
- ``GET  /v1/models/{version}``  — manifest + model card for one version
- ``GET  /healthz`` / ``GET /readyz``
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Response, status

from feg_mlops.config import FegConfig, load_configs
from feg_mlops.modeling.registry import ModelRegistry
from feg_mlops.serving.decision import DecisionEngine, ServingError
from feg_mlops.serving.schemas import (
    DecisionRequest,
    DecisionResponse,
    ModelVersionOut,
    ResolveReviewRequest,
    ReviewOut,
)

DEFAULT_CONFIG_DIR = os.environ.get("FEG_CONFIG_DIR", "configs")
DEFAULT_ARTIFACTS_DIR = os.environ.get("FEG_ARTIFACTS_DIR", "artifacts")


def _string_keyed(value: object) -> dict[str, Any]:
    """Coerce a JSON-loaded manifest section to ``dict[str, Any]``."""
    return dict(value) if isinstance(value, dict) else {}


def create_app(
    config_dir: str | Path = DEFAULT_CONFIG_DIR,
    artifacts_dir: str | Path = DEFAULT_ARTIFACTS_DIR,
) -> FastAPI:
    """Application factory. Falls back to a 503-ready app when no model exists."""
    registry = ModelRegistry(Path(artifacts_dir) / "registry")
    engine: DecisionEngine | None = None
    startup_error: str | None = None
    config: FegConfig | None = None
    try:
        config = load_configs(config_dir)
        engine = DecisionEngine(config, artifacts_dir)
    except (ServingError, OSError, ValueError) as exc:
        startup_error = str(exc)

    app = FastAPI(
        title="feg-mlops KYC decision service",
        version="0.1.0",
        description=(
            "Responsible KYC/credit decisioning API: every decision is "
            "explainable (SHAP reason codes + counterfactuals), audited "
            "(hash-chained trail) and banded for human review."
        ),
    )
    app.state.engine = engine
    app.state.config = config
    app.state.registry = registry

    def require_engine() -> DecisionEngine:
        if engine is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=startup_error or "model not loaded",
            )
        return engine

    @app.get("/healthz", tags=["ops"])
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz", tags=["ops"])
    def readyz(response: Response) -> dict[str, str]:
        if engine is None:
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
            return {"status": "not-ready", "reason": startup_error or "no model"}
        return {"status": "ready", "model_version": str(engine.model.version)}

    @app.post("/v1/decisions", response_model=DecisionResponse, tags=["decisions"])
    def decide(request: DecisionRequest) -> DecisionResponse:
        return require_engine().decide(
            request.features.model_dump(), applicant_ref=request.applicant_ref
        )

    @app.get("/v1/reviews", response_model=list[ReviewOut], tags=["reviews"])
    def reviews(status_filter: str | None = Query(default=None, alias="status")) -> list[ReviewOut]:
        return require_engine().list_reviews(status=status_filter)

    @app.post("/v1/reviews/{review_id}/resolve", response_model=ReviewOut, tags=["reviews"])
    def resolve_review(review_id: str, request: ResolveReviewRequest) -> ReviewOut:
        try:
            return require_engine().resolve_review(
                review_id, request.outcome, request.reviewer, request.note
            )
        except KeyError:
            raise HTTPException(status_code=404, detail=f"review {review_id} not found") from None
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.get("/v1/models", response_model=list[ModelVersionOut], tags=["models"])
    def models() -> list[ModelVersionOut]:
        try:
            versions = registry.list_versions("kyc_risk_model")
        except KeyError:
            return []
        return [
            ModelVersionOut(
                name=v.name,
                version=v.version,
                status=v.status,
                created_at=str(v.manifest.get("created_at", "")),
                metrics=_string_keyed(v.manifest.get("metrics")),
                fairness=_string_keyed(v.manifest.get("fairness")),
            )
            for v in versions
        ]

    @app.get("/v1/models/{version}", tags=["models"])
    def model_detail(version: int) -> dict[str, Any]:
        try:
            model = registry.get("kyc_risk_model", version)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"version {version} not found") from None
        card_path = model.manifest_path.parent / "model_card.md"
        card = card_path.read_text(encoding="utf-8") if card_path.exists() else ""
        return {"manifest": model.manifest, "model_card": card}

    return app


app = create_app()
