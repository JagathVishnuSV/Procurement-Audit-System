"""
backend/api/routers/health.py
─────────────────────────────
GET /api/v1/health  –  liveness + readiness probe.
"""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from backend.ml import scorer as scorer_module

router = APIRouter()


class HealthResponse(BaseModel):
    status: str
    ml_model_loaded: bool


@router.get("/health", response_model=HealthResponse, tags=["health"])
async def health_check() -> HealthResponse:
    """Liveness + readiness probe. Returns 200 once the ML model is loaded."""
    singleton = scorer_module._scorer_singleton
    loaded = singleton is not None and singleton._loaded
    return HealthResponse(status="ok", ml_model_loaded=loaded)
