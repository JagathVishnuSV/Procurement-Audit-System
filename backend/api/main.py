"""
backend/api/main.py
──────────────────────────────────────────────────────────────────────────────
FastAPI application factory.

Startup:  loads the IsolationForest model into memory (sub-100ms)
Shutdown: graceful cleanup

Run:
    uvicorn backend.api.main:app --reload --port 8000
"""

from __future__ import annotations

from contextlib import asynccontextmanager
import asyncio
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from backend.config import get_settings
from backend.api.routers import health, score, contracts, cases, audit, action_plans, metrics, realtime, orchestration


# ── Lifespan ───────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load ML model and RAG components on startup; release on shutdown."""
    orchestration_task: asyncio.Task | None = None
    ingestion_task: asyncio.Task | None = None
    kafka_consumer_task: asyncio.Task | None = None
    kafka_consumer_instance = None

    logger.info("═══ Procurement Audit API starting ═══")

    # ML scorer (Sprint 2)
    try:
        from backend.ml.scorer import get_scorer
        get_scorer()   # pre-load — logs its own success message
    except FileNotFoundError as exc:
        logger.warning(
            "ML model not found: {}  →  /score will return 503 until trained.", exc
        )

    # RAG embedder + FAISS index (Sprint 3)
    try:
        from backend.rag.embedder import get_embedder
        from backend.rag.vector_store import get_vector_store
        get_embedder()       # loads sentence-transformers model
        get_vector_store()   # loads persisted FAISS index (or starts empty)
        logger.success("RAG embedder and FAISS vector store loaded")
    except Exception as exc:
        logger.warning(
            "RAG components failed to load: {}  →  /contracts/search will return 503", exc
        )

    settings = get_settings()
    if settings.ORCH_ENABLED:
        from backend.orchestration import get_orchestration_engine

        async def orchestration_loop() -> None:
            engine = get_orchestration_engine()
            while True:
                try:
                    summary = await engine.run_once(batch_size=settings.ORCH_BATCH_SIZE)
                    logger.info(
                        "Orchestration run completed: scored={} cases_created={} high={} med={} low={}",
                        summary.scored,
                        summary.cases_created,
                        summary.high_risk,
                        summary.medium_risk,
                        summary.low_risk,
                    )
                except Exception as exc:
                    logger.warning("Orchestration run failed: {}", exc)
                await asyncio.sleep(max(10, settings.ORCH_INTERVAL_SECONDS))

        orchestration_task = asyncio.create_task(orchestration_loop())
        logger.success(
            "Auto-orchestration enabled (interval={}s, batch_size={})",
            settings.ORCH_INTERVAL_SECONDS,
            settings.ORCH_BATCH_SIZE,
        )

    if settings.INGESTION_LOOP_ENABLED:
        from backend.ingestion.realtime_pipeline import NearRealtimeIngestionPipeline

        async def ingestion_loop() -> None:
            pipeline = NearRealtimeIngestionPipeline(
                poll_interval_seconds=settings.INGESTION_POLL_INTERVAL_SECONDS,
            )
            while True:
                try:
                    await pipeline.run_once(limit_per_source=settings.INGESTION_LIMIT_PER_SOURCE)
                except Exception as exc:
                    logger.warning("Near-realtime ingestion cycle failed: {}", exc)
                await asyncio.sleep(max(30, settings.INGESTION_POLL_INTERVAL_SECONDS))

        ingestion_task = asyncio.create_task(ingestion_loop())
        logger.success(
            "Near-realtime ingestion enabled (interval={}s, limit_per_source={})",
            settings.INGESTION_POLL_INTERVAL_SECONDS,
            settings.INGESTION_LIMIT_PER_SOURCE,
        )

    if settings.KAFKA_CONSUMER_ENABLED:
        from backend.ingestion.kafka_consumer import ProcurementKafkaConsumer

        kafka_consumer_instance = ProcurementKafkaConsumer()

        async def kafka_consumer_loop() -> None:
            await kafka_consumer_instance.start()
            await kafka_consumer_instance.run_forever()

        kafka_consumer_task = asyncio.create_task(kafka_consumer_loop())
        logger.success(
            "Kafka raw-ingestion consumer enabled (topic={}, group={})",
            settings.KAFKA_TOPIC_RAW_TRANSACTIONS,
            settings.KAFKA_CONSUMER_GROUP,
        )

    yield  # ← API is live

    if orchestration_task:
        orchestration_task.cancel()
        try:
            await orchestration_task
        except asyncio.CancelledError:
            pass

    if ingestion_task:
        ingestion_task.cancel()
        try:
            await ingestion_task
        except asyncio.CancelledError:
            pass

    if kafka_consumer_task:
        kafka_consumer_task.cancel()
        try:
            await kafka_consumer_task
        except asyncio.CancelledError:
            pass

    if kafka_consumer_instance is not None:
        try:
            await kafka_consumer_instance.stop()
        except Exception:
            pass

    logger.info("═══ Procurement Audit API shutting down ═══")


# ── Application factory ────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="Procurement Audit API",
        version="5.0.0",
        description=(
            "Intelligent Procurement Audit and Anomaly Detection System — "
            "Sprint 5: Case Workspace + Executive Metrics APIs"
        ),
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router,     prefix="/api/v1")
    app.include_router(score.router,      prefix="/api/v1")
    app.include_router(contracts.router,  prefix="/api/v1")
    app.include_router(cases.router,      prefix="/api/v1")
    app.include_router(audit.router,      prefix="/api/v1")
    app.include_router(action_plans.router, prefix="/api/v1")
    app.include_router(metrics.router,      prefix="/api/v1")
    app.include_router(realtime.router,     prefix="/api/v1")
    app.include_router(orchestration.router, prefix="/api/v1")

    return app


app = create_app()
