"""
backend/database.py
─────────────────────────────────────────────────────────────
SQLAlchemy 2.0 database layer.

Provides:
  • Declarative Base          – shared by all ORM models
  • Sync engine + session     – used by Alembic migrations & scripts
  • Async engine + session    – used by FastAPI request handlers

Session lifecycle
-----------------
FastAPI handlers should use `get_async_session` as a dependency.
Scripts / tasks should use `get_sync_session` as a context manager.
Alembic uses `sync_engine` directly via `alembic/env.py`.
"""

from __future__ import annotations

from contextlib import asynccontextmanager, contextmanager
from typing import AsyncGenerator, Generator

from sqlalchemy import create_engine, event, text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker
from sqlalchemy.pool import NullPool, QueuePool

from backend.config import get_settings

settings = get_settings()


# ─────────────────────────────────────────────────────────────────────────────
# Declarative Base
# Every ORM model inherits from this single Base so Alembic can auto-detect
# all table definitions via metadata.
# ─────────────────────────────────────────────────────────────────────────────

class Base(DeclarativeBase):
    """
    Shared declarative base for all ORM models.

    Provides a unified `metadata` object that Alembic reads for migrations.
    """
    pass


# ─────────────────────────────────────────────────────────────────────────────
# Synchronous Engine (Alembic + CLI scripts)
# ─────────────────────────────────────────────────────────────────────────────

sync_engine = create_engine(
    settings.database_url,
    poolclass=QueuePool,
    pool_size=5,
    max_overflow=10,
    pool_pre_ping=True,       # Validate connections before checkout
    pool_recycle=3600,        # Recycle connections after 1 hour
    echo=settings.DEBUG,      # Log SQL only in DEBUG mode
)

SyncSessionLocal = sessionmaker(
    bind=sync_engine,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False,   # Prevent implicit re-queries after commit
)


@contextmanager
def get_sync_session() -> Generator[Session, None, None]:
    """
    Context manager for synchronous database sessions.

    Usage (scripts / CLI):
        with get_sync_session() as session:
            session.add(obj)
            session.commit()
    """
    session: Session = SyncSessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# ─────────────────────────────────────────────────────────────────────────────
# Asynchronous Engine (FastAPI request handlers)
# ─────────────────────────────────────────────────────────────────────────────

async_engine: AsyncEngine = create_async_engine(
    settings.async_database_url,
    poolclass=NullPool,        # NullPool is recommended for async to avoid
                               # connection-leaking under async workloads
    echo=settings.DEBUG,
)

AsyncSessionLocal = async_sessionmaker(
    bind=async_engine,
    class_=AsyncSession,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False,
)


@asynccontextmanager
async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Async context manager for database sessions.

    Use as a FastAPI dependency (via `Depends`) or directly in background tasks:

        async with get_async_session() as session:
            result = await session.execute(select(Vendor))
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


# ── FastAPI Dependency ─────────────────────────────────────────────────────────

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency that yields an AsyncSession per request.

    Usage in a router:
        @router.get("/vendors")
        async def list_vendors(db: AsyncSession = Depends(get_db)):
            ...
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


# ─────────────────────────────────────────────────────────────────────────────
# Health Check
# ─────────────────────────────────────────────────────────────────────────────

async def check_database_health() -> bool:
    """
    Lightweight async health check – executes `SELECT 1`.

    Returns True if the database is reachable, False otherwise.
    Used by the /health endpoint in Sprint 5.
    """
    try:
        async with async_engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False
