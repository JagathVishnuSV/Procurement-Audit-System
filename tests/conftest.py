"""
tests/conftest.py
─────────────────────────────────────────────────────────────
Shared pytest fixtures for the Procurement Audit System test suite.

Design decisions
────────────────
1. UNIT tests use SQLite in-memory to run fast without Docker.
   PostgreSQL-specific types (JSONB, UUID) are mapped to SQLite
   equivalents via a custom TypeDecorator shim.

2. INTEGRATION tests (marked `@pytest.mark.integration`) use the
   REAL PostgreSQL URL from .env and are skipped by default.
   Run with: pytest -m integration

3. Settings are patched in unit tests to prevent accidental
   production database connections.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from typing import AsyncGenerator, Generator
from unittest.mock import patch

import pytest
import pytest_asyncio
from sqlalchemy import create_engine, event, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from backend.config import Settings
from backend.database import Base


# ─────────────────────────────────────────────────────────────────────────────
# SQLite compatibility shim for JSONB columns
# ─────────────────────────────────────────────────────────────────────────────

def _patch_jsonb_for_sqlite() -> None:
    """
    Replace PostgreSQL JSONB with a JSON-serialising Text type for SQLite.

    This allows unit tests to run without PostgreSQL while keeping the
    same ORM models. Only affects the test process.
    """
    from sqlalchemy import Text
    from sqlalchemy.dialects.postgresql import JSONB as _JSONB
    from sqlalchemy.types import TypeDecorator

    class _SQLiteJSONB(TypeDecorator):
        """JSONB shim for SQLite: stores JSON as TEXT."""
        impl = Text
        cache_ok = True

        def process_bind_param(self, value, dialect):
            if value is None:
                return None
            return json.dumps(value, default=str)

        def process_result_value(self, value, dialect):
            if value is None:
                return None
            return json.loads(value)

    # Patch at import level so all models pick it up
    import sqlalchemy.dialects.postgresql as pg_module
    pg_module.JSONB = _SQLiteJSONB


_patch_jsonb_for_sqlite()

# Must import models AFTER patching JSONB
import backend.models  # noqa: F401, E402


# ─────────────────────────────────────────────────────────────────────────────
# Test settings override
# ─────────────────────────────────────────────────────────────────────────────

TEST_SETTINGS = Settings(
    APP_ENV="testing",
    DEBUG=True,
    LOG_LEVEL="DEBUG",
    POSTGRES_DB="test_procurement_audit",
    KAFKA_BOOTSTRAP_SERVERS="localhost:19092",
    GROQ_API_KEY="",
    GEMINI_API_KEY="",
)


@pytest.fixture(autouse=True)
def patch_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Auto-use fixture: overrides `get_settings()` for every test.
    Prevents accidental reads from the real .env file.
    """
    from backend import config as config_module
    # Clear lru_cache on the real function before monkeypatching
    real_fn = getattr(config_module.get_settings, "__wrapped__", config_module.get_settings)
    if hasattr(real_fn, "cache_clear"):
        real_fn.cache_clear()
    monkeypatch.setattr(config_module, "get_settings", lambda: TEST_SETTINGS)
    yield
    # Restore cache-clear on the original lru_cache function
    if hasattr(real_fn, "cache_clear"):
        real_fn.cache_clear()


# ─────────────────────────────────────────────────────────────────────────────
# SQLite in-memory engine (unit tests)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def sqlite_engine():
    """
    Session-scoped SQLite in-memory engine.
    All tables are created once and shared across all unit tests.
    """
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        echo=False,
    )

    # Enable foreign key enforcement in SQLite
    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    Base.metadata.create_all(engine)
    yield engine
    Base.metadata.drop_all(engine)
    engine.dispose()


@pytest.fixture
def db_session(sqlite_engine) -> Generator[Session, None, None]:
    """
    Function-scoped database session.
    Each test gets a clean transaction that is rolled back on teardown.
    This ensures test isolation without re-creating tables.
    """
    connection = sqlite_engine.connect()
    transaction = connection.begin()
    session_factory = sessionmaker(bind=connection, expire_on_commit=False)
    session = session_factory()

    yield session

    session.close()
    transaction.rollback()
    connection.close()


# ─────────────────────────────────────────────────────────────────────────────
# Model factories (reusable builders for test data)
# ─────────────────────────────────────────────────────────────────────────────

class VendorFactory:
    """Builds Vendor instances with sensible defaults."""

    @staticmethod
    def build(
        name: str = "Acme Corporation",
        risk_tier=None,
        total_spend_ytd: Decimal = Decimal("50000.00"),
    ):
        from backend.models.vendor import Vendor, RiskTier
        risk_tier = risk_tier or RiskTier.LOW
        return Vendor(
            name=name,
            normalized_name=name.lower().strip(),
            risk_tier=risk_tier,
            total_spend_ytd=total_spend_ytd,
            country="USA",
        )

    @staticmethod
    def create(session: Session, **kwargs) -> "Vendor":
        from backend.models.vendor import Vendor
        vendor = VendorFactory.build(**kwargs)
        session.add(vendor)
        session.flush()
        return vendor


class TransactionFactory:
    """Builds Transaction instances with sensible defaults."""

    @staticmethod
    def build(
        vendor_id=None,
        amount: Decimal = Decimal("12500.00"),
        date: datetime = None,
        external_id: str = None,
    ):
        from backend.models.transaction import Transaction, TransactionSource
        return Transaction(
            vendor_id=vendor_id or uuid.uuid4(),
            external_id=external_id or f"TEST-{uuid.uuid4().hex[:8]}",
            source=TransactionSource.USASPENDING,
            amount=amount,
            currency="USD",
            date=date or datetime.now(timezone.utc),
            category="IT Services",
            description="Cloud computing services",
            awarding_agency="Department of Defense",
        )

    @staticmethod
    def create(session: Session, vendor_id=None, **kwargs) -> "Transaction":
        tx = TransactionFactory.build(vendor_id=vendor_id, **kwargs)
        session.add(tx)
        session.flush()
        return tx


@pytest.fixture
def vendor_factory() -> type:
    return VendorFactory


@pytest.fixture
def transaction_factory() -> type:
    return TransactionFactory


# ─────────────────────────────────────────────────────────────────────────────
# Async SQLite engine (for Sprint 3 RAG / API tests)
# ─────────────────────────────────────────────────────────────────────────────

@pytest_asyncio.fixture(scope="function")
async def async_sqlite_engine():
    """
    Function-scoped async SQLite in-memory engine for tests that use
    async sessions (CLMService, FastAPI route handlers).
    """
    from sqlalchemy.ext.asyncio import create_async_engine

    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        echo=False,
    )

    async with engine.begin() as conn:
        # aiosqlite doesn't support synchronous listeners, so we enable
        # foreign keys via a raw execute inside an async context.
        await conn.execute(text("PRAGMA foreign_keys=ON"))
        await conn.run_sync(Base.metadata.create_all)

    yield engine

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()
