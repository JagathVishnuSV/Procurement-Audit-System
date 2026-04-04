"""
backend/models/mixins.py
─────────────────────────────────────────────────────────────
Reusable SQLAlchemy 2.0 column mixins.

TimestampMixin – adds `created_at` / `updated_at` to any model.
UUIDMixin      – adds a UUID primary key column.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime, func
from sqlalchemy.orm import Mapped, mapped_column


def _utcnow() -> datetime:
    """Return timezone-aware UTC datetime."""
    return datetime.now(timezone.utc)


class TimestampMixin:
    """
    Adds `created_at` and `updated_at` columns to a model.

    `created_at` is set once on INSERT.
    `updated_at` is refreshed automatically on every UPDATE via server_default
    and `onupdate`.
    """

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=_utcnow,
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=_utcnow,
        server_default=func.now(),
        onupdate=_utcnow,
        nullable=False,
    )


class UUIDPrimaryKeyMixin:
    """
    Adds a UUID v4 primary key column named `id`.

    Using UUID as PK is preferable to integer sequences in distributed
    systems because IDs are globally unique and can be generated client-side.
    """

    id: Mapped[uuid.UUID] = mapped_column(
        primary_key=True,
        default=uuid.uuid4,
        index=True,
    )
