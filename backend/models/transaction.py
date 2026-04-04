"""
backend/models/transaction.py
─────────────────────────────────────────────────────────────
Transaction ORM model.

Transactions are the primary input to the ML anomaly scoring
pipeline. They flow through the Kafka topics in this order:

  raw_transactions → enriched_transactions → anomalies_topic

Pipeline state is tracked via `is_enriched`, `is_scored`, and
`ml_score` columns so that any microservice failure can resume
processing from the correct Kafka offset.

External source data (full API JSON) is preserved in `raw_data`
(JSONB) for auditability – never discard original payloads.
"""

from __future__ import annotations

import enum
import uuid
from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from sqlalchemy import DateTime, Enum as SAEnum, ForeignKey, Numeric, String, Text
from sqlalchemy import Index
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.database import Base
from backend.models.mixins import TimestampMixin, UUIDPrimaryKeyMixin

if TYPE_CHECKING:
    from backend.models.vendor import Vendor
    from backend.models.audit_case import AuditCase


class TransactionSource(str, enum.Enum):
    """Origin of the transaction record."""
    USASPENDING = "USASPENDING"    # USAspending.gov API
    CPPP = "CPPP"                  # India Central Public Procurement Portal
    OCDS = "OCDS"                  # Open Contracting Data Standard feeds
    WORLDBANK = "WORLDBANK"        # World Bank Procurement API
    SEC_EDGAR = "SEC_EDGAR"        # SEC EDGAR vendor agreements
    NYC_OPENDATA = "NYC_OPENDATA"  # NYC Open Data contracts
    MANUAL = "MANUAL"              # Manually entered by an auditor


class Transaction(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """
    Represents a single procurement transaction (invoice or PO).

    ML Pipeline State Tracking
    --------------------------
    is_enriched  – True after Flink updates Redis feature store
    is_scored    – True after IsolationForest scores the record
    ml_score     – Raw anomaly score from IsolationForest
                   (negative = more anomalous; threshold from config)
    """

    __tablename__ = "transactions"

    # ── Foreign Keys ─────────────────────────────────────────────────────────
    vendor_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("vendors.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # ── Transaction identity ──────────────────────────────────────────────────
    external_id: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        unique=True,
        index=True,
        comment="Unique ID from the source API (deduplication key)",
    )
    source: Mapped[TransactionSource] = mapped_column(
        SAEnum(TransactionSource, name="transaction_source_enum", create_type=True),
        nullable=False,
        default=TransactionSource.USASPENDING,
        index=True,
    )

    # ── Financial data ────────────────────────────────────────────────────────
    amount: Mapped[Decimal] = mapped_column(
        Numeric(precision=18, scale=2),
        nullable=False,
        comment="Transaction amount in USD",
    )
    currency: Mapped[str] = mapped_column(
        String(3),
        nullable=False,
        default="USD",
        server_default="USD",
    )
    date: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
        comment="Transaction date (action date from source API)",
    )
    fiscal_year: Mapped[Optional[int]] = mapped_column(
        nullable=True,
        comment="US Federal fiscal year (Oct–Sep)",
    )

    # ── Classification ────────────────────────────────────────────────────────
    category: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        index=True,
        comment="Product/service category (e.g., IT Services, Construction)",
    )
    award_type: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
        comment="Award type code from USAspending (A/B/C/D/etc.)",
    )
    description: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        comment="Transaction description from source API",
    )
    awarding_agency: Mapped[Optional[str]] = mapped_column(
        String(500),
        nullable=True,
        comment="Federal agency that issued the award",
    )

    # ── ML Pipeline State ─────────────────────────────────────────────────────
    is_enriched: Mapped[bool] = mapped_column(
        nullable=False,
        default=False,
        server_default="false",
        comment="True after Flink enriches with 30-day vendor velocity",
    )
    is_scored: Mapped[bool] = mapped_column(
        nullable=False,
        default=False,
        server_default="false",
        index=True,
        comment="True after IsolationForest scoring",
    )
    ml_score: Mapped[Optional[float]] = mapped_column(
        nullable=True,
        comment="IsolationForest anomaly score (lower = more anomalous)",
    )

    # ── Raw payload (audit trail) ─────────────────────────────────────────────
    raw_data: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSONB,
        nullable=True,
        comment="Full original API response payload – never discard",
    )

    # ── Relationships ─────────────────────────────────────────────────────────
    vendor: Mapped["Vendor"] = relationship(
        "Vendor",
        back_populates="transactions",
        lazy="select",
    )
    audit_case: Mapped[Optional["AuditCase"]] = relationship(
        "AuditCase",
        back_populates="transaction",
        uselist=False,   # One-to-one
        lazy="select",
    )

    # ── Composite indexes for common query patterns ───────────────────────────
    __table_args__ = (
        Index("ix_transactions_vendor_date", "vendor_id", "date"),
        Index("ix_transactions_scored_score", "is_scored", "ml_score"),
        Index("ix_transactions_category_date", "category", "date"),
    )

    def __repr__(self) -> str:
        return (
            f"<Transaction id={self.id} amount={self.amount} "
            f"vendor_id={self.vendor_id} date={self.date.date()}>"
        )
