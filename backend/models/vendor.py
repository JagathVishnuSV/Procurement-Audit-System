"""
backend/models/vendor.py
─────────────────────────────────────────────────────────────
Vendor ORM model.

Maps to the `vendors` table. Vendors are the root entity in the
procurement graph – all transactions and contracts link back here.

Risk Tiers (LOW → CRITICAL) drive routing priority in the audit
pipeline: CRITICAL vendors get expedited Gemini deep-audit.
"""

from __future__ import annotations

import enum
import uuid
from decimal import Decimal
from typing import TYPE_CHECKING, List, Optional

from sqlalchemy import Enum as SAEnum
from sqlalchemy import Index, Numeric, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.database import Base
from backend.models.mixins import TimestampMixin, UUIDPrimaryKeyMixin

if TYPE_CHECKING:
    from backend.models.contract import Contract
    from backend.models.transaction import Transaction


class RiskTier(str, enum.Enum):
    """
    Vendor-level risk classification.
    Updated by the ML layer after each scoring cycle.
    """
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class Vendor(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """
    Represents a procurement vendor / supplier.

    Relationships
    -------------
    contracts    – all contracts signed with this vendor
    transactions – all purchase orders / invoices from this vendor
    """

    __tablename__ = "vendors"

    # ── Core fields ─────────────────────────────────────────────────────────
    name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        unique=True,
        index=True,
        comment="Canonical vendor name (normalised from raw API data)",
    )
    normalized_name: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        index=True,
        comment="Lowercase/stripped name used for deduplication",
    )
    risk_tier: Mapped[RiskTier] = mapped_column(
        SAEnum(RiskTier, name="risk_tier_enum", create_type=True),
        nullable=False,
        default=RiskTier.LOW,
        server_default=RiskTier.LOW.value,
        index=True,
        comment="Risk classification driven by ML scoring history",
    )
    total_spend_ytd: Mapped[Decimal] = mapped_column(
        Numeric(precision=18, scale=2),
        nullable=False,
        default=Decimal("0.00"),
        server_default="0.00",
        comment="Year-to-date total procurement spend in USD",
    )
    country: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
        comment="Vendor country of registration (from USAspending data)",
    )
    duns_number: Mapped[Optional[str]] = mapped_column(
        String(50),
        nullable=True,
        unique=True,
        comment="DUNS/UEI number from federal procurement data",
    )
    notes: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    # ── Relationships ────────────────────────────────────────────────────────
    contracts: Mapped[List["Contract"]] = relationship(
        "Contract",
        back_populates="vendor",
        cascade="all, delete-orphan",
        lazy="select",
    )
    transactions: Mapped[List["Transaction"]] = relationship(
        "Transaction",
        back_populates="vendor",
        cascade="all, delete-orphan",
        lazy="select",
    )

    # ── Indexes ──────────────────────────────────────────────────────────────
    __table_args__ = (
        Index("ix_vendors_risk_tier_name", "risk_tier", "name"),
    )

    def __repr__(self) -> str:
        return (
            f"<Vendor id={self.id} name={self.name!r} "
            f"risk_tier={self.risk_tier.value}>"
        )
