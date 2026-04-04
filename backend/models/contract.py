"""
backend/models/contract.py
─────────────────────────────────────────────────────────────
Contract ORM model (CLM – Contract Lifecycle Management).

When a PDF contract is uploaded via POST /api/contracts/upload:
  1. The file is stored on disk (or S3 in production).
  2. The PDF is semantically chunked by LangChain.
  3. Chunks are embedded (all-MiniLM-L6-v2) and stored in FAISS.
  4. A Contract row is created here linking the vendor to the
     FAISS index entry so audit queries can retrieve relevant clauses.

`faiss_index_id` is the key used to scope FAISS searches to a
specific contract's vector embeddings.
"""

from __future__ import annotations

import enum
import uuid
from datetime import datetime
from typing import TYPE_CHECKING, List, Optional

from sqlalchemy import DateTime, Enum as SAEnum, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.database import Base
from backend.models.mixins import TimestampMixin, UUIDPrimaryKeyMixin

if TYPE_CHECKING:
    from backend.models.vendor import Vendor
    from backend.models.audit_case import AuditCase


class ContractStatus(str, enum.Enum):
    """Lifecycle state of a vendor contract."""
    ACTIVE = "ACTIVE"
    EXPIRED = "EXPIRED"
    TERMINATED = "TERMINATED"
    UNDER_REVIEW = "UNDER_REVIEW"
    DRAFT = "DRAFT"


class Contract(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """
    Represents a vendor contract document stored in the CLM.

    RAG fields
    ----------
    faiss_index_id  – identifier scoping FAISS vectors to this contract.
    chunk_count     – number of semantic chunks stored in FAISS.
    embedding_model – embedding model version used (for reproducibility).
    """

    __tablename__ = "contracts"

    # ── Foreign Keys ─────────────────────────────────────────────────────────
    vendor_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("vendors.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # ── Contract metadata ─────────────────────────────────────────────────────
    title: Mapped[str] = mapped_column(
        String(500),
        nullable=False,
        comment="Human-readable contract title",
    )
    contract_number: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
        unique=True,
        comment="Official contract reference number",
    )
    file_path: Mapped[Optional[str]] = mapped_column(
        String(1024),
        nullable=True,
        comment="Relative path to the stored PDF file",
    )
    file_hash: Mapped[Optional[str]] = mapped_column(
        String(64),
        nullable=True,
        comment="SHA-256 of the PDF – used to detect re-uploads",
    )
    upload_date: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        comment="When the PDF was uploaded into the system",
    )
    effective_date: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="Date from which the contract is legally binding",
    )
    expiry_date: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
        comment="Contract expiry date – used to flag at-risk renewals",
    )
    total_value: Mapped[Optional[float]] = mapped_column(
        nullable=True,
        comment="Total contract value in USD",
    )
    status: Mapped[ContractStatus] = mapped_column(
        SAEnum(ContractStatus, name="contract_status_enum", create_type=True),
        nullable=False,
        default=ContractStatus.ACTIVE,
        server_default=ContractStatus.ACTIVE.value,
        index=True,
    )
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # ── RAG / FAISS fields ────────────────────────────────────────────────────
    faiss_index_id: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        unique=True,
        comment="Key used to retrieve this contract's vectors from FAISS",
    )
    chunk_count: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
        comment="Number of semantic chunks stored in FAISS for this contract",
    )
    embedding_model: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        comment="Embedding model version used for RAG (reproducibility)",
    )
    is_indexed: Mapped[bool] = mapped_column(
        nullable=False,
        default=False,
        server_default="false",
        comment="True once the PDF has been chunked and loaded into FAISS",
    )

    # ── Relationships ─────────────────────────────────────────────────────────
    vendor: Mapped["Vendor"] = relationship(
        "Vendor",
        back_populates="contracts",
        lazy="select",
    )

    def __repr__(self) -> str:
        return (
            f"<Contract id={self.id} title={self.title!r} "
            f"status={self.status.value} vendor_id={self.vendor_id}>"
        )
