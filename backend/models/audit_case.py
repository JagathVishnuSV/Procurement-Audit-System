"""
backend/models/audit_case.py
─────────────────────────────────────────────────────────────
AuditCase ORM model.

An AuditCase is created when the ML pipeline's IsolationForest
score exceeds the anomaly threshold. It aggregates:

  • ML Layer output  – ml_score, shap_reason (JSON waterfall data)
  • Groq Triage      – groq_verdict JSON {"escalate": bool, "reason": str}
  • Gemini Deep Audit– gemini_report JSON {"verdict": ..., "violated_clause": ...}

Status flow:
  OPEN → IN_REVIEW → CLOSED (resolved or dismissed)

This table is the source of truth for the Audit Inbox Kanban board.
"""

from __future__ import annotations

import enum
import uuid
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from sqlalchemy import Enum as SAEnum, ForeignKey, String, Text
from sqlalchemy import Index
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.database import Base
from backend.models.mixins import TimestampMixin, UUIDPrimaryKeyMixin

if TYPE_CHECKING:
    from backend.models.transaction import Transaction
    from backend.models.action_plan import ActionPlan


class AuditCaseStatus(str, enum.Enum):
    """Kanban column state for the Audit Inbox."""
    OPEN = "OPEN"               # Newly flagged, awaiting human review
    IN_REVIEW = "IN_REVIEW"     # Auditor has opened the case
    CLOSED = "CLOSED"           # Resolved (fraud confirmed or dismissed)


class LLMVerdict(str, enum.Enum):
    """
    Final verdict from the Gemini Deep Audit layer.
    This is the human-readable conclusion stored in the database.
    """
    FRAUD = "FRAUD"
    SUSPICIOUS = "SUSPICIOUS"
    NORMAL = "NORMAL"
    INCONCLUSIVE = "INCONCLUSIVE"


class AuditCase(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """
    Represents a full forensic audit case for a flagged transaction.

    The Evidence Card displayed in the Forensic Case Workspace is
    assembled from:
        - transaction (via FK)
        - shap_reason  → SHAP Waterfall chart
        - groq_verdict → Triage reasoning
        - gemini_report → Contract clause cross-reference
        - contract_clause_cited → Exact PDF text highlighted in UI
    """

    __tablename__ = "audit_cases"

    # ── Foreign Keys ─────────────────────────────────────────────────────────
    transaction_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("transactions.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,    # One audit case per transaction (1:1)
        index=True,
    )

    # ── ML Layer output ───────────────────────────────────────────────────────
    ml_score: Mapped[float] = mapped_column(
        nullable=False,
        comment="IsolationForest decision score (lower = more anomalous)",
    )
    shap_reason: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSONB,
        nullable=True,
        comment=(
            "SHAP feature attributions as JSON, e.g. "
            "{'amount': 4.2, 'vendor_30d_velocity': 1.8}"
        ),
    )
    shap_summary: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        comment="Human-readable SHAP summary, e.g. 'Amount is 4.2x vendor average'",
    )

    # ── Groq Triage output ────────────────────────────────────────────────────
    groq_verdict: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSONB,
        nullable=True,
        comment='Groq JSON output: {"escalate": true, "reason": "..."}',
    )
    groq_escalated: Mapped[Optional[bool]] = mapped_column(
        nullable=True,
        comment="Denormalized field: True if Groq recommended deep audit",
    )

    # ── Gemini Deep Audit output ──────────────────────────────────────────────
    llm_verdict: Mapped[Optional[LLMVerdict]] = mapped_column(
        SAEnum(LLMVerdict, name="llm_verdict_enum", create_type=True),
        nullable=True,
        index=True,
        comment="Final verdict from Gemini deep audit",
    )
    gemini_report: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSONB,
        nullable=True,
        comment=(
            'Gemini JSON: {"verdict": "FRAUD", '
            '"violated_clause": "Section 4a", "confidence": 0.95}'
        ),
    )
    confidence: Mapped[Optional[float]] = mapped_column(
        nullable=True,
        comment="Gemini confidence score [0.0 – 1.0]",
    )
    contract_clause_cited: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        comment="Exact contract clause text retrieved from FAISS and cited by Gemini",
    )
    violated_clause_id: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
        comment="Section reference, e.g. 'Section 4a'",
    )

    # ── Case management ────────────────────────────────────────────────────────
    status: Mapped[AuditCaseStatus] = mapped_column(
        SAEnum(AuditCaseStatus, name="audit_case_status_enum", create_type=True),
        nullable=False,
        default=AuditCaseStatus.OPEN,
        server_default=AuditCaseStatus.OPEN.value,
        index=True,
    )
    risk_level: Mapped[Optional[str]] = mapped_column(
        String(10),
        nullable=True,
        comment="Derived risk label: HIGH / MEDIUM / LOW (for Kanban card display)",
    )
    estimated_impact_usd: Mapped[Optional[float]] = mapped_column(
        nullable=True,
        comment="Estimated financial risk in USD (displayed on Kanban card)",
    )
    auditor_notes: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        comment="Freeform notes added by the human auditor",
    )

    # ── Relationships ─────────────────────────────────────────────────────────
    transaction: Mapped["Transaction"] = relationship(
        "Transaction",
        back_populates="audit_case",
        lazy="select",
    )
    action_plans: Mapped[List["ActionPlan"]] = relationship(
        "ActionPlan",
        back_populates="audit_case",
        cascade="all, delete-orphan",
        lazy="select",
    )

    # ── Indexes ───────────────────────────────────────────────────────────────
    __table_args__ = (
        Index("ix_audit_cases_status_verdict", "status", "llm_verdict"),
        Index("ix_audit_cases_status_score", "status", "ml_score"),
    )

    def __repr__(self) -> str:
        return (
            f"<AuditCase id={self.id} status={self.status.value} "
            f"verdict={self.llm_verdict} ml_score={self.ml_score:.4f}>"
        )
