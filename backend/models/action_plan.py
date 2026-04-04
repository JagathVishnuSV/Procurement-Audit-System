"""
backend/models/action_plan.py
─────────────────────────────────────────────────────────────
ActionPlan ORM model.

When an auditor reviews a forensic case and decides on a course
of action, they fill out the Action Plan form in the Case Workspace.

The `dollars_saved` field is the key ROI metric that feeds the
Executive Dashboard. Aggregated across all COMPLETED ActionPlans,
this is the "Cost Savings YTD" KPI shown on the dashboard.

Lifecycle:
  PENDING → IN_PROGRESS → COMPLETED | CANCELLED
"""

from __future__ import annotations

import enum
import uuid
from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Optional

from sqlalchemy import DateTime, Enum as SAEnum, ForeignKey, Numeric, String, Text
from sqlalchemy import Index
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.database import Base
from backend.models.mixins import TimestampMixin, UUIDPrimaryKeyMixin

if TYPE_CHECKING:
    from backend.models.audit_case import AuditCase


class ActionType(str, enum.Enum):
    """
    The type of corrective action assigned by the auditor.
    Maps to the dropdown options in the React Case Workspace form.
    """
    CLAWBACK = "CLAWBACK"                  # Recover overpaid funds
    PAYMENT_HALT = "PAYMENT_HALT"          # Stop a pending payment
    VENDOR_REVIEW = "VENDOR_REVIEW"        # Initiate vendor audit / evaluation
    CONTRACT_RENEGOTIATION = "CONTRACT_RENEGOTIATION"  # Fix pricing terms
    TEMPLATE_UPDATE = "TEMPLATE_UPDATE"    # Update contract template
    ESCALATE = "ESCALATE"                  # Escalate to legal / executive team
    DISMISS = "DISMISS"                    # False positive – no action needed


class ActionPlanStatus(str, enum.Enum):
    """Execution state of the action plan."""
    PENDING = "PENDING"           # Assigned but work not started
    IN_PROGRESS = "IN_PROGRESS"   # Owner is actively working on it
    COMPLETED = "COMPLETED"       # Resolved – dollars_saved recorded
    CANCELLED = "CANCELLED"       # Voided by auditor


class ActionPlan(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """
    Corrective action assigned to an audit case by a human auditor.

    ROI Aggregation
    ---------------
    GET /api/metrics/roi sums `dollars_saved` across all COMPLETED
    ActionPlans to compute the "Cost Savings YTD" dashboard KPI.
    This creates a closed feedback loop: AI flags → Human acts →
    Business value is quantified and displayed.
    """

    __tablename__ = "action_plans"

    # ── Foreign Keys ─────────────────────────────────────────────────────────
    case_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("audit_cases.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # ── Assignee ──────────────────────────────────────────────────────────────
    owner_email: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        index=True,
        comment="Email of the person responsible for executing this action",
    )
    owner_department: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
        comment="Department (e.g., Legal, Finance, Procurement Ops)",
    )

    # ── Action details ────────────────────────────────────────────────────────
    action_type: Mapped[ActionType] = mapped_column(
        SAEnum(ActionType, name="action_type_enum", create_type=True),
        nullable=False,
        index=True,
    )
    deadline: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
        comment="Target completion date set by the auditor",
    )
    notes: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        comment="Freeform instructions or context for the action owner",
    )

    # ── Financial impact (ROI feed) ───────────────────────────────────────────
    dollars_saved: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(precision=18, scale=2),
        nullable=True,
        comment=(
            "Estimated or confirmed savings in USD. "
            "Feeds the Executive ROI dashboard when status=COMPLETED."
        ),
    )
    estimated_recovery_usd: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(precision=18, scale=2),
        nullable=True,
        comment="Estimated recoverable amount before action is completed",
    )

    # ── Status ────────────────────────────────────────────────────────────────
    status: Mapped[ActionPlanStatus] = mapped_column(
        SAEnum(ActionPlanStatus, name="action_plan_status_enum", create_type=True),
        nullable=False,
        default=ActionPlanStatus.PENDING,
        server_default=ActionPlanStatus.PENDING.value,
        index=True,
    )
    completed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="Timestamp when the action was marked COMPLETED",
    )
    resolution_notes: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        comment="Outcome description written by the owner on completion",
    )

    # ── Relationships ─────────────────────────────────────────────────────────
    audit_case: Mapped["AuditCase"] = relationship(
        "AuditCase",
        back_populates="action_plans",
        lazy="select",
    )

    # ── Indexes ───────────────────────────────────────────────────────────────
    __table_args__ = (
        Index("ix_action_plans_status_type", "status", "action_type"),
        Index("ix_action_plans_owner_status", "owner_email", "status"),
    )

    def __repr__(self) -> str:
        return (
            f"<ActionPlan id={self.id} type={self.action_type.value} "
            f"status={self.status.value} owner={self.owner_email!r}>"
        )
