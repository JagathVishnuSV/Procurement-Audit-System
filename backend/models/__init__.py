"""
backend/models/__init__.py
─────────────────────────────────────────────────────────────
Public model imports.

Importing this module guarantees all ORM models are registered
on `Base.metadata` before Alembic autogenerate runs.
"""

from backend.models.vendor import Vendor, RiskTier
from backend.models.contract import Contract, ContractStatus
from backend.models.transaction import Transaction, TransactionSource
from backend.models.audit_case import AuditCase, AuditCaseStatus, LLMVerdict
from backend.models.action_plan import ActionPlan, ActionType, ActionPlanStatus

__all__ = [
    # Vendor
    "Vendor",
    "RiskTier",
    # Contract
    "Contract",
    "ContractStatus",
    # Transaction
    "Transaction",
    "TransactionSource",
    # Audit Case
    "AuditCase",
    "AuditCaseStatus",
    "LLMVerdict",
    # Action Plan
    "ActionPlan",
    "ActionType",
    "ActionPlanStatus",
]
