"""Initial schema – all 5 procurement audit tables.

Revision ID: 0001_initial_schema
Revises:
Create Date: 2026-03-09 00:00:00.000000
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic
revision: str = "0001_initial_schema"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:

    # ── ENUM types (DO block catches duplicate_object on re-runs) ─────────────
    _enums = [
        ("risk_tier_enum",          "LOW, MEDIUM, HIGH, CRITICAL"),
        ("contract_status_enum",    "ACTIVE, EXPIRED, TERMINATED, UNDER_REVIEW, DRAFT"),
        ("transaction_source_enum", "USASPENDING, WORLDBANK, SEC_EDGAR, NYC_OPENDATA, MANUAL"),
        ("audit_case_status_enum",  "OPEN, IN_REVIEW, CLOSED"),
        ("llm_verdict_enum",        "FRAUD, SUSPICIOUS, NORMAL, INCONCLUSIVE"),
        ("action_type_enum",        "CLAWBACK, PAYMENT_HALT, VENDOR_REVIEW, CONTRACT_RENEGOTIATION, TEMPLATE_UPDATE, ESCALATE, DISMISS"),
        ("action_plan_status_enum", "PENDING, IN_PROGRESS, COMPLETED, CANCELLED"),
    ]
    for name, values in _enums:
        quoted = ", ".join(f"'{v.strip()}'" for v in values.split(","))
        op.execute(
            f"DO $$ BEGIN CREATE TYPE {name} AS ENUM ({quoted}); "
            f"EXCEPTION WHEN duplicate_object THEN NULL; END $$;"
        )

    # ── vendors ────────────────────────────────────────────────────────────────
    op.create_table(
        "vendors",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("normalized_name", sa.String(255), nullable=True),
        sa.Column(
            "risk_tier",
            postgresql.ENUM("LOW", "MEDIUM", "HIGH", "CRITICAL", name="risk_tier_enum", create_type=False),
            nullable=False,
            server_default="LOW",
        ),
        sa.Column(
            "total_spend_ytd",
            sa.Numeric(precision=18, scale=2),
            nullable=False,
            server_default="0.00",
        ),
        sa.Column("country", sa.String(100), nullable=True),
        sa.Column("duns_number", sa.String(50), nullable=True),
        sa.Column("notes", sa.Text, nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("name"),
        sa.UniqueConstraint("duns_number"),
    )
    op.create_index("ix_vendors_id",              "vendors", ["id"])
    op.create_index("ix_vendors_name",            "vendors", ["name"])
    op.create_index("ix_vendors_normalized_name", "vendors", ["normalized_name"])
    op.create_index("ix_vendors_risk_tier",       "vendors", ["risk_tier"])
    op.create_index("ix_vendors_risk_tier_name",  "vendors", ["risk_tier", "name"])

    # ── contracts ──────────────────────────────────────────────────────────────
    op.create_table(
        "contracts",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("vendor_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("title", sa.String(500), nullable=False),
        sa.Column("contract_number", sa.String(100), nullable=True),
        sa.Column("file_path", sa.String(1024), nullable=True),
        sa.Column("file_hash", sa.String(64), nullable=True),
        sa.Column("upload_date", sa.DateTime(timezone=True), nullable=False),
        sa.Column("effective_date", sa.DateTime(timezone=True), nullable=True),
        sa.Column("expiry_date", sa.DateTime(timezone=True), nullable=True),
        sa.Column("total_value", sa.Float, nullable=True),
        sa.Column(
            "status",
            postgresql.ENUM(
                "ACTIVE", "EXPIRED", "TERMINATED", "UNDER_REVIEW", "DRAFT",
                name="contract_status_enum",
                create_type=False,
            ),
            nullable=False,
            server_default="ACTIVE",
        ),
        sa.Column("description", sa.Text, nullable=True),
        sa.Column("faiss_index_id", sa.String(255), nullable=True),
        sa.Column("chunk_count", sa.Integer, nullable=True),
        sa.Column("embedding_model", sa.String(255), nullable=True),
        sa.Column("is_indexed", sa.Boolean, nullable=False, server_default="false"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.ForeignKeyConstraint(["vendor_id"], ["vendors.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("contract_number"),
        sa.UniqueConstraint("faiss_index_id"),
    )
    op.create_index("ix_contracts_id",          "contracts", ["id"])
    op.create_index("ix_contracts_vendor_id",   "contracts", ["vendor_id"])
    op.create_index("ix_contracts_status",      "contracts", ["status"])
    op.create_index("ix_contracts_expiry_date", "contracts", ["expiry_date"])

    # ── transactions ───────────────────────────────────────────────────────────
    op.create_table(
        "transactions",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("vendor_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("external_id", sa.String(255), nullable=True),
        sa.Column(
            "source",
            postgresql.ENUM(
                "USASPENDING", "WORLDBANK", "SEC_EDGAR", "NYC_OPENDATA", "MANUAL",
                name="transaction_source_enum",
                create_type=False,
            ),
            nullable=False,
            server_default="USASPENDING",
        ),
        sa.Column("amount", sa.Numeric(precision=18, scale=2), nullable=False),
        sa.Column("currency", sa.String(3), nullable=False, server_default="USD"),
        sa.Column("date", sa.DateTime(timezone=True), nullable=False),
        sa.Column("fiscal_year", sa.Integer, nullable=True),
        sa.Column("category", sa.String(255), nullable=True),
        sa.Column("award_type", sa.String(100), nullable=True),
        sa.Column("description", sa.Text, nullable=True),
        sa.Column("awarding_agency", sa.String(500), nullable=True),
        sa.Column("is_enriched", sa.Boolean, nullable=False, server_default="false"),
        sa.Column("is_scored", sa.Boolean, nullable=False, server_default="false"),
        sa.Column("ml_score", sa.Float, nullable=True),
        sa.Column("raw_data", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.ForeignKeyConstraint(["vendor_id"], ["vendors.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("external_id"),
    )
    op.create_index("ix_transactions_id",             "transactions", ["id"])
    op.create_index("ix_transactions_vendor_id",      "transactions", ["vendor_id"])
    op.create_index("ix_transactions_external_id",    "transactions", ["external_id"])
    op.create_index("ix_transactions_date",           "transactions", ["date"])
    op.create_index("ix_transactions_source",         "transactions", ["source"])
    op.create_index("ix_transactions_is_scored",      "transactions", ["is_scored"])
    op.create_index("ix_transactions_category",       "transactions", ["category"])
    op.create_index("ix_transactions_vendor_date",    "transactions", ["vendor_id", "date"])
    op.create_index("ix_transactions_scored_score",   "transactions", ["is_scored", "ml_score"])
    op.create_index("ix_transactions_category_date",  "transactions", ["category", "date"])

    # ── audit_cases ────────────────────────────────────────────────────────────
    op.create_table(
        "audit_cases",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("transaction_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("ml_score", sa.Float, nullable=False),
        sa.Column("shap_reason", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("shap_summary", sa.Text, nullable=True),
        sa.Column("groq_verdict", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("groq_escalated", sa.Boolean, nullable=True),
        sa.Column(
            "llm_verdict",
            postgresql.ENUM("FRAUD", "SUSPICIOUS", "NORMAL", "INCONCLUSIVE", name="llm_verdict_enum", create_type=False),
            nullable=True,
        ),
        sa.Column("gemini_report", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("confidence", sa.Float, nullable=True),
        sa.Column("contract_clause_cited", sa.Text, nullable=True),
        sa.Column("violated_clause_id", sa.String(100), nullable=True),
        sa.Column(
            "status",
            postgresql.ENUM("OPEN", "IN_REVIEW", "CLOSED", name="audit_case_status_enum", create_type=False),
            nullable=False,
            server_default="OPEN",
        ),
        sa.Column("risk_level", sa.String(10), nullable=True),
        sa.Column("estimated_impact_usd", sa.Float, nullable=True),
        sa.Column("auditor_notes", sa.Text, nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.ForeignKeyConstraint(
            ["transaction_id"], ["transactions.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("transaction_id"),
    )
    op.create_index("ix_audit_cases_id",             "audit_cases", ["id"])
    op.create_index("ix_audit_cases_transaction_id", "audit_cases", ["transaction_id"])
    op.create_index("ix_audit_cases_llm_verdict",    "audit_cases", ["llm_verdict"])
    op.create_index("ix_audit_cases_status",         "audit_cases", ["status"])
    op.create_index("ix_audit_cases_status_verdict", "audit_cases", ["status", "llm_verdict"])
    op.create_index("ix_audit_cases_status_score",   "audit_cases", ["status", "ml_score"])

    # ── action_plans ───────────────────────────────────────────────────────────
    op.create_table(
        "action_plans",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("case_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("owner_email", sa.String(255), nullable=False),
        sa.Column("owner_department", sa.String(100), nullable=True),
        sa.Column(
            "action_type",
            postgresql.ENUM(
                "CLAWBACK", "PAYMENT_HALT", "VENDOR_REVIEW",
                "CONTRACT_RENEGOTIATION", "TEMPLATE_UPDATE", "ESCALATE", "DISMISS",
                name="action_type_enum",
                create_type=False,
            ),
            nullable=False,
        ),
        sa.Column("deadline", sa.DateTime(timezone=True), nullable=False),
        sa.Column("notes", sa.Text, nullable=True),
        sa.Column("dollars_saved", sa.Numeric(precision=18, scale=2), nullable=True),
        sa.Column(
            "estimated_recovery_usd",
            sa.Numeric(precision=18, scale=2),
            nullable=True,
        ),
        sa.Column(
            "status",
            postgresql.ENUM(
                "PENDING", "IN_PROGRESS", "COMPLETED", "CANCELLED",
                name="action_plan_status_enum",
                create_type=False,
            ),
            nullable=False,
            server_default="PENDING",
        ),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("resolution_notes", sa.Text, nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.ForeignKeyConstraint(["case_id"], ["audit_cases.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_action_plans_id",           "action_plans", ["id"])
    op.create_index("ix_action_plans_case_id",      "action_plans", ["case_id"])
    op.create_index("ix_action_plans_owner_email",  "action_plans", ["owner_email"])
    op.create_index("ix_action_plans_deadline",     "action_plans", ["deadline"])
    op.create_index("ix_action_plans_status",       "action_plans", ["status"])
    op.create_index("ix_action_plans_action_type",  "action_plans", ["action_type"])
    op.create_index("ix_action_plans_status_type",  "action_plans", ["status", "action_type"])
    op.create_index("ix_action_plans_owner_status", "action_plans", ["owner_email", "status"])


def downgrade() -> None:
    op.drop_table("action_plans")
    op.drop_table("audit_cases")
    op.drop_table("transactions")
    op.drop_table("contracts")
    op.drop_table("vendors")

    # Drop ENUM types (PostgreSQL keeps them even after table drop)
    for enum_name in [
        "action_plan_status_enum",
        "action_type_enum",
        "llm_verdict_enum",
        "audit_case_status_enum",
        "transaction_source_enum",
        "contract_status_enum",
        "risk_tier_enum",
    ]:
        op.execute(f"DROP TYPE IF EXISTS {enum_name}")
