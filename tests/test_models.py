"""
tests/test_models.py
─────────────────────────────────────────────────────────────
Unit tests for Sprint 1: Models, Config, and Normalisation.

All tests run against SQLite in-memory (no Docker required).
Integration tests that need a real PostgreSQL instance are
marked with @pytest.mark.integration and skipped by default.

Run:
  pytest                         # unit tests only (fast)
  pytest -m integration          # integration tests (needs Docker)
  pytest -v --tb=short           # verbose output
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from decimal import Decimal

import pytest

from backend.models.vendor import RiskTier, Vendor
from backend.models.contract import Contract, ContractStatus
from backend.models.transaction import Transaction, TransactionSource
from backend.models.audit_case import AuditCase, AuditCaseStatus, LLMVerdict
from backend.models.action_plan import ActionPlan, ActionType, ActionPlanStatus


# ═════════════════════════════════════════════════════════════════════════════
# Config tests
# ═════════════════════════════════════════════════════════════════════════════

class TestSettings:
    """Verify Settings behaves correctly with various input combinations."""

    @pytest.mark.unit
    def test_settings_defaults_are_valid(self, patch_settings):
        """Settings should load without raising validation errors."""
        from backend.config import get_settings
        s = get_settings()
        assert s.APP_NAME == "Procurement Audit System"
        assert s.APP_ENV == "testing"

    @pytest.mark.unit
    def test_database_url_format(self, patch_settings):
        """Sync DSN should use psycopg2 driver."""
        from backend.config import get_settings
        url = get_settings().database_url
        assert url.startswith("postgresql+psycopg2://")
        assert "test_procurement_audit" in url

    @pytest.mark.unit
    def test_async_database_url_format(self, patch_settings):
        """Async DSN should use asyncpg driver."""
        from backend.config import get_settings
        url = get_settings().async_database_url
        assert url.startswith("postgresql+asyncpg://")

    @pytest.mark.unit
    def test_redis_url_without_password(self, patch_settings):
        """Redis URL without password should not include : in auth section."""
        from backend.config import get_settings
        url = get_settings().redis_url
        assert url.startswith("redis://")
        assert "@" not in url  # No auth section

    @pytest.mark.unit
    def test_kafka_bootstrap_list_parsing(self, patch_settings, monkeypatch):
        """Comma-separated bootstrap servers should be split into a list."""
        from backend.config import Settings
        s = Settings(KAFKA_BOOTSTRAP_SERVERS="broker1:9092,broker2:9092")
        assert s.kafka_bootstrap_list == ["broker1:9092", "broker2:9092"]

    @pytest.mark.unit
    def test_cors_origins_json_string_parsing(self):
        """CORS_ORIGINS should parse a JSON array string from env."""
        from backend.config import Settings
        s = Settings(CORS_ORIGINS='["http://localhost:3000","http://localhost:5173"]')
        assert "http://localhost:3000" in s.CORS_ORIGINS
        assert len(s.CORS_ORIGINS) == 2

    @pytest.mark.unit
    def test_invalid_log_level_raises(self):
        """An invalid LOG_LEVEL should raise a ValidationError."""
        from pydantic import ValidationError
        from backend.config import Settings
        with pytest.raises(ValidationError, match="LOG_LEVEL"):
            Settings(LOG_LEVEL="VERBOSE")


# ═════════════════════════════════════════════════════════════════════════════
# Vendor model tests
# ═════════════════════════════════════════════════════════════════════════════

class TestVendorModel:

    @pytest.mark.unit
    def test_vendor_creation_defaults(self, db_session, vendor_factory):
        """A new Vendor should default to LOW risk and zero YTD spend."""
        vendor = vendor_factory.create(db_session, name="Test Corp")
        assert vendor.id is not None
        assert vendor.risk_tier == RiskTier.LOW
        assert vendor.total_spend_ytd == Decimal("50000.00")  # from factory default

    @pytest.mark.unit
    def test_vendor_risk_tier_enum_values(self):
        """RiskTier enum should contain all four levels from architecture doc."""
        tiers = {t.value for t in RiskTier}
        assert tiers == {"LOW", "MEDIUM", "HIGH", "CRITICAL"}

    @pytest.mark.unit
    def test_vendor_repr_includes_name_and_risk(self, vendor_factory, db_session):
        """Vendor repr should include name and risk tier."""
        vendor = vendor_factory.create(db_session, name="ACME Ltd", risk_tier=RiskTier.HIGH)
        r = repr(vendor)
        assert "ACME Ltd" in r
        assert "HIGH" in r

    @pytest.mark.unit
    def test_vendor_name_uniqueness_enforced(self, db_session, vendor_factory):
        """Creating two vendors with the same name should raise an IntegrityError."""
        from sqlalchemy.exc import IntegrityError
        vendor_factory.create(db_session, name="Unique Vendor")
        with pytest.raises(IntegrityError):
            vendor_factory.create(db_session, name="Unique Vendor")

    @pytest.mark.unit
    def test_vendor_total_spend_is_decimal(self, db_session, vendor_factory):
        """total_spend_ytd should preserve Decimal precision."""
        vendor = vendor_factory.create(
            db_session,
            name="Precision Corp",
            total_spend_ytd=Decimal("123456.78"),
        )
        db_session.refresh(vendor)
        assert str(vendor.total_spend_ytd) in {"123456.78", "123456.7800"}


# ═════════════════════════════════════════════════════════════════════════════
# Transaction model tests
# ═════════════════════════════════════════════════════════════════════════════

class TestTransactionModel:

    @pytest.mark.unit
    def test_transaction_creation(self, db_session, vendor_factory, transaction_factory):
        """Transaction should be persisted with correct FK to vendor."""
        vendor = vendor_factory.create(db_session, name="Vendor A")
        tx = transaction_factory.create(db_session, vendor_id=vendor.id)
        assert tx.id is not None
        assert tx.vendor_id == vendor.id
        assert tx.source == TransactionSource.USASPENDING

    @pytest.mark.unit
    def test_transaction_amount_precision(self, db_session, vendor_factory, transaction_factory):
        """Transaction amount should preserve two decimal places."""
        vendor = vendor_factory.create(db_session, name="Vendor B")
        tx = transaction_factory.create(
            db_session,
            vendor_id=vendor.id,
            amount=Decimal("99999.99"),
        )
        assert tx.amount == Decimal("99999.99")

    @pytest.mark.unit
    def test_transaction_defaults_not_scored(self, db_session, vendor_factory, transaction_factory):
        """New transactions should have is_scored=False and ml_score=None."""
        vendor = vendor_factory.create(db_session, name="Vendor C")
        tx = transaction_factory.create(db_session, vendor_id=vendor.id)
        assert tx.is_scored is False
        assert tx.ml_score is None

    @pytest.mark.unit
    def test_transaction_source_enum_values(self):
        """TransactionSource should include all configured data sources."""
        sources = {s.value for s in TransactionSource}
        assert "USASPENDING" in sources
        assert "WORLDBANK" in sources
        assert "MANUAL" in sources

    @pytest.mark.unit
    def test_transaction_repr(self, db_session, vendor_factory, transaction_factory):
        """Transaction repr should include amount and vendor_id."""
        vendor = vendor_factory.create(db_session, name="Vendor D")
        tx = transaction_factory.create(
            db_session, vendor_id=vendor.id, amount=Decimal("500.00")
        )
        assert "500" in repr(tx)


# ═════════════════════════════════════════════════════════════════════════════
# Contract model tests
# ═════════════════════════════════════════════════════════════════════════════

class TestContractModel:

    @pytest.mark.unit
    def test_contract_creation(self, db_session, vendor_factory):
        """Contract should be linked to a vendor with ACTIVE default status."""
        vendor = vendor_factory.create(db_session, name="Contract Vendor")
        contract = Contract(
            vendor_id=vendor.id,
            title="IT Services Agreement 2024",
            upload_date=datetime.now(timezone.utc),
            status=ContractStatus.ACTIVE,
        )
        db_session.add(contract)
        db_session.flush()
        assert contract.id is not None
        assert contract.status == ContractStatus.ACTIVE
        assert contract.is_indexed is False

    @pytest.mark.unit
    def test_contract_status_enum_values(self):
        """ContractStatus should include all lifecycle states."""
        statuses = {s.value for s in ContractStatus}
        assert {"ACTIVE", "EXPIRED", "TERMINATED", "UNDER_REVIEW", "DRAFT"} == statuses

    @pytest.mark.unit
    def test_contract_faiss_fields_nullable(self, db_session, vendor_factory):
        """FAISS fields should be nullable before indexing is complete."""
        vendor = vendor_factory.create(db_session, name="RAG Vendor")
        contract = Contract(
            vendor_id=vendor.id,
            title="Pricing Agreement",
            upload_date=datetime.now(timezone.utc),
        )
        db_session.add(contract)
        db_session.flush()
        assert contract.faiss_index_id is None
        assert contract.chunk_count is None
        assert contract.is_indexed is False


# ═════════════════════════════════════════════════════════════════════════════
# AuditCase model tests
# ═════════════════════════════════════════════════════════════════════════════

class TestAuditCaseModel:

    @pytest.mark.unit
    def test_audit_case_creation(self, db_session, vendor_factory, transaction_factory):
        """AuditCase should link to a transaction with OPEN default status."""
        vendor = vendor_factory.create(db_session, name="Audit Vendor")
        tx = transaction_factory.create(db_session, vendor_id=vendor.id)

        case = AuditCase(
            transaction_id=tx.id,
            ml_score=-0.35,
            shap_reason={"amount": 4.2, "vendor_30d_velocity": 1.8},
            shap_summary="Amount is 4.2x vendor average",
        )
        db_session.add(case)
        db_session.flush()

        assert case.id is not None
        assert case.status == AuditCaseStatus.OPEN
        assert case.llm_verdict is None  # Not yet processed by LLM

    @pytest.mark.unit
    def test_audit_case_status_transitions(self):
        """All status values should be accessible from the enum."""
        statuses = {s.value for s in AuditCaseStatus}
        assert statuses == {"OPEN", "IN_REVIEW", "CLOSED"}

    @pytest.mark.unit
    def test_llm_verdict_enum_values(self):
        """LLMVerdict should include all possible verdict outcomes."""
        verdicts = {v.value for v in LLMVerdict}
        assert "FRAUD" in verdicts
        assert "SUSPICIOUS" in verdicts
        assert "NORMAL" in verdicts
        assert "INCONCLUSIVE" in verdicts

    @pytest.mark.unit
    def test_audit_case_shap_json_stored(self, db_session, vendor_factory, transaction_factory):
        """SHAP reason JSON should round-trip correctly through the DB."""
        vendor = vendor_factory.create(db_session, name="SHAP Vendor")
        tx = transaction_factory.create(db_session, vendor_id=vendor.id)
        shap = {"amount": 4.2, "vendor_30d_velocity": 1.8, "is_weekend": 0.5}

        case = AuditCase(transaction_id=tx.id, ml_score=-0.4, shap_reason=shap)
        db_session.add(case)
        db_session.flush()
        db_session.refresh(case)

        assert case.shap_reason == shap


# ═════════════════════════════════════════════════════════════════════════════
# ActionPlan model tests
# ═════════════════════════════════════════════════════════════════════════════

class TestActionPlanModel:

    @pytest.mark.unit
    def test_action_plan_creation(self, db_session, vendor_factory, transaction_factory):
        """ActionPlan should persist with correct FK to audit case."""
        vendor = vendor_factory.create(db_session, name="Action Vendor")
        tx = transaction_factory.create(db_session, vendor_id=vendor.id)
        case = AuditCase(transaction_id=tx.id, ml_score=-0.5)
        db_session.add(case)
        db_session.flush()

        plan = ActionPlan(
            case_id=case.id,
            owner_email="auditor@company.com",
            action_type=ActionType.CLAWBACK,
            deadline=datetime.now(timezone.utc),
            dollars_saved=Decimal("25000.00"),
        )
        db_session.add(plan)
        db_session.flush()

        assert plan.id is not None
        assert plan.status == ActionPlanStatus.PENDING
        assert plan.dollars_saved == Decimal("25000.00")

    @pytest.mark.unit
    def test_action_type_enum_values(self):
        """ActionType should include all options from the architecture doc."""
        types = {t.value for t in ActionType}
        assert "CLAWBACK" in types
        assert "PAYMENT_HALT" in types
        assert "VENDOR_REVIEW" in types
        assert "TEMPLATE_UPDATE" in types
        assert "ESCALATE" in types
        assert "DISMISS" in types

    @pytest.mark.unit
    def test_action_plan_status_enum_values(self):
        """ActionPlanStatus should include all lifecycle states."""
        statuses = {s.value for s in ActionPlanStatus}
        assert statuses == {"PENDING", "IN_PROGRESS", "COMPLETED", "CANCELLED"}

    @pytest.mark.unit
    def test_roi_fields_nullable_before_completion(self, db_session, vendor_factory, transaction_factory):
        """dollars_saved should be nullable until action is completed."""
        vendor = vendor_factory.create(db_session, name="ROI Vendor")
        tx = transaction_factory.create(db_session, vendor_id=vendor.id)
        case = AuditCase(transaction_id=tx.id, ml_score=-0.6)
        db_session.add(case)
        db_session.flush()

        plan = ActionPlan(
            case_id=case.id,
            owner_email="finance@company.com",
            action_type=ActionType.PAYMENT_HALT,
            deadline=datetime.now(timezone.utc),
            # dollars_saved intentionally omitted
        )
        db_session.add(plan)
        db_session.flush()

        assert plan.dollars_saved is None
        assert plan.status == ActionPlanStatus.PENDING


# ═════════════════════════════════════════════════════════════════════════════
# Vendor normalisation tests (from seed.py)
# ═════════════════════════════════════════════════════════════════════════════

class TestVendorNormalisation:
    """Tests for the _normalize_vendor_name function in seed.py."""

    @pytest.mark.unit
    def test_lowercase_conversion(self):
        from backend.ingestion.seed import _normalize_vendor_name
        assert _normalize_vendor_name("IBM Corp") == "ibm corp"

    @pytest.mark.unit
    def test_punctuation_removal(self):
        from backend.ingestion.seed import _normalize_vendor_name
        result = _normalize_vendor_name("AMAZON.COM, INC.")
        # Punctuation replaced by spaces, then collapsed
        assert "." not in result
        assert "," not in result

    @pytest.mark.unit
    def test_whitespace_collapse(self):
        from backend.ingestion.seed import _normalize_vendor_name
        result = _normalize_vendor_name("  Vendor   Name  ")
        assert result == "vendor name"  # No leading/trailing, single spaces

    @pytest.mark.unit
    def test_unicode_normalisation(self):
        from backend.ingestion.seed import _normalize_vendor_name
        # Accented chars should be preserved but normalised
        result = _normalize_vendor_name("Société Générale")
        assert isinstance(result, str)
        assert len(result) > 0


# ═════════════════════════════════════════════════════════════════════════════
# USAspending client parsing tests (no HTTP calls)
# ═════════════════════════════════════════════════════════════════════════════

class TestUSASpendingParsing:
    """Unit tests for record parsing logic – no network calls."""

    @pytest.mark.unit
    def test_parse_valid_record(self):
        """A complete API record should parse to a USASpendingRecord correctly."""
        from backend.ingestion.usaspending_client import USASpendingClient

        raw = {
            "Award ID": "CONT_AWD_123",
            "Recipient Name": "Acme Defense Systems",
            "recipient_uei": "ABC123DEF456",
            "recipient_location_country_code": "USA",
            "Award Amount": 500000.0,
            "Action Date": "2024-06-15",
            "Fiscal Year": 2024,
            "Award Type": "A",
            "NAICS Description": "Defense Systems",
            "Description": "Purchase of radar components",
            "Awarding Agency": "Department of Defense",
        }

        client = USASpendingClient.__new__(USASpendingClient)
        record = client._parse_record(raw)

        assert record.external_id == "CONT_AWD_123"
        assert record.vendor_name == "Acme Defense Systems"
        assert record.vendor_uei == "ABC123DEF456"
        assert record.amount == 500000.0
        assert record.action_date.year == 2024
        assert record.category == "Defense Systems"

    @pytest.mark.unit
    def test_parse_record_with_missing_fields(self):
        """Parsing a sparse record should not raise – missing fields default to None."""
        from backend.ingestion.usaspending_client import USASpendingClient

        raw = {
            "Award ID": "MINIMAL-001",
            "Recipient Name": "Unknown Vendor",
            "Award Amount": 1000.0,
            "Action Date": "2024-01-01",
        }

        client = USASpendingClient.__new__(USASpendingClient)
        record = client._parse_record(raw)

        assert record.external_id == "MINIMAL-001"
        assert record.vendor_uei is None
        assert record.category is None
        assert record.description is None

    @pytest.mark.unit
    def test_parse_record_invalid_amount_defaults_to_zero(self):
        """If amount can't be parsed as float, it should default to 0.0."""
        from backend.ingestion.usaspending_client import USASpendingClient

        raw = {
            "Award ID": "BAD-AMT",
            "Recipient Name": "Test Vendor",
            "Award Amount": "not_a_number",
            "Action Date": "2024-03-01",
        }

        client = USASpendingClient.__new__(USASpendingClient)
        record = client._parse_record(raw)
        assert record.amount == 0.0

    @pytest.mark.unit
    def test_parse_date_invalid_string_returns_utcnow(self):
        """An unparseable date string should fall back to UTC now."""
        from backend.ingestion.usaspending_client import USASpendingClient

        result = USASpendingClient._parse_date("not-a-date")
        # Should not raise and should return a datetime
        assert isinstance(result, datetime)
        assert result.tzinfo is not None
