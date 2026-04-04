"""Kafka/Redpanda consumer that persists raw ingestion events and triggers orchestration."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, Optional

from aiokafka import AIOKafkaConsumer
from loguru import logger
from sqlalchemy import select

from backend.config import get_settings
from backend.database import AsyncSessionLocal
from backend.models.transaction import Transaction, TransactionSource
from backend.models.vendor import Vendor
from backend.ml.entity_resolution import EntityResolver
from backend.orchestration import get_orchestration_engine


class ProcurementKafkaConsumer:
    def __init__(self) -> None:
        self._settings = get_settings()
        self._consumer: Optional[AIOKafkaConsumer] = None
        self._resolver = EntityResolver()

    async def start(self) -> None:
        if self._consumer is not None:
            return

        self._consumer = AIOKafkaConsumer(
            self._settings.KAFKA_TOPIC_RAW_TRANSACTIONS,
            bootstrap_servers=self._settings.kafka_bootstrap_list,
            group_id=getattr(self._settings, "KAFKA_CONSUMER_GROUP", "procurement_ingestion_workers"),
            enable_auto_commit=True,
            auto_offset_reset=getattr(self._settings, "KAFKA_CONSUMER_AUTO_OFFSET_RESET", "latest"),
            value_deserializer=lambda payload: json.loads(payload.decode("utf-8")),
        )
        await self._consumer.start()
        logger.success(
            "Kafka consumer started | topic={} group={} brokers={}",
            self._settings.KAFKA_TOPIC_RAW_TRANSACTIONS,
            getattr(self._settings, "KAFKA_CONSUMER_GROUP", "procurement_ingestion_workers"),
            self._settings.KAFKA_BOOTSTRAP_SERVERS,
        )

    async def stop(self) -> None:
        if self._consumer is not None:
            await self._consumer.stop()
            self._consumer = None
            logger.info("Kafka consumer stopped")

    async def run_forever(self) -> None:
        if self._consumer is None:
            await self.start()

        assert self._consumer is not None
        async for message in self._consumer:
            payload = message.value
            try:
                await self.process_payload(payload)
            except Exception as exc:
                logger.exception("Failed processing raw transaction event: {}", exc)

    async def process_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Persist one raw event into DB and trigger scoring/case orchestration.
        Returns a processing trace dict for verification tooling.
        """
        trace = {
            "external_id": str(payload.get("transaction_id") or "").strip(),
            "vendor_created": False,
            "transaction_created": False,
            "transaction_updated": False,
            "orchestration_scored": 0,
            "orchestration_cases_created": 0,
        }

        external_id = trace["external_id"]
        if not external_id:
            raise ValueError("Missing required field: transaction_id")

        vendor_name_raw = str(payload.get("vendor") or "UNKNOWN_VENDOR").strip() or "UNKNOWN_VENDOR"
        canonical_vendor_name = self._resolver.normalize_name(vendor_name_raw)
        canonical_vendor_id = self._resolver.canonical_vendor_id(vendor_name_raw)

        buyer = str(payload.get("buyer") or "UNKNOWN_BUYER").strip() or "UNKNOWN_BUYER"
        currency = str(payload.get("currency") or "USD").upper()
        amount = _safe_decimal(payload.get("amount"))
        tx_date = _safe_datetime(payload.get("timestamp"))
        source = _map_source(payload.get("source"))
        category = _truncate(str(payload.get("category") or "").strip(), 255) or None
        description = str(payload.get("description") or "").strip() or None

        async with AsyncSessionLocal() as db:
            vendor = await db.scalar(
                select(Vendor).where(Vendor.normalized_name == canonical_vendor_name)
            )
            if vendor is None:
                vendor = Vendor(
                    name=_truncate(vendor_name_raw, 255) or "UNKNOWN_VENDOR",
                    normalized_name=canonical_vendor_name,
                    country=_truncate(str(payload.get("country") or "").strip(), 100) or None,
                )
                db.add(vendor)
                await db.flush()
                trace["vendor_created"] = True

            tx = await db.scalar(
                select(Transaction).where(Transaction.external_id == external_id)
            )

            raw_data = {
                **(payload or {}),
                "canonical_vendor_name": canonical_vendor_name,
                "canonical_vendor_id": canonical_vendor_id,
                "ingestion_trace_stage": "RAW_TOPIC_CONSUMED",
            }

            if tx is None:
                tx = Transaction(
                    vendor_id=vendor.id,
                    external_id=external_id,
                    source=source,
                    amount=amount,
                    currency=currency,
                    date=tx_date,
                    category=category,
                    description=description,
                    awarding_agency=buyer,
                    is_enriched=False,
                    is_scored=False,
                    raw_data=raw_data,
                )
                db.add(tx)
                trace["transaction_created"] = True
            else:
                tx.vendor_id = vendor.id
                tx.source = source
                tx.amount = amount
                tx.currency = currency
                tx.date = tx_date
                tx.category = category
                tx.description = description
                tx.awarding_agency = buyer
                tx.raw_data = raw_data
                tx.is_scored = False
                trace["transaction_updated"] = True

            await db.commit()

        if getattr(self._settings, "KAFKA_CONSUMER_TRIGGER_ORCHESTRATION", True):
            engine = get_orchestration_engine()
            summary = await engine.run_once(batch_size=25, run_llm=self._settings.ORCH_RUN_LLM)
            trace["orchestration_scored"] = summary.scored
            trace["orchestration_cases_created"] = summary.cases_created

        return trace



def _safe_decimal(value: Any) -> Decimal:
    try:
        return Decimal(str(value))
    except Exception:
        return Decimal("0")



def _safe_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, str) and value.strip():
        candidate = value.replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(candidate)
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
        except Exception:
            pass
    return datetime.now(timezone.utc)



def _map_source(raw_source: Any) -> TransactionSource:
    normalized = str(raw_source or "").strip().upper()
    if normalized in {"USA", "USASPENDING", "US"}:
        return TransactionSource.USASPENDING
    if normalized in {"CPPP", "INDIA_CPPP", "EPROCURE_GOV_IN"}:
        return TransactionSource.CPPP
    if normalized in {"OCDS", "OPEN_CONTRACTING"}:
        return TransactionSource.OCDS
    return TransactionSource.MANUAL



def _truncate(value: str, max_len: int) -> str:
    return value[:max_len] if value else value
