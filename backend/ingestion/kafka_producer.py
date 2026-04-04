"""
backend/ingestion/kafka_producer.py
─────────────────────────────────────────────────────────────
Async Kafka / Redpanda producer.

Wraps aiokafka's AIOKafkaProducer with:
  • Topic management (auto-create topics on startup)
  • JSON serialisation with UTC datetime handling
  • Structured logging of every published message
  • Graceful shutdown (flush + stop)
  • Singleton pattern via module-level instance

Architecture role
─────────────────
After the seed script writes a transaction to PostgreSQL, it
also publishes it to the `raw_transactions` Kafka topic as JSON.
This decouples the ingestion layer from the ML scoring layer:
the Flink stream processor reads from `raw_transactions` without
coupling to PostgreSQL.

Topic routing (from architecture doc):
  raw_transactions         ← Ingestion layer publishes here (this file)
  enriched_transactions    ← Flink publishes here (Sprint 2)
  anomalies_topic          ← ML microservice publishes here (Sprint 2)
  triage_topic             ← Groq triage publishes here (Sprint 4)
  deep_audit_topic         ← Gemini audit publishes here (Sprint 4)
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, Optional

from aiokafka import AIOKafkaProducer
from aiokafka.admin import AIOKafkaAdminClient, NewTopic
from loguru import logger

from backend.config import get_settings

settings = get_settings()


# ─────────────────────────────────────────────────────────────────────────────
# Custom JSON Encoder
# ─────────────────────────────────────────────────────────────────────────────

class _ProcurementJSONEncoder(json.JSONEncoder):
    """Handles types not natively serialisable by the standard JSON encoder."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, Decimal):
            return float(obj)
        if isinstance(obj, uuid.UUID):
            return str(obj)
        return super().default(obj)


def _serialize(data: Dict[str, Any]) -> bytes:
    """Serialise a dict to UTF-8 JSON bytes."""
    return json.dumps(data, cls=_ProcurementJSONEncoder).encode("utf-8")


def _key(value: str) -> bytes:
    """Encode a partition key as UTF-8 bytes."""
    return value.encode("utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# Kafka Producer
# ─────────────────────────────────────────────────────────────────────────────

class ProcurementKafkaProducer:
    """
    Async Kafka producer for the procurement audit pipeline.

    Lifecycle
    ---------
    Must be used as an async context manager or explicitly started/stopped:

        async with ProcurementKafkaProducer() as producer:
            await producer.publish_raw_transaction(record_dict)

    Or:
        producer = ProcurementKafkaProducer()
        await producer.start()
        ...
        await producer.stop()
    """

    # Topics to auto-create on startup with sensible defaults
    TOPICS_CONFIG: Dict[str, Dict[str, Any]] = {
        settings.KAFKA_TOPIC_RAW_TRANSACTIONS: {
            "num_partitions": 3,
            "replication_factor": 1,
            "retention_hours": 168,  # 7 days
        },
        settings.KAFKA_TOPIC_ENRICHED_TRANSACTIONS: {
            "num_partitions": 3,
            "replication_factor": 1,
            "retention_hours": 168,
        },
        settings.KAFKA_TOPIC_ANOMALIES: {
            "num_partitions": 1,
            "replication_factor": 1,
            "retention_hours": 720,  # 30 days
        },
        settings.KAFKA_TOPIC_TRIAGE: {
            "num_partitions": 1,
            "replication_factor": 1,
            "retention_hours": 720,
        },
        settings.KAFKA_TOPIC_DEEP_AUDIT: {
            "num_partitions": 1,
            "replication_factor": 1,
            "retention_hours": 720,
        },
    }

    def __init__(self) -> None:
        self._producer: Optional[AIOKafkaProducer] = None
        self._started = False

    async def __aenter__(self) -> "ProcurementKafkaProducer":
        await self.start()
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.stop()

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Start the producer and ensure all required topics exist."""
        if self._started:
            return

        await self._ensure_topics_exist()

        self._producer = AIOKafkaProducer(
            bootstrap_servers=settings.kafka_bootstrap_list,
            value_serializer=_serialize,
            key_serializer=_key,
            # Durability: wait for leader + all in-sync replicas
            acks="all",
            # Batching for throughput
            linger_ms=10,
            # Compression
            compression_type="gzip",
            # Retry backoff for transient retries handled by client internals
            retry_backoff_ms=500,
        )
        try:
            await self._producer.start()
            self._started = True
            logger.info(
                "Kafka producer started | brokers={}",
                settings.KAFKA_BOOTSTRAP_SERVERS,
            )
        except Exception:
            try:
                await self._producer.stop()
            except Exception:
                pass
            self._producer = None
            self._started = False
            raise

    async def stop(self) -> None:
        """Flush pending messages and close the producer."""
        if self._producer and self._started:
            await self._producer.flush()
            await self._producer.stop()
            self._started = False
            logger.info("Kafka producer stopped.")

    # ── Publishing methods ─────────────────────────────────────────────────────

    async def publish_raw_transaction(
        self,
        transaction_id: str,
        payload: Dict[str, Any],
    ) -> None:
        """
        Publish a raw transaction to the `raw_transactions` topic.

        Called by the seed script immediately after DB insert.
        The transaction_id is used as the partition key so all events
        for the same transaction always land on the same partition.

        Parameters
        ----------
        transaction_id – UUID string of the Transaction row
        payload        – dict representation of the transaction
        """
        await self._send(
            topic=settings.KAFKA_TOPIC_RAW_TRANSACTIONS,
            key=transaction_id,
            value={"event_type": "TRANSACTION_CREATED", "transaction_id": transaction_id, **payload},
        )

    async def publish_anomaly(
        self,
        case_id: str,
        payload: Dict[str, Any],
    ) -> None:
        """Publish a scored anomaly to `anomalies_topic` (Sprint 2)."""
        await self._send(
            topic=settings.KAFKA_TOPIC_ANOMALIES,
            key=case_id,
            value={"event_type": "ANOMALY_DETECTED", "case_id": case_id, **payload},
        )

    async def publish_triage_result(
        self,
        case_id: str,
        payload: Dict[str, Any],
    ) -> None:
        """Publish a Groq triage result to `triage_topic` (Sprint 4)."""
        await self._send(
            topic=settings.KAFKA_TOPIC_TRIAGE,
            key=case_id,
            value={"event_type": "TRIAGE_COMPLETE", "case_id": case_id, **payload},
        )

    async def publish_deep_audit(
        self,
        case_id: str,
        payload: Dict[str, Any],
    ) -> None:
        """Publish a Gemini deep audit trigger to `deep_audit_topic` (Sprint 4)."""
        await self._send(
            topic=settings.KAFKA_TOPIC_DEEP_AUDIT,
            key=case_id,
            value={"event_type": "DEEP_AUDIT_REQUESTED", "case_id": case_id, **payload},
        )

    # ── Internal helpers ───────────────────────────────────────────────────────

    async def _send(
        self,
        topic: str,
        key: str,
        value: Dict[str, Any],
    ) -> None:
        """Internal send with error handling and structured logging."""
        if not self._producer or not self._started:
            raise RuntimeError("Producer not started. Call `await producer.start()` first.")

        try:
            record_metadata = await self._producer.send_and_wait(
                topic=topic,
                key=key,
                value=value,
            )
            logger.debug(
                "Published to Kafka | topic={} partition={} offset={} key={}",
                topic,
                record_metadata.partition,
                record_metadata.offset,
                key,
            )
        except Exception as exc:
            logger.error(
                "Failed to publish to Kafka | topic={} key={} error={}",
                topic, key, exc,
            )
            raise

    async def _ensure_topics_exist(self) -> None:
        """
        Auto-create required Kafka topics if they don't already exist.

        This is idempotent – safe to run on every startup.
        Redpanda allows `auto.create.topics.enable`, but explicit
        creation here lets us control partition count and retention.
        """
        admin = AIOKafkaAdminClient(
            bootstrap_servers=settings.kafka_bootstrap_list,
        )
        try:
            await admin.start()
            existing = set(await admin.list_topics())
            topics_to_create = [
                NewTopic(
                    name=name,
                    num_partitions=cfg["num_partitions"],
                    replication_factor=cfg["replication_factor"],
                    topic_configs={
                        "retention.ms": str(cfg["retention_hours"] * 3_600_000),
                    },
                )
                for name, cfg in self.TOPICS_CONFIG.items()
                if name not in existing
            ]
            if topics_to_create:
                await admin.create_topics(topics_to_create)
                logger.info(
                    "Created Kafka topics: {}",
                    [t.name for t in topics_to_create],
                )
            else:
                logger.debug("All required Kafka topics already exist.")
        except Exception as exc:
            # Non-fatal: topics may exist or Redpanda may auto-create them
            logger.warning("Could not auto-create Kafka topics: {}", exc)
        finally:
            await admin.close()
