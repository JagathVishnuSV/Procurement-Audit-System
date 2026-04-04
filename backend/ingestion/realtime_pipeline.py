"""Near real-time multi-source ingestion pipeline backed by Redpanda/Kafka."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Dict, Iterable, List

from loguru import logger

from backend.ingestion.data_quality import run_data_quality_pipeline
from backend.ingestion.kafka_producer import ProcurementKafkaProducer
from backend.ingestion.multi_source import (
    BaseIngestionAdapter,
    CPPPAdapter,
    CanonicalProcurementRecord,
    OCDSAdapter,
    USASpendingAdapter,
)

STATE_FILE = Path("data/ingestion_change_state.json")


@dataclass(slots=True)
class IngestionPipelineStats:
    polled: int = 0
    normalized: int = 0
    changed: int = 0
    published: int = 0
    sources: Dict[str, int] = field(default_factory=dict)


class ChangeDetector:
    """Simple hash-based new/updated detector per source transaction id."""

    def __init__(self, state_path: Path = STATE_FILE) -> None:
        self.state_path = state_path
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self._state = self._load_state()

    def _load_state(self) -> Dict[str, str]:
        if self.state_path.exists():
            try:
                return json.loads(self.state_path.read_text(encoding="utf-8"))
            except Exception:
                return {}
        return {}

    def _save_state(self) -> None:
        self.state_path.write_text(json.dumps(self._state, indent=2), encoding="utf-8")

    def diff(self, records: Iterable[CanonicalProcurementRecord]) -> List[CanonicalProcurementRecord]:
        changed: List[CanonicalProcurementRecord] = []
        for record in records:
            key = f"{record.source}:{record.transaction_id}"
            digest = sha256(json.dumps(record.to_dict(), sort_keys=True).encode("utf-8")).hexdigest()
            if self._state.get(key) != digest:
                changed.append(record)
                self._state[key] = digest
        self._save_state()
        return changed


class NearRealtimeIngestionPipeline:
    """
    Pollers (5–10 min) -> change detection -> Redpanda backbone -> downstream consumers.
    """

    def __init__(
        self,
        adapters: List[BaseIngestionAdapter] | None = None,
        poll_interval_seconds: int = 300,
    ) -> None:
        self.adapters = adapters or [USASpendingAdapter(), CPPPAdapter(), OCDSAdapter()]
        self.poll_interval_seconds = poll_interval_seconds
        self.change_detector = ChangeDetector()

    async def run_once(self, limit_per_source: int = 100) -> IngestionPipelineStats:
        return await self.run_once_with_payloads(limit_per_source=limit_per_source, source_payloads=None)

    async def run_once_with_payloads(
        self,
        limit_per_source: int = 100,
        source_payloads: Dict[str, List[Dict]] | None = None,
    ) -> IngestionPipelineStats:
        stats = IngestionPipelineStats()

        async with ProcurementKafkaProducer() as producer:
            all_raw: List[CanonicalProcurementRecord] = []
            for adapter in self.adapters:
                payload = None
                if source_payloads is not None:
                    payload = source_payloads.get(adapter.source_name)

                raw_records = await adapter.fetch(limit=limit_per_source, payload=payload)
                stats.polled += len(raw_records)
                stats.sources[adapter.source_name] = len(raw_records)
                normalized = adapter.normalize(raw_records)
                all_raw.extend(normalized)

            cleaned = run_data_quality_pipeline(all_raw)
            stats.normalized = len(cleaned)

            changed = self.change_detector.diff(cleaned)
            stats.changed = len(changed)

            for record in changed:
                await producer.publish_raw_transaction(
                    transaction_id=record.transaction_id,
                    payload={
                        "transaction_id": record.transaction_id,
                        "buyer": record.buyer,
                        "vendor": record.vendor,
                        "amount": float(record.amount),
                        "currency": record.currency,
                        "timestamp": record.timestamp.isoformat(),
                        "source": record.source,
                        "description": record.description,
                        "category": record.category,
                        "country": record.country,
                        "ingested_at": datetime.now(timezone.utc).isoformat(),
                    },
                )
                stats.published += 1

        logger.info(
            "Near-realtime ingestion run complete | polled={} normalized={} changed={} published={} sources={}",
            stats.polled,
            stats.normalized,
            stats.changed,
            stats.published,
            stats.sources,
        )
        return stats

    async def run_forever(self, limit_per_source: int = 100) -> None:
        logger.info(
            "Starting near-realtime ingestion loop | poll_interval_seconds={} adapters={}",
            self.poll_interval_seconds,
            [adapter.source_name for adapter in self.adapters],
        )
        cycle = 0
        while True:
            cycle += 1
            try:
                await self.run_once(limit_per_source=limit_per_source)
            except Exception as exc:
                logger.exception("Ingestion cycle {} failed: {}", cycle, exc)
            await asyncio.sleep(self.poll_interval_seconds)
