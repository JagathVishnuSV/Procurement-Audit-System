"""
backend/ingestion/__init__.py
─────────────────────────────────────────────────────────────
Data ingestion sub-package.

Contains:
  base_client         – abstract async HTTP client with retries
  usaspending_client  – USAspending.gov API adapter
  multi_source        – multi-source adapter framework + canonical schema
  data_quality        – schema cleanup, currency normalisation, deduplication
  realtime_pipeline   – near-realtime poll/detect/publish backbone
  kafka_producer      – Redpanda/Kafka producer for event topics
  seed                – CLI script: pull data and seed PostgreSQL
"""

from backend.ingestion.multi_source import (
    BaseIngestionAdapter,
    CanonicalProcurementRecord,
    CPPPAdapter,
    OCDSAdapter,
    USASpendingAdapter,
)
from backend.ingestion.realtime_pipeline import NearRealtimeIngestionPipeline
from backend.ingestion.kafka_consumer import ProcurementKafkaConsumer

__all__ = [
    "BaseIngestionAdapter",
    "CanonicalProcurementRecord",
    "USASpendingAdapter",
    "CPPPAdapter",
    "OCDSAdapter",
    "NearRealtimeIngestionPipeline",
    "ProcurementKafkaConsumer",
]
