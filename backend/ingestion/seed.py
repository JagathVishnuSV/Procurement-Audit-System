"""
backend/ingestion/seed.py
─────────────────────────────────────────────────────────────
Sprint 1 milestone script: Pull real procurement data from
USAspending.gov and seed the PostgreSQL database.

What it does
────────────
1. Connects to PostgreSQL via SQLAlchemy
2. Fetches real federal contract awards from USAspending.gov API
3. Upserts Vendor records (deduplicates by name / UEI)
4. Inserts Transaction records (deduplicates by external_id)
5. Publishes each new transaction to the `raw_transactions`
   Kafka topic for downstream ML processing

Usage
─────
  # One-time seed (100 records from last 90 days)
  python -m backend.ingestion.seed

  # Custom date range and record count
  python -m backend.ingestion.seed --start 2024-01-01 --end 2024-12-31 --limit 500

  # Continuous polling mode (runs every N seconds)
  python -m backend.ingestion.seed --watch

Sprint 1 Milestone Check
──────────────────────────
After running this script you can verify the database via psql:
  SELECT v.name, v.risk_tier, COUNT(t.id) AS tx_count
  FROM vendors v
  LEFT JOIN transactions t ON t.vendor_id = v.id
  GROUP BY v.id
  ORDER BY tx_count DESC
  LIMIT 20;
"""

from __future__ import annotations

import argparse
import asyncio
import re
import sys
import unicodedata
from decimal import Decimal
from typing import Dict, List, Optional

from loguru import logger
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from backend.config import get_settings
from backend.database import get_sync_session
from backend.ingestion.kafka_producer import ProcurementKafkaProducer
from backend.ingestion.usaspending_client import USASpendingClient, USASpendingRecord
from backend.models import Transaction, TransactionSource, Vendor

settings = get_settings()

# ─────────────────────────────────────────────────────────────────────────────
# Logging setup
# ─────────────────────────────────────────────────────────────────────────────

logger.remove()
logger.add(
    sys.stderr,
    level=settings.LOG_LEVEL,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
    colorize=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# Vendor normalisation
# ─────────────────────────────────────────────────────────────────────────────

def _normalize_vendor_name(name: str) -> str:
    """
    Produce a canonical lowercase identifier for deduplication.

    Examples:
      "IBM Corp."          → "ibm corp"
      "I.B.M. Corporation" → "ibm corporation"
      "AMAZON.COM, INC."   → "amazoncom inc"
    """
    # Unicode normalise
    name = unicodedata.normalize("NFKD", name)
    # Lowercase
    name = name.lower()
    # Remove punctuation except spaces
    name = re.sub(r"[^\w\s]", " ", name)
    # Collapse whitespace
    name = re.sub(r"\s+", " ", name).strip()
    return name


# ─────────────────────────────────────────────────────────────────────────────
# Upsert helpers
# ─────────────────────────────────────────────────────────────────────────────

def _upsert_vendor(session, record: USASpendingRecord) -> Vendor:
    """
    Get or create a Vendor row for the given record.

    Deduplication strategy:
      1. Match by `duns_number` (UEI) if available – most precise
      2. Match by `normalized_name` – handles slight name variation
      3. Create new vendor if no match found

    Returns the persisted Vendor instance.
    """
    vendor: Optional[Vendor] = None

    # Priority 1: match by UEI/DUNS
    if record.vendor_uei:
        vendor = session.scalars(
            select(Vendor).where(Vendor.duns_number == record.vendor_uei)
        ).first()

    # Priority 2: match by normalized name
    if vendor is None:
        normalized = _normalize_vendor_name(record.vendor_name)
        vendor = session.scalars(
            select(Vendor).where(Vendor.normalized_name == normalized)
        ).first()

    # Priority 3: create new
    if vendor is None:
        vendor = Vendor(
            name=record.vendor_name,
            normalized_name=_normalize_vendor_name(record.vendor_name),
            duns_number=record.vendor_uei,
            country=record.vendor_country,
        )
        session.add(vendor)
        session.flush()  # Get the generated id without committing
        logger.debug("Created new vendor: {}", vendor.name)
    else:
        # Update fields that may have changed
        if record.vendor_country and not vendor.country:
            vendor.country = record.vendor_country

    return vendor


def _transaction_exists(session, external_id: str) -> bool:
    """Return True if a transaction with this external_id already exists."""
    return session.scalars(
        select(Transaction.id).where(Transaction.external_id == external_id)
    ).first() is not None


def _build_transaction(record: USASpendingRecord, vendor: Vendor) -> Transaction:
    """Build a Transaction ORM object from a USASpendingRecord."""
    return Transaction(
        vendor_id=vendor.id,
        external_id=record.external_id,
        source=TransactionSource.USASPENDING,
        amount=Decimal(str(record.amount)).quantize(Decimal("0.01")),
        currency="USD",
        date=record.action_date,
        fiscal_year=record.fiscal_year,
        category=record.category,
        award_type=record.award_type,
        description=record.description,
        awarding_agency=record.awarding_agency,
        raw_data=record.raw_data,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Vendor spend aggregation
# ─────────────────────────────────────────────────────────────────────────────

def _update_vendor_spend_ytd(session) -> None:
    """
    Recalculate `total_spend_ytd` for all vendors in a single pass.
    Called once at the end of a seed batch.
    """
    from sqlalchemy import func, extract
    from datetime import datetime, timezone

    current_year = datetime.now(timezone.utc).year

    # Aggregate spend per vendor for the current calendar year
    rows = session.execute(
        select(
            Transaction.vendor_id,
            func.sum(Transaction.amount).label("ytd_spend"),
        )
        .where(
            extract("year", Transaction.date) == current_year
        )
        .group_by(Transaction.vendor_id)
    ).all()

    for vendor_id, ytd_spend in rows:
        vendor = session.get(Vendor, vendor_id)
        if vendor:
            vendor.total_spend_ytd = ytd_spend or Decimal("0.00")

    logger.info("Updated YTD spend for {} vendors.", len(rows))


# ─────────────────────────────────────────────────────────────────────────────
# Core seeding logic
# ─────────────────────────────────────────────────────────────────────────────

async def seed_batch(
    records: List[USASpendingRecord],
    publish_to_kafka: bool = True,
) -> Dict[str, int]:
    """
    Persist a batch of USASpendingRecords to PostgreSQL.

    Returns
    -------
    dict with counts: {"inserted": N, "skipped": N, "vendors_created": N}
    """
    inserted = 0
    skipped = 0
    kafka_payloads: List[Dict] = []

    with get_sync_session() as session:
        for record in records:
            # Skip records with invalid amounts
            if record.amount <= 0:
                skipped += 1
                continue

            # Skip if already persisted (idempotency)
            if record.external_id and _transaction_exists(session, record.external_id):
                skipped += 1
                logger.debug("Skipping duplicate: {}", record.external_id)
                continue

            vendor = _upsert_vendor(session, record)
            tx = _build_transaction(record, vendor)
            session.add(tx)
            session.flush()  # Get tx.id for Kafka payload

            inserted += 1

            # Queue Kafka payload (publish after commit for consistency)
            kafka_payloads.append({
                "transaction_id": str(tx.id),
                "vendor_id": str(vendor.id),
                "vendor_name": vendor.name,
                "amount": float(tx.amount),
                "currency": tx.currency,
                "date": tx.date.isoformat() if tx.date else None,
                "category": tx.category,
                "description": tx.description,
                "source": tx.source.value,
            })

        # Update YTD spend aggregates in the same transaction
        if inserted > 0:
            _update_vendor_spend_ytd(session)

        # session commit happens via get_sync_session() context manager

    logger.info(
        "Seed batch complete | inserted={} skipped={}",
        inserted, skipped,
    )

    # Publish to Kafka AFTER successful DB commit
    if publish_to_kafka and kafka_payloads:
        await _publish_to_kafka(kafka_payloads)

    return {"inserted": inserted, "skipped": skipped}


async def _publish_to_kafka(payloads: List[Dict]) -> None:
    """Publish transaction payloads to the raw_transactions topic."""
    try:
        async with ProcurementKafkaProducer() as producer:
            for payload in payloads:
                await producer.publish_raw_transaction(
                    transaction_id=payload["transaction_id"],
                    payload=payload,
                )
        logger.info("Published {} transactions to Kafka.", len(payloads))
    except Exception as exc:
        # Kafka publish failure is non-fatal for Sprint 1
        # (Kafka may not be running yet during initial DB setup)
        logger.warning(
            "Kafka publish failed (non-fatal in Sprint 1): {}. "
            "Start Redpanda via docker-compose to enable streaming.",
            exc,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Polling loop (--watch mode)
# ─────────────────────────────────────────────────────────────────────────────

async def poll_continuously(interval_seconds: int, limit_per_cycle: int) -> None:
    """
    Continuously pull new transactions every `interval_seconds`.
    Runs a polling-based near real-time ingestion stream.
    """
    logger.info(
        "Starting continuous polling | interval={}s limit_per_cycle={}",
        interval_seconds, limit_per_cycle,
    )
    cycle = 0
    async with USASpendingClient() as client:
        while True:
            cycle += 1
            logger.info("─── Poll cycle {} ───", cycle)
            try:
                records = await client.fetch_recent_transactions(
                    days_back=1,   # Only look back 1 day in watch mode
                    max_records=limit_per_cycle,
                )
                await seed_batch(records, publish_to_kafka=True)
            except Exception as exc:
                logger.error("Poll cycle {} failed: {}", cycle, exc)

            logger.info(
                "Sleeping {}s until next poll...", interval_seconds
            )
            await asyncio.sleep(interval_seconds)


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

async def main(args: argparse.Namespace) -> None:
    logger.info("=== Procurement Audit System – Data Seeder ===")
    logger.info("Environment: {} | DB: {}", settings.APP_ENV, settings.POSTGRES_DB)

    if args.watch:
        await poll_continuously(
            interval_seconds=settings.USASPENDING_PULL_INTERVAL_SECONDS,
            limit_per_cycle=settings.USASPENDING_MAX_RECORDS_PER_PULL,
        )
    else:
        # One-shot seed – fetch 1..pages pages, dedup on external_id
        all_records: List[USASpendingRecord] = []
        seen_ids: set = set()
        async with USASpendingClient() as client:
            for page_num in range(1, args.pages + 1):
                logger.info("Fetching page {}/{} (limit={})…", page_num, args.pages, args.limit)
                page_records = await client.fetch_records(
                    page=page_num,
                    limit=args.limit,
                    start_date=args.start,
                    end_date=args.end,
                )
                if not page_records:
                    logger.info("No more records on page {} – stopping early.", page_num)
                    break
                new = [r for r in page_records if r.external_id not in seen_ids]
                seen_ids.update(r.external_id for r in new)
                all_records.extend(new)
                logger.info("Page {} → {} new records (total so far: {})", page_num, len(new), len(all_records))

        records = all_records
        if not records:
            logger.warning("No records returned from USAspending API. Check date range.")
            return

        logger.info("Fetched {} total records from USAspending. Seeding database…", len(records))
        stats = await seed_batch(records, publish_to_kafka=not args.no_kafka)

        logger.info(
            "✓ Seed complete | inserted={} skipped={}",
            stats["inserted"], stats["skipped"],
        )
        logger.info(
            "\nSprint 1 Milestone: Run this SQL to verify:\n"
            "  SELECT v.name, v.risk_tier, COUNT(t.id) AS tx_count\n"
            "  FROM vendors v\n"
            "  LEFT JOIN transactions t ON t.vendor_id = v.id\n"
            "  GROUP BY v.id ORDER BY tx_count DESC LIMIT 20;"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Seed the Procurement Audit PostgreSQL database from USAspending.gov"
    )
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="Start date YYYY-MM-DD (default: 90 days ago)",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="End date YYYY-MM-DD (default: today)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=settings.USASPENDING_MAX_RECORDS_PER_PULL,
        help="Records per page (max 100 per USAspending API, default: %(default)s)",
    )
    parser.add_argument(
        "--pages",
        type=int,
        default=1,
        help="Number of pages to fetch (e.g. --pages 10 fetches up to 1000 records)",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Enable continuous polling mode (runs every USASPENDING_PULL_INTERVAL_SECONDS)",
    )
    parser.add_argument(
        "--no-kafka",
        action="store_true",
        help="Disable Kafka publishing (useful when Redpanda is not running)",
    )

    parsed = parser.parse_args()
    asyncio.run(main(parsed))
