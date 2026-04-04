"""
backend/ingestion/deep_seed.py
──────────────────────────────────────────────────────────────────────────────
Deep multi-sector seeder for ML training data richness.

Fetches 50,000–100,000 real U.S. federal procurement transactions spread
across 8 industry verticals and 7 fiscal years (2019–2025). This builds
the vendor transaction history depth required for:

    - Velocity spike detection  (needs many txns per vendor)
    - Split-billing detection   (needs consecutive small invoices)
    - Behavioral profiling      (needs months of per-vendor history)
    - BURST_24H / BURST_48H     (needs high transaction density)
    - Diverse vendor types      (IT, construction, consulting, etc.)

Architecture
────────────
  • 8 industry sectors × 7 year windows = 56 query combinations
  • Each combination: up to MAX_PAGES_PER_COMBO pages (100 records each)
  • CONCURRENCY sector-year combinations run simultaneously
  • Within each combo, pages are fetched sequentially (early-stop on empty)
  • Deduplication by external_id in-memory + DB upsert uses ON CONFLICT
  • Progress saved to data/deep_seed_progress.json — resumable

Usage
─────
  # Aim for 50k records (default)
  python -m backend.ingestion.deep_seed

  # Aim for 100k records
  python -m backend.ingestion.deep_seed --target 100000

  # Higher concurrency (more parallel API calls, watch rate limits)
  python -m backend.ingestion.deep_seed --concurrency 8

  # Resume a previously interrupted run
  python -m backend.ingestion.deep_seed --resume

  # Dry run — print plan without hitting DB
  python -m backend.ingestion.deep_seed --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from loguru import logger

from backend.config import get_settings
from backend.ingestion.seed import seed_batch
from backend.ingestion.usaspending_client import USASpendingClient, USASpendingRecord

settings = get_settings()

# ── Logging ────────────────────────────────────────────────────────────────────
logger.remove()
logger.add(
    sys.stderr,
    level=settings.LOG_LEVEL,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
    colorize=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# Industry Sector Definitions  (8 verticals, 5 NAICS codes each)
# ─────────────────────────────────────────────────────────────────────────────

INDUSTRY_SECTORS: Dict[str, List[str]] = {
    # Software & cloud vendors — large recurring contracts, high velocities
    "information_technology": [
        "541511",  # Custom Computer Programming Services
        "541512",  # Computer Systems Design Services
        "541513",  # Computer Facilities Management Services
        "541519",  # Other Computer Related Services
        "518210",  # Data Processing, Hosting and Related Services
    ],
    # Civil contractors — large contracts split into milestones (split-billing)
    "construction": [
        "236220",  # Commercial and Institutional Building Construction
        "237110",  # Water and Sewer Line and Related Structures
        "237310",  # Highway, Street, Bridge Construction
        "238210",  # Electrical Contractors and Other Wiring Installation
        "238990",  # All Other Specialty Trade Contractors
    ],
    # Management/strategy consultants — weekend submissions, round numbers
    "professional_consulting": [
        "541611",  # Administrative Management and General Mgmt Consulting
        "541613",  # Marketing Consulting Services
        "541620",  # Environmental Consulting Services
        "541690",  # Other Scientific and Technical Consulting Services
        "541990",  # All Other Professional, Scientific, Technical Services
    ],
    # Defense/aerospace equipment — near-threshold purchase orders
    "manufacturing_equipment": [
        "332710",  # Machine Shops
        "332999",  # All Other Miscellaneous Fabricated Metal Manufacturing
        "334111",  # Electronic Computer Manufacturing
        "334511",  # Search, Detection, Navigation Instruments
        "336413",  # Other Aircraft Parts and Equipment Manufacturing
    ],
    # Medical suppliers and pharma — high-volume, consistent amounts
    "healthcare_medical": [
        "325412",  # Pharmaceutical Preparations Manufacturing
        "339112",  # Surgical and Medical Instrument Manufacturing
        "541714",  # R&D in Biotechnology
        "621498",  # All Other Outpatient Care Centers
        "334510",  # Electromedical and Electrotherapeutic Apparatus
    ],
    # Logistics companies — burst invoicing around delivery clusters
    "logistics_transport": [
        "484110",  # General Freight Trucking, Local
        "484121",  # General Freight Trucking, Long-Distance, TL
        "488510",  # Freight Transportation Arrangement
        "492110",  # Couriers and Express Delivery Services
        "541614",  # Process, Physical Distribution and Logistics Consulting
    ],
    # Architecture/engineering firms — long time-series, project-based billing
    "engineering_scientific": [
        "541310",  # Architectural Services
        "541330",  # Engineering Services
        "541360",  # Geophysical Surveying and Mapping Services
        "541380",  # Testing Laboratories and Services
        "541712",  # Research and Development in Physical, Engineering Sciences
    ],
    # Facilities/O&M — high-frequency small invoices (split-billing risk)
    "facilities_maintenance": [
        "561210",  # Facilities Support Services
        "561720",  # Building Cleaning and Maintenance Services
        "561730",  # Landscaping Services
        "238220",  # Plumbing, Heating, Air-Conditioning Contractors
        "238350",  # Finish Carpentry Contractors
    ],
}

# 7 annual windows — gives cross-year behavioral patterns per vendor
YEAR_WINDOWS: List[Tuple[str, str]] = [
    ("2019-01-01", "2019-12-31"),
    ("2020-01-01", "2020-12-31"),
    ("2021-01-01", "2021-12-31"),
    ("2022-01-01", "2022-12-31"),
    ("2023-01-01", "2023-12-31"),
    ("2024-01-01", "2024-12-31"),
    ("2025-01-01", "2025-03-31"),  # partial year
]

# ── Tuning constants ───────────────────────────────────────────────────────────
MAX_PAGES_PER_COMBO = 100   # 100 pages × 100 records = 10,000 max per combo
RECORDS_PER_PAGE    = 100   # USASpending hard cap
INSERT_BATCH_SIZE   = 500   # rows to accumulate before flushing to DB
PAGE_SLEEP_SECS     = 0.35  # polite delay between pages (≤1000 req/hr)

PROGRESS_FILE = Path("data/deep_seed_progress.json")


# ─────────────────────────────────────────────────────────────────────────────
# Progress tracking  (resumable runs)
# ─────────────────────────────────────────────────────────────────────────────

def _load_progress() -> Dict:
    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    if PROGRESS_FILE.exists():
        try:
            return json.loads(PROGRESS_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {
        "completed_combos": [],
        "total_inserted": 0,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }


def _save_progress(progress: Dict) -> None:
    PROGRESS_FILE.write_text(
        json.dumps(progress, indent=2), encoding="utf-8"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Per-combo async worker
# ─────────────────────────────────────────────────────────────────────────────

async def _seed_combo(
    sector_name: str,
    naics_codes: List[str],
    start_date: str,
    end_date: str,
    max_pages: int,
    semaphore: asyncio.Semaphore,
    global_seen: Set[str],
    seen_lock: asyncio.Lock,
    dry_run: bool,
) -> Dict:
    """
    Fetch all pages for one (sector, year) combination and insert into DB.

    Uses the shared semaphore to limit concurrency.
    Within each combo, pages are fetched sequentially so we can stop early
    when the API returns no more results.
    """
    combo_id = f"{sector_name}|{start_date[:4]}"
    inserted = 0
    fetched  = 0
    batch: List[USASpendingRecord] = []

    async with semaphore:
        async with USASpendingClient() as client:
            for page in range(1, max_pages + 1):
                try:
                    records = await client.fetch_records(
                        page=page,
                        limit=RECORDS_PER_PAGE,
                        start_date=start_date,
                        end_date=end_date,
                        naics_codes=naics_codes,
                    )
                except Exception as exc:
                    logger.warning("[{}] page {} error: {} — skipping combo", combo_id, page, exc)
                    break

                if not records:
                    # API returned nothing; no more pages exist for this combo
                    logger.debug("[{}] empty page {} — stopping combo", combo_id, page)
                    break

                # Deduplicate against global in-memory seen set
                async with seen_lock:
                    new_records = [
                        r for r in records
                        if r.external_id and r.external_id not in global_seen
                    ]
                    for r in new_records:
                        global_seen.add(r.external_id)

                fetched += len(new_records)
                batch.extend(new_records)

                # Flush batch to DB when it reaches the batch size threshold
                if len(batch) >= INSERT_BATCH_SIZE and not dry_run:
                    stats = await seed_batch(batch, publish_to_kafka=False)
                    inserted += stats["inserted"]
                    batch = []
                    logger.info(
                        "[{}] flushed batch — combo_inserted={} page={}",
                        combo_id, inserted, page,
                    )

                await asyncio.sleep(PAGE_SLEEP_SECS)

        # Flush any remaining records
        if batch and not dry_run:
            stats = await seed_batch(batch, publish_to_kafka=False)
            inserted += stats["inserted"]

    logger.success(
        "[{}] complete | fetched_new={} db_inserted={}",
        combo_id, fetched, inserted,
    )
    return {"combo_id": combo_id, "fetched": fetched, "inserted": inserted}


# ─────────────────────────────────────────────────────────────────────────────
# Main orchestrator
# ─────────────────────────────────────────────────────────────────────────────

async def main(args: argparse.Namespace) -> None:
    logger.info("=== Deep Multi-Sector Seeder ===")
    logger.info(
        "Target: {:,} records | Concurrency: {} | Dry-run: {}",
        args.target, args.concurrency, args.dry_run,
    )

    # Build full combo list
    all_combos = [
        (sector_name, naics_codes, start, end)
        for sector_name, naics_codes in INDUSTRY_SECTORS.items()
        for start, end in YEAR_WINDOWS
    ]
    total_combos     = len(all_combos)
    theoretical_max  = total_combos * MAX_PAGES_PER_COMBO * RECORDS_PER_PAGE

    logger.info(
        "Plan: {} sectors × {} year windows = {} combos | "
        "theoretical max: {:,} records",
        len(INDUSTRY_SECTORS), len(YEAR_WINDOWS), total_combos, theoretical_max,
    )

    if args.dry_run:
        logger.info("Dry-run — no DB writes. Exiting.")
        for sector_name, naics_codes, start, end in all_combos:
            logger.info("  {:35s} {} → {}", f"{sector_name}|{start[:4]}", start, end)
        return

    # Load progress for resume mode
    progress          = _load_progress() if args.resume else {"completed_combos": [], "total_inserted": 0}
    completed_set: Set[str] = set(progress.get("completed_combos", []))

    remaining = [
        c for c in all_combos
        if f"{c[0]}|{c[2][:4]}" not in completed_set
    ]
    skipped_count = total_combos - len(remaining)
    if skipped_count:
        logger.info("Resuming — skipping {} already-completed combos", skipped_count)

    logger.info("Launching {} combo tasks (concurrency={})…", len(remaining), args.concurrency)

    # Shared state
    semaphore   = asyncio.Semaphore(args.concurrency)
    global_seen: Set[str] = set()
    seen_lock   = asyncio.Lock()

    total_inserted = int(progress.get("total_inserted", 0))
    target_reached = False

    # Create all tasks upfront so we can cancel stragglers if target is hit
    tasks = [
        asyncio.create_task(
            _seed_combo(
                sector_name=sector,
                naics_codes=naics,
                start_date=start,
                end_date=end,
                max_pages=MAX_PAGES_PER_COMBO,
                semaphore=semaphore,
                global_seen=global_seen,
                seen_lock=seen_lock,
                dry_run=args.dry_run,
            )
        )
        for sector, naics, start, end in remaining
    ]

    try:
        for future in asyncio.as_completed(tasks):
            result = await future
            combo_id = result["combo_id"]
            inserted_this = result.get("inserted", 0)
            total_inserted += inserted_this

            completed_set.add(combo_id)
            progress["completed_combos"] = list(completed_set)
            progress["total_inserted"]   = total_inserted
            _save_progress(progress)

            logger.info(
                "Progress: {:>3}/{} combos | total_inserted={:,} | target={:,}",
                len(completed_set), total_combos, total_inserted, args.target,
            )

            if total_inserted >= args.target:
                logger.success(
                    "Target of {:,} reached ({:,} inserted). Stopping remaining tasks.",
                    args.target, total_inserted,
                )
                target_reached = True
                break
    finally:
        # Cancel any tasks still waiting/running
        cancelled = 0
        for t in tasks:
            if not t.done():
                t.cancel()
                cancelled += 1
        if cancelled:
            await asyncio.gather(*tasks, return_exceptions=True)
            logger.info("Cancelled {} remaining tasks.", cancelled)

    logger.success(
        "=== Deep Seed Complete === | inserted={:,} | target={:,} | target_reached={}",
        total_inserted, args.target, target_reached,
    )
    logger.info(
        "Run `python -m backend.ml.trainer` to retrain on the expanded dataset."
    )


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Deep multi-sector seeder — fetches 50k–100k real procurement records.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--target",
        type=int,
        default=50_000,
        help="Stop after this many new DB rows inserted (e.g. 50000 or 100000)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=5,
        help="Number of simultaneous sector-year combos (each uses one HTTP connection)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip combos already recorded in data/deep_seed_progress.json",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the seeding plan without making any API calls or DB writes",
    )
    parsed = parser.parse_args()
    asyncio.run(main(parsed))
