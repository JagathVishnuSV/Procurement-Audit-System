"""
backend/rag/seed_contracts.py
──────────────────────────────────────────────────────────────────────────────
CLI seeder that ingests the 5 sample contracts into the FAISS index.

Usage
-----
    python -m backend.rag.seed_contracts

    # Optionally clear existing index first:
    python -m backend.rag.seed_contracts --reset
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from typing import Optional

# ---------------------------------------------------------------------------
# Standalone DB bootstrap (so we can run outside uvicorn)
# ---------------------------------------------------------------------------
from backend.database import AsyncSessionLocal
from backend.models.vendor import Vendor
from backend.rag.clm_service import CLMService
from backend.rag.sample_contracts import SAMPLE_CONTRACTS
from sqlalchemy import select

# Map sample contract keys → NAICS-ish description for vendor lookup
CATEGORY_HINTS: dict[str, str] = {
    "apex_it":    "Information Technology",
    "buildright": "Construction",
    "medsupply":  "Medical",
    "logitrans":  "Logistics",
    "cloudscale": "Consulting",
}


async def _pick_vendor(session, hint: str) -> Optional[int]:
    """Return the first vendor whose name or category loosely matches *hint*."""
    keyword = f"%{hint.split()[0]}%"
    result = await session.execute(
        select(Vendor.id, Vendor.name)
        .where(Vendor.name.ilike(keyword))
        .limit(1)
    )
    row = result.first()
    if row:
        return row.id

    # Fallback: any vendor
    result = await session.execute(select(Vendor.id).limit(1))
    row = result.first()
    return row.id if row else None


async def seed(reset: bool = False) -> None:
    from backend.rag.vector_store import get_vector_store

    clm = CLMService()
    vs = get_vector_store()

    if reset:
        print("⚠  Resetting FAISS index …")
        vs.reset()
        vs.save()

    async with AsyncSessionLocal() as session:
        for key, meta in SAMPLE_CONTRACTS.items():
            print(f"\n→ Ingesting [{meta['title']}] …")

            vendor_hint = CATEGORY_HINTS.get(key, "")
            vendor_id = await _pick_vendor(session, vendor_hint)
            if vendor_id is None:
                print(f"   ⚠  No vendor found for hint '{vendor_hint}', skipping.")
                continue

            contract = await clm.ingest_text(
                text=meta["text"],
                vendor_id=vendor_id,
                title=meta["title"],
                session=session,
            )
            await session.commit()
            print(
                f"   ✅  Contract #{contract.id}  |  "
                f"chunks={contract.chunk_count}  |  "
                f"indexed={contract.is_indexed}"
            )

    vs.save()
    vs_info = get_vector_store()
    print(
        f"\n✅  Seeding complete.  FAISS total vectors: {vs_info.total_vectors}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed sample contracts into FAISS")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Clear the FAISS index before seeding",
    )
    args = parser.parse_args()

    asyncio.run(seed(reset=args.reset))


if __name__ == "__main__":
    main()
